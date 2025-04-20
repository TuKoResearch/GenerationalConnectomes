import os
import time
import math
from contextlib import nullcontext
import argparse
import numpy as np
import torch
import wandb
import importlib
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group

from distributed_dataloader import TokenDistributedDataLoader
from config import GPTConfig
from model import GPT


def eval_model(model, dataloader, device, max_iters=100):
    val_loss = []
    dataloader.reset()

    with torch.no_grad():
        for _ in range(max_iters):
            x_in, x_tgt = dataloader.next_batch()
            output = model(x_in, labels=x_tgt)  # model returns dict or tuple
            # handle both styles
            if isinstance(output, dict):
                loss = output.get('loss')
            else:
                _, loss = output
            val_loss.append(loss.item())

    return float(np.mean(val_loss))


def main():
    ### GET ARGUMENTS ###
    parser = argparse.ArgumentParser()
    # Run parameters
    parser.add_argument("--run_name", type=str, 
                        default="test")
                        # required=True)
    parser.add_argument("--train_data_dir", type=str, 
                        default="/ccn2a/dataset/fineweb/fineweb-10b/*.bin")
                        # required=True)
    parser.add_argument("--val_data_dir", type=str, 
                        default="/ccn2a/dataset/fineweb/fineweb-val/*.bin")
                        # required=True)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--save_frequency", default=1000, type=int)
    parser.add_argument("--eval_frequency", default=250, type=int)
    parser.add_argument("--log_frequency", default=10, type=int)
    parser.add_argument("--wandb", default=False, action="store_true")
    parser.add_argument("--push_to_hf", default=False, action="store_true",
                        help="Push trained model to Hugging Face hub after training")
    parser.add_argument('--optimizer', type=str, default='adamw')

    # Initialization parameters
    parser.add_argument("--init_from", default="model_generation_5.pt", type=str)
    parser.add_argument("--init_method", default="init_pruned", type=str)
    parser.add_argument("--init_alpha", default=0.2, type=float)
    parser.add_argument("--fully_train_embs", default=False, action="store_true")
    parser.add_argument('--random_seed', type=int, default=1110)

    # Optimizer/scheduler parameters
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--lr", default=0.0018, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--decay_type", default="hold", choices=["cosine", "hold"], type=str)
    parser.add_argument("--num_iters", default=2001, type=int)
    parser.add_argument("--warmdown_iters", default=1750, type=int)
    parser.add_argument("--warmup_iters", default=250, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--dtype", default='bfloat16' if torch.cuda.is_available() \
                        and torch.cuda.is_bf16_supported() else 'float16', type=str)

    # Hardware parameters
    parser.add_argument("--accelerator_type", default="A40", type=str)
    parser.add_argument("--per_device_batch_size", default=16, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument("--compile", default=False, action="store_true")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--backend", default="nccl", type=str)

    args = parser.parse_args()

    # ------------------ Initialize distributed training ------------------
    gradient_accumulation_steps = args.batch_size // args.per_device_batch_size
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=args.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = args.device
        ddp_rank = 0
        ddp_world_size = 1

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Seeds and TF32 config
    torch.manual_seed(args.random_seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---------------------- Load model config ----------------------
    cfg = GPTConfig()
    cfg.block_size = args.max_seq_length

    # ----------------------- DataLoaders -----------------------
    train_dataloader = TokenDistributedDataLoader(
        args.train_data_dir, args.per_device_batch_size, args.max_seq_length, ddp_rank, ddp_world_size
    )
    if master_process:
        train_eval_dataloader = TokenDistributedDataLoader(
            args.train_data_dir, args.per_device_batch_size, args.max_seq_length, 0, 1
        )
        val_dataloader = TokenDistributedDataLoader(
            args.val_data_dir, args.per_device_batch_size, args.max_seq_length, 0, 1
        )

    tokens_per_iter = args.batch_size * train_dataloader.T
    print(f"tokens per iteration will be: {tokens_per_iter:,}, sequence length: {train_dataloader.T}")

    # ---------------------- Initialize model ----------------------
    model = GPT(cfg)

    # ----------------------- Sparsity map -----------------------
    sparse_mask = {}
    if args.init_from is not None:
        checkpoint = torch.load(args.init_from, map_location=device, weights_only=False)
        # support legacy keys
        ckpt_state = checkpoint.get('model_state_dict', checkpoint.get('weights', checkpoint))
        checkpoint_params = ckpt_state
        # remove _orig_mod. prefix
        for key in list(ckpt_state.keys()):
            if key.startswith("module."):
                new_key = key.replace("module.", "")
                ckpt_state[new_key] = ckpt_state.pop(key)
            if key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "")
                ckpt_state[new_key] = ckpt_state.pop(key)

        # init_pruned or other methods
        if args.init_method == "init_pruned":
            missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
            if master_process:
                print(f"Loaded checkpoint from {args.init_from}.")
                if missing:
                    print(f"Missing keys in loaded state_dict: {missing}")
                if unexpected:
                    print(f"Unexpected keys in loaded state_dict: {unexpected}")
            model.to(device)
            tol = 1e-5
            for name, param in model.named_parameters():
                simple_name = name.replace("module.", "").replace("_orig_mod.", "").lower()
                if args.fully_train_embs and any(x in simple_name for x in ["wte", "wpe", "lm_head"]):
                    mask = torch.ones_like(param, device=device)
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                else:
                    with torch.no_grad():
                        data_abs = param.data.abs()
                        zero_mask = (data_abs < tol)
                        pos_mask = (param.data > 0) & ((param.data - 0.02).abs() < tol)
                        neg_mask = (param.data < 0) & ((param.data + 0.02).abs() < tol)
                        mask = torch.zeros_like(param, device=device)
                        mask[pos_mask] = 1.0
                        mask[neg_mask] = 1.0
                        param.data[zero_mask] = 0.0
                        param.data[pos_mask] = 0.02
                        param.data[neg_mask] = -0.02
                sparse_mask[simple_name] = mask
            if master_process:
                pruned_count = sum((mask == 0).sum().item() for mask in sparse_mask.values())
                total_count = sum(param.numel() for param in model.parameters())
                fraction_pruned = 100.0 * pruned_count / total_count if total_count > 0 else 0.0
                print(f"Percentage of pruned parameters (excluding wte/wpe/lm_head): {fraction_pruned:.2f}%")
        else:
            # other initialization methods can follow similar pattern
            for (name, param) in model.named_parameters():
                sparse_mask[name] = torch.ones_like(param).to(device)
    else:
        for (name, param) in model.named_parameters():
            sparse_mask[name] = torch.ones_like(param).to(device)

    model.to(device)

    # ----------------------- Optimizer -----------------------
    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (args.beta1, args.beta2), device_type, optimizer=args.optimizer
    )

    # ----------------------- Compile -----------------------
    if args.compile:
        print("compiling the model...")
        unoptimized_model = model
        model = torch.compile(model)

    torch.cuda.empty_cache()

    # ----------------------- DDP wrap -----------------------
    if ddp:
        model = DDP(model, device_ids=[int(device.split(':')[-1])])

    output_path = os.path.join("out", args.run_name)
    if master_process:
        os.makedirs(output_path, exist_ok=True)

    raw_model = model.module if hasattr(model, "module") else model

    # ----------------------- Weights & Biases -----------------------
    if master_process and args.wandb:
        wandb.init(project="ConnectomePruning", entity="NaudioGPT",
                   name=args.run_name, config=vars(args))
        wandb.define_metric("iter")
        wandb.define_metric("*", step_metric="iter")

    # ----------------------- LR Scheduler -----------------------
    def get_lr(it):
        if it < args.warmup_iters:
            return args.lr * it / args.warmup_iters
        if it > args.num_iters - args.warmdown_iters:
            return args.lr - (args.lr - args.min_lr) * (
                it - (args.num_iters - args.warmdown_iters)) / args.warmdown_iters
        if args.decay_type == "hold":
            return args.lr
        elif args.decay_type == "cosine":
            decay_ratio = (it - args.warmup_iters) / (args.warmdown_iters - args.warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return args.min_lr + coeff * (args.lr - args.min_lr)

    best_val_loss = float('inf')
    running_mfu = -1.0
    x_in, x_tgt = train_dataloader.next_batch()
    t0 = time.time()

    # ----------------------- Main training loop -----------------------
    for it in range(args.num_iters):
        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if master_process and it % args.eval_frequency == 0:
            train_loss = eval_model(model, train_eval_dataloader, device)
            val_loss = eval_model(model, val_dataloader, device)
            if args.wandb:
                wandb.log({
                    "train/loss": train_loss,
                    "validation/loss": val_loss,
                    "lr": lr,
                    "iter": it,
                    "tokens": tokens_per_iter * it,
                })
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving best model at iteration {it} with loss {best_val_loss:.4f}")
                best_ckpt = {
                    'model_state_dict': raw_model.state_dict(),
                    'iteration': it,
                    'tokens': tokens_per_iter * it,
                    'best_val_loss': best_val_loss,
                    'cfg': cfg.to_dict(),
                    'args': vars(args),
                }
                torch.save(best_ckpt, os.path.join(output_path, "model_best.pt"))
            # Periodic save
            if it % args.save_frequency == 0:
                ckpt = {
                    'model_state_dict': raw_model.state_dict(),
                    'iteration': it,
                    'tokens': tokens_per_iter * it,
                    'best_val_loss': best_val_loss,
                    'cfg': cfg.to_dict(),
                    'args': vars(args),
                }
                torch.save(ckpt, os.path.join(output_path, f"model_{it:06d}.pt"))

        # gradient accumulation and backward
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                output = model(x_in, labels=x_tgt)
                if isinstance(output, dict):
                    loss = output['loss']
                else:
                    _, loss = output
                loss = loss / gradient_accumulation_steps
            x_in, x_tgt = train_dataloader.next_batch()
            loss.backward()

        # zero grads for pruned weights
        for name, param in model.named_parameters():
            if param.grad is not None:
                key = name.replace("module.", "").replace("_orig_mod.", "")
                param.grad.mul_(sparse_mask[key])

        if args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # re-zero pruned weights
        for name, param in model.named_parameters():
            with torch.no_grad():
                key = name.replace("module.", "").replace("_orig_mod.", "")
                param.mul_(sparse_mask[key])

        optimizer.zero_grad(set_to_none=True)

        t1 = time.time(); dt = t1 - t0; t0 = t1
        if it % args.log_frequency == 0 and master_process:
            perf_loss = loss.item() * gradient_accumulation_steps
            dW = raw_model.record_power_usage(dt * ddp_world_size)
            if it >= 20:
                mfu = raw_model.estimate_mfu(
                    args.per_device_batch_size * gradient_accumulation_steps,
                    train_dataloader.T, dt,
                    accelerator_type=args.accelerator_type
                )
                running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {it}: loss {perf_loss:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, dW {dW:.2f}dW")

    # ----------------------- Post-training operations -----------------------
    if master_process and args.push_to_hf:
        print("Pushing trained model to Hugging Face hub...")
        raw_model.push_to_hub(repo_name=args.run_name)


if __name__ == "__main__":
    main()
