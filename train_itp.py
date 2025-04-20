import os
import time
import math
import glob
from contextlib import nullcontext
import argparse
import numpy as np
import torch
import wandb
import importlib

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group

from .distributed_dataloader import TokenDistributedDataLoader


def eval_model(model, dataloader, device, max_iters=100):
    val_loss = []
    dataloader.reset()

    with torch.no_grad():
        for _ in range(max_iters):
            x_in, x_tgt = dataloader.next_batch()
            x_in, x_tgt = x_in.to(device), x_tgt.to(device)
            logits, loss = model(x_in, targets=x_tgt)
            val_loss.append(loss.item())

    mean_loss = np.mean(val_loss)
    return mean_loss


def prune_bottom_pct_of_weights(model, sparse_mask, pct_to_prune=0.2, global_pruning=False):
    """
    Prunes the bottom pct_to_prune fraction of *remaining unpruned* weights (by absolute value).
    
    - Layers whose name contains 'wte', 'wpe', or 'lm_head' are fully exempt 
      from pruning: all their mask values are set to 1.0.
    - If `global_pruning=True`, a single global threshold is found from all 
      applicable layers combined. Then we prune the fraction `pct_to_prune` 
      across all layers at once.
    - If `global_pruning=False`, we compute and prune that fraction per-layer.
    """

    # (If you want to fully exempt certain layers, uncomment or adjust the code below)
    # for name, param in model.named_parameters():
    #     name = name.replace("module.", "").replace("_orig_mod.", "")
    #     if any(x in name.lower() for x in ["wte", "wpe", "lm_head"]):
    #         if name in sparse_mask and sparse_mask[name] is not None:
    #             sparse_mask[name].fill_(1.0)

    if global_pruning:
        # Global pruning
        all_unpruned_weights = []
        all_unpruned_params = []

        for name, param in model.named_parameters():
            name = name.replace("module.", "").replace("_orig_mod.", "")
            # Skip if not in mask or if mask is None
            if name not in sparse_mask or sparse_mask[name] is None:
                continue
            # Optionally skip 1D parameters (biases, etc.)
            if len(param.shape) == 1:
                continue

            mask = sparse_mask[name]
            unpruned_vals = param.data.abs()[mask.bool() > 0]
            if unpruned_vals.numel() > 0:
                all_unpruned_weights.append(unpruned_vals)
                all_unpruned_params.append((name, param))

        if len(all_unpruned_weights) == 0:
            print("No parameters to prune. Skipping pruning.")
            return sparse_mask

        all_unpruned_weights = torch.cat(all_unpruned_weights)
        total_unpruned = all_unpruned_weights.numel()
        prune_count = int(pct_to_prune * total_unpruned)

        if prune_count < 1:
            print("Prune count < 1; skipping pruning.")
            return sparse_mask

        threshold = torch.topk(all_unpruned_weights, prune_count, largest=False).values[-1]

        for name, param in all_unpruned_params:
            name = name.replace("module.", "").replace("_orig_mod.", "")
            mask = sparse_mask[name]
            to_prune = (param.data.abs() < threshold) & (mask.bool() > 0)
            mask[to_prune] = 0.0

    else:
        # Per-layer pruning
        for name, param in model.named_parameters():
            name = name.replace("module.", "").replace("_orig_mod.", "")
            # Skip if not in mask or if mask is None
            if name not in sparse_mask or sparse_mask[name] is None:
                continue
            # Optionally skip 1D parameters (biases, etc.)
            if len(param.shape) == 1:
                continue

            mask = sparse_mask[name]
            unpruned_vals = param.data.abs()[mask.bool() > 0]

            if unpruned_vals.numel() == 0:
                continue

            total_unpruned = unpruned_vals.numel()
            prune_count = int(pct_to_prune * total_unpruned)

            if prune_count < 1:
                continue

            threshold = torch.topk(unpruned_vals, prune_count, largest=False).values[-1]
            to_prune = (param.data.abs() < threshold) & (mask.bool() > 0)
            mask[to_prune] = 0.0

    return sparse_mask


def reinitialize_unpruned_weights(model, sparse_mask, magnitude=0.02):
    """
    After pruning, re-initialize the surviving weights (mask==1) to +/-magnitude
    depending on their sign before re-initialization.
    """
    for name, param in model.named_parameters():
        name = name.replace("module.", "").replace("_orig_mod.", "")
        if name not in sparse_mask:
            continue
        mask = sparse_mask[name]
        if mask is None:
            continue

        with torch.no_grad():
            signs = torch.sign(param.data)
            param.data[mask.bool() > 0] = signs[mask.bool() > 0] * magnitude


def rewind_unpruned_weights(model, sparse_mask, weights_rewind_dict):
    """
    For each unpruned weight (mask==1), reset the parameter to the stored 
    rewind-point value. 
    """
    for name, param in model.named_parameters():
        name = name.replace("module.", "").replace("_orig_mod.", "")
        if name not in sparse_mask:
            continue
        mask = sparse_mask[name]
        if mask is None:
            continue

        with torch.no_grad():
            param.data[mask.bool() > 0] = weights_rewind_dict[name][mask.bool() > 0]


def main():

    ### GET ARGUMENTS ###
    parser = argparse.ArgumentParser()

    ## Training Run Parameters
    parser.add_argument("--run_name", type=str, default="GPT-100M-1Bclassictokens-10k-perlayerprune-connectomemask")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--train_data_dir", type=str,
                        default="/ccn2a/dataset/fineweb/fineweb-1b/*.bin")
    parser.add_argument("--val_data_dir", type=str,
                        default="/ccn2a/dataset/fineweb/fineweb-val/*.bin")
    parser.add_argument("--save_frequency", default=2000, type=int)
    parser.add_argument("--eval_frequency", default=250, type=int)
    parser.add_argument("--log_frequency", default=10, type=int)
    parser.add_argument("--save_to_gcloud", default=False, action="store_true")
    parser.add_argument("--wandb", default=False, action="store_true")
    parser.add_argument('--model_config', type=str,
        default='ConnectomePruning.GPT100MConfig')
    parser.add_argument('--optimizer', type=str, default='muon')
    
    ## Initialization Parameters
    parser.add_argument("--init_from",
            default=None,
            type=str)
    parser.add_argument("--init_method", default="random_conectome", type=str)
    parser.add_argument("--init_alpha", default=0.1, type=float)
    parser.add_argument('--random_seed', type=int,
                        default=1110)

    ## Optimizer Parameters
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--lr", default=0.0018, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--decay_type", default="hold", choices=["cosine", "hold"], type=str)
    parser.add_argument("--num_iters", default=7001, type=int)
    parser.add_argument("--warmdown_iters", default=1800, type=int)
    parser.add_argument("--warmup_iters", default=250, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--dtype", default='bfloat16' if torch.cuda.is_available() \
        and torch.cuda.is_bf16_supported() else 'float16', type=str)

    ## Hardware Parameters
    parser.add_argument("--accelerator_type", default="A40", type=str)
    parser.add_argument("--per_device_batch_size", default=16, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument("--compile", default=False, action="store_true")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--backend", default="nccl", type=str)

    ## Lottery Ticket Parameters
    parser.add_argument("--prune_pct", default=0.2, type=float)
    parser.add_argument("--rewind_point", default=None, type=int)
    parser.add_argument("--num_generations", default=11, type=int)
    parser.add_argument("--pruning_strategy", default="per-layer", choices=["global", "per-layer"], type=str)

    args = parser.parse_args()

    ### INITIALIZE DISTRIBUTED TRAINING ###
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

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Seeds and TF32 config
    torch.manual_seed(args.random_seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load config class
    module_path, class_name = args.model_config.rsplit('.', 1)
    module = importlib.import_module(module_path)
    init_cfg_cls = getattr(module, class_name)
    cfg = init_cfg_cls()
    cfg.block_size = args.max_seq_length
    module_path, class_name = cfg.model_type.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)
    for key, value in vars(args).items():
        if hasattr(cfg, key) and value is not None:
            setattr(cfg, key, value)

    # Dataloaders
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

    # ----------------------------------------------------------------------------
    # NEW: Check if there's an existing checkpoint for any generation. If so,
    # resume from the highest generation checkpoint found.
    # ----------------------------------------------------------------------------
    checkpoint_files = glob.glob(os.path.join("out", args.run_name, "model_generation_*.pt"))
    if checkpoint_files:
        # Parse out generation numbers from filenames of form "model_generation_X.pt"
        gen_indices = []
        for f in checkpoint_files:
            base = os.path.basename(f)
            # Attempt to parse integer after "model_generation_"
            try:
                # e.g. "model_generation_3.pt" -> "3"
                idx_str = base.split("_")[-1].split(".")[0]
                gen_idx = int(idx_str)
                gen_indices.append((gen_idx, f))
            except ValueError:
                pass

        if gen_indices:
            # Take highest generation checkpoint
            last_gen_idx, ckpt_path = max(gen_indices, key=lambda x: x[0])
            print(f"Found existing checkpoint at {ckpt_path}, generation={last_gen_idx}")
            start_gen_idx = last_gen_idx + 1

            # Build the same model
            model = model_cls(cfg).to(device)

            # Load that checkpoint
            checkpoint = torch.load(ckpt_path, map_location=device)
            weights = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['weights'].items()}
            model.load_state_dict(weights)
            sparse_mask = checkpoint['mask']

            # We do not store or load weights_rewind_dict in the original code, so set it to None
            weights_rewind_dict = None

            # We'll set the global iteration to reflect the end of last_gen_idx
            global_it = last_gen_idx * args.num_iters

            print(f"Resuming from generation {start_gen_idx} ...")

        else:
            # If no valid generation files matched, just proceed normally
            sparse_mask = {}
            model = model_cls(cfg).to(device)
            start_gen_idx = 0
            weights_rewind_dict = None

    else:
        # No generation checkpoints found; do the standard initialization
        sparse_mask = {}
        model = model_cls(cfg).to(device)
        start_gen_idx = 0
        weights_rewind_dict = None

    # ----------------------------------------------------------------------------
    # (Below is the original initialization for `sparse_mask` and optional
    #  init_from logic, executed only if we did NOT load from a generation.)
    # ----------------------------------------------------------------------------
    if start_gen_idx == 0:
        # If we are not resuming from a generation checkpoint, run the usual logic:
        if args.init_from is not None:
            checkpoint_model = ModelFactory().load_model(args.init_from, force_download=False)
            checkpoint_params = dict(checkpoint_model.named_parameters())

            if args.init_method == "largest_final_only_dense":
                for (name, param) in model.named_parameters():
                    if name in checkpoint_params and len(param.shape) == 2:
                        checkpoint_param = checkpoint_params[name]
                        print(f"Initializing layer {name} with largest_final_only_dense")
                        idx = int(args.init_alpha * checkpoint_param.numel())
                        threshold = torch.topk(torch.abs(checkpoint_param).view(-1), idx).values[-1]
                        keep_mask = torch.ge(torch.abs(checkpoint_param), threshold).to(param.device)
                        param.data[~keep_mask] = 0.0
                        param.data[keep_mask] = torch.sign(param.data[keep_mask]) * 0.02
                        sparse_mask[name] = keep_mask.float().to(param.device)
                    else:
                        sparse_mask[name] = torch.ones_like(param).to(param.device)

                del checkpoint_model

            elif args.init_method == "random_conectome":
                for (name, param) in model.named_parameters():
                    if name in checkpoint_params and len(param.shape) == 2:
                        checkpoint_param = checkpoint_params[name]
                        print(f"Initializing layer {name} with random_conectome")
                        param.data = torch.zeros_like(param.data)
                        idx = torch.randperm(checkpoint_param.numel())[:int(args.init_alpha * checkpoint_param.numel())]
                        idx_mask = torch.zeros_like(checkpoint_param).view(-1)
                        idx_mask[idx] = 1
                        idx_mask = idx_mask.view_as(checkpoint_param).bool()
                        # random +/- 0.1
                        param.data[idx_mask] = torch.randint(0, 2, (len(idx),)).float() * 0.2 - 0.1
                        sparse_mask[name] = torch.ones_like(param).to(param.device)
                    else:
                        sparse_mask[name] = torch.ones_like(param).to(param.device)

            else:
                # No special pruning logic
                print(f"No special pruning for init_method = {args.init_method}, or param not found in checkpoint.")
                for (name, param) in model.named_parameters():
                    sparse_mask[name] = torch.ones_like(param).to(param.device)

        else:
            # If no checkpoint init, create a dense mask of all ones
            for (name, param) in model.named_parameters():
                name = name.replace("module.", "").replace("_orig_mod.", "")
                sparse_mask[name] = torch.ones_like(param).to(param.device)

    # Create optimizer
    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (args.beta1, args.beta2),
        device_type, optimizer=args.optimizer
    )

    # Compile if requested
    if args.compile:
        print("compiling the model...")
        unoptimized_model = model
        model = torch.compile(model)

    torch.cuda.empty_cache()

    # Wrap in DDP
    if ddp:
        model = DDP(model, device_ids=[int(device.split(':')[-1])])

    output_path = os.path.join("out", args.run_name)
    if master_process:
        os.makedirs(output_path, exist_ok=True)

    raw_model = model.module if hasattr(model, "module") else model

    # The LR schedule for each generation
    def get_lr(it):
        if it < args.warmup_iters:
            return args.lr * it / args.warmup_iters
        if it > args.num_iters - args.warmdown_iters:
            return args.lr - (args.lr - args.min_lr) * (
                it - (args.num_iters - args.warmdown_iters)) / args.warmdown_iters
        if args.decay_type == "hold":
            return args.lr
        elif args.decay_type == "cosine":
            # Cosine decay from warmup_iters..warmdown_iters
            decay_ratio = (it - args.warmup_iters) / (args.warmdown_iters - args.warmup_iters)
            # clamp just in case
            decay_ratio = max(0, min(1, decay_ratio))
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return args.min_lr + coeff * (args.lr - args.min_lr)

    def train_one_generation(gen_idx, sparse_mask, start_global_it):
        """
        Trains the model for one generation.
        Returns (updated_model, updated_sparse_mask, final_global_iteration).
        """
        nonlocal weights_rewind_dict

        if master_process and args.wandb:
            runname = f"{args.run_name}-generation{gen_idx}"
            wandb_run = wandb.init(
                project="ConnectomePruning",
                entity="NaudioGPT",
                name=runname,
                config=vars(args),
                reinit=True
            )
            wandb.define_metric("iter")
            wandb.define_metric("*", step_metric="iter")
        else:
            wandb_run = None

        # Evaluate at iteration 0 of this generation
        if master_process:
            train_loss_0 = eval_model(model, train_eval_dataloader, device)
            val_loss_0 = eval_model(model, val_dataloader, device)
            if wandb_run:
                wandb.log({
                    "train/loss": train_loss_0,
                    "validation/loss": val_loss_0,
                    "lr": 0.0,
                    "iter": 0,
                    "tokens": 0
                })
            print(f"[Generation {gen_idx}, iter 0] train_loss={train_loss_0:.4f}, val_loss={val_loss_0:.4f}")

        best_val_loss = float('inf')
        running_mfu = -1.0

        x_in, x_tgt = train_dataloader.next_batch()
        x_in, x_tgt = x_in.to(device), x_tgt.to(device)
        t0 = time.time()

        local_it = 0
        for it in range(args.num_iters):
            # Update LR
            lr = get_lr(it)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Evaluate periodically
            if master_process and (it % args.eval_frequency == 0 and it > 0):
                train_loss = eval_model(model, train_eval_dataloader, device)
                val_loss = eval_model(model, val_dataloader, device)
                if wandb_run:
                    wandb.log({
                        "train/loss": train_loss,
                        "validation/loss": val_loss,
                        "lr": lr,
                        "iter": it,
                        "tokens": tokens_per_iter * it
                    })
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"Saving model at iteration {it}, best val_loss so far {best_val_loss:.4f}")
                    raw_model.save(
                        {
                            'weights': raw_model.state_dict(),
                            'iteration': it,
                            "tokens": tokens_per_iter * it,
                            'best_val_loss': best_val_loss,
                            'cfg': cfg_to_dict(cfg),
                            'args': args,
                            'mask': sparse_mask
                        },
                        os.path.join(output_path, "model_best.pt"),
                        gcloud=args.save_to_gcloud
                    )
 
            # Gradient accumulation
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                with ctx:
                    logits, loss = model(x_in, targets=x_tgt)
                    loss = loss / gradient_accumulation_steps
                x_in, x_tgt = train_dataloader.next_batch()
                x_in, x_tgt = x_in.to(device), x_tgt.to(device)

                loss.backward()

            # Zero out grad for pruned weights
            for name, param in model.named_parameters():
                name = name.replace("module.", "").replace("_orig_mod.", "")
                if param.grad is not None:
                    param.grad.mul_(sparse_mask[name])

            if args.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            # Re-zero pruned weights
            for name, param in model.named_parameters():
                name = name.replace("module.", "").replace("_orig_mod.", "")
                with torch.no_grad():
                    param.mul_(sparse_mask[name])

            optimizer.zero_grad(set_to_none=True)

            # Check if we hit rewind_point
            global_iter_now = start_global_it + it
            if args.rewind_point is not None and weights_rewind_dict is None:
                if global_iter_now == args.rewind_point:
                    # store the entire model state for rewinding
                    print(f"Storing rewind weights at global iteration = {global_iter_now}")
                    weights_rewind_dict = {
                        k: v.clone() for k, v in raw_model.state_dict().items()
                    }

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if it % args.log_frequency == 0 and master_process:
                lossf = loss.item() * gradient_accumulation_steps
                dW = raw_model.record_power_usage(dt * ddp_world_size)
                if it >= 20:
                    mfu = raw_model.estimate_mfu(
                        args.per_device_batch_size * gradient_accumulation_steps,
                        train_dataloader.T, dt,
                        accelerator_type=args.accelerator_type
                    )
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

                print(f"[Gen {gen_idx} | iter {it}] loss {lossf:.4f}, time {dt*1000:.2f}ms, "
                      f"mfu {running_mfu*100:.2f}%, dW {dW:.2f}dW")

            local_it = it

        # End-of-generation pruning
        print(f"[Generation {gen_idx}] Pruning {args.prune_pct*100}% of remaining unpruned weights...")
        prune_bottom_pct_of_weights(raw_model, sparse_mask, pct_to_prune=args.prune_pct, 
                                    global_pruning=args.pruning_strategy == "global")

        # Re-initialize surviving weights (or rewind if we have a rewind checkpoint)
        if weights_rewind_dict is not None:
            print(f"[Generation {gen_idx}] Rewinding surviving weights to iteration {args.rewind_point} checkpoint.")
            rewind_unpruned_weights(raw_model, sparse_mask, weights_rewind_dict)
        else:
            print(f"[Generation {gen_idx}] Re-initialize surviving weights to +/-0.02.")
            reinitialize_unpruned_weights(raw_model, sparse_mask, magnitude=0.02)

        # Save final checkpoint for this generation
        if master_process:
            raw_model.save(
                {
                    'weights': raw_model.state_dict(),
                    'iteration': local_it,
                    "tokens": tokens_per_iter * local_it,
                    'best_val_loss': best_val_loss,
                    'cfg': cfg_to_dict(cfg),
                    'args': args,
                    'mask': sparse_mask
                },
                os.path.join(output_path, f"model_generation_{gen_idx}.pt"),
                gcloud=args.save_to_gcloud
            )

        if wandb_run is not None:
            wandb_run.finish()

        return raw_model, sparse_mask, (start_global_it + args.num_iters)

    # If we found a checkpoint, we set global_it accordingly above. Otherwise:
    if start_gen_idx == 0:
        global_it = 0

    # ---------------------------
    # MAIN: run multiple generations
    # ---------------------------
    for gen_idx in range(start_gen_idx, args.num_generations):
        raw_model, sparse_mask, global_it = train_one_generation(gen_idx, sparse_mask, global_it)
        optimizer = raw_model.configure_optimizers(
            args.weight_decay, args.lr, (args.beta1, args.beta2),
            device_type, optimizer=args.optimizer
        )
        if ddp:
            model = DDP(raw_model, device_ids=[int(device.split(':')[-1])])
        else:
            model = raw_model


if __name__ == "__main__":
    main()
