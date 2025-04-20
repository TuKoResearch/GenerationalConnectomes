#!/usr/bin/env python
"""
Evaluates a GPT model from HuggingFace on the HellaSwag benchmark.
https://github.com/rowanz/hellaswag

Inspired by and adapted from c.llm: https://github.com/karpathy/llm.c by Andrej Karpathy
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
from torch.nn import functional as F
import numpy as np
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given URL."""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

# ---------------------------------------------------------------------
HEADERS_INFO = {
    "gpt-2": {
        "magic": 20240520,
        "version": 1,
        "token_dtype": np.uint16,
    },
    "llama-3": {
        "magic": 20240801,
        "version": 7,
        "token_dtype": np.uint32,
    },
}

# write_datafile and write_evalfile unchanged from original
def write_datafile(filename, toks, model_desc="gpt-2"):
    assert len(toks) < 2**31, "token count too large"
    assert model_desc in ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = HEADERS_INFO[model_desc]
    header = np.zeros(256, dtype=np.int32)
    header[0] = info["magic"]
    header[1] = info["version"]
    header[2] = len(toks)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    num_bytes = (256 * 4) + (len(toks) * toks_np.itemsize)
    print(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def write_evalfile(filename, datas):
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240522
    header[1] = 1
    header[2] = len(datas)
    header[3] = 0
    longest_example_bytes = 0
    full_stream = []
    assert len(datas) < 2**16, "too many examples?"
    for idx, data in enumerate(datas):
        stream = []
        stream.append(2**16-1)
        stream.append(0)
        stream.append(idx)
        stream.append(data["label"])
        ending_tokens = data["ending_tokens"]
        assert len(ending_tokens) == 4, "expected 4 completions"
        stream.append(len(ending_tokens))
        ctx_tokens = data["ctx_tokens"]
        assert all(0 <= t < 2**16-1 for t in ctx_tokens), "bad context token"
        stream.append(len(ctx_tokens))
        stream.extend(ctx_tokens)
        for end_tokens in ending_tokens:
            assert all(0 <= t < 2**16-1 for t in end_tokens), "bad completion token"
            stream.append(len(end_tokens))
            stream.extend(end_tokens)
        nbytes = len(stream) * 2
        assert nbytes < 2**16, "example too large?"
        stream[1] = nbytes
        longest_example_bytes = max(longest_example_bytes, nbytes)
        full_stream.extend(stream)
    stream_np = np.array(full_stream, dtype=np.uint16)
    header[3] = longest_example_bytes
    print(f"writing {len(datas):,} examples to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(stream_np.tobytes())

# ---------------------------------------------------------------------
def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

def iterate_examples(split):
    download(split)
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    with open(data_filename, "r") as f:
        for line in f:
            yield json.loads(line)

def render_example(example):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

@torch.no_grad()
def evaluate(model, device):
    torch.set_float32_matmul_precision('high')
    model.to(device)
    model.eval()

    datas = []
    num_correct = 0
    num_correct_norm = 0
    num_total = 0

    for example in tqdm(iterate_examples("val"), desc="evaluating", total=10042):
        data, tokens, mask, label = render_example(example)
        datas.append(data)
        tokens = tokens.to(device)
        mask = mask.to(device)

        if tokens.size(1) > model.config.block_size:
            chunk_size = model.config.block_size
            overlap = 0.9
            logits = None
            for chunk_start in range(0, tokens.size(1), int(chunk_size * (1 - overlap))):
                chunk_end = min(chunk_start + chunk_size, tokens.size(1))
                outputs = model(tokens[:, chunk_start:chunk_end])
                chunk_logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
                if logits is None:
                    logits = chunk_logits
                else:
                    chunk_to_add = min(tokens.size(1) - logits.size(1), int(chunk_size * (1 - overlap)))
                    logits = torch.cat((logits, chunk_logits[:, -chunk_to_add:]), dim=1)
                if chunk_end == tokens.size(1):
                    break
        else:
            logits = model(tokens)['logits']

        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)

        shift_mask = mask[..., 1:].contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

    eval_filename = os.path.join(DATA_CACHE_DIR, "hellaswag_val.bin")
    if not os.path.exists(eval_filename):
        write_evalfile(eval_filename, datas)

    acc = num_correct / num_total
    acc_norm = num_correct_norm / num_total
    return acc, acc_norm

# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="the device to use")
    parser.add_argument("-b", "--model_name", type=str, required=True, help="the model name from HuggingFace")
    parser.add_argument("-t", "--tokenizer_name", type=str, default="gpt2", help="the tokenizer name from HuggingFace")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    acc, acc_norm = evaluate(model, args.device)
    print(f"Model: {args.model_name} | Accuracy: {acc:.4f} | Accuracy_norm: {acc_norm:.4f}")
