#!/usr/bin/env python
"""
Evaluates the Generationally Pruned GPT model from HuggingFace on the MMLU benchmark.
https://github.com/hendrycks/test

Inspired by and adapted from c.llm: https://github.com/karpathy/llm.c by Andrej Karpathy
"""

import os
import tiktoken
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F
import argparse
import requests
from transformers import AutoModelForCausalLM, GPT2Tokenizer


DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "mmlu")
data_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
enc = tiktoken.get_encoding("gpt2")


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
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

def write_datafile(filename, toks, model_desc="gpt-2"):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as uint16 (gpt-2) or uint32 (llama)
    """
    assert len(toks) < 2**31, "token count too large"  # ~2.1B tokens
    assert model_desc in ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = HEADERS_INFO[model_desc]
    # construct the header
    header = np.zeros(256, dtype=np.int32)  # header is always 256 int32 values
    header[0] = info["magic"]
    header[1] = info["version"]
    header[2] = len(toks)  # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    num_bytes = (256 * 4) + (len(toks) * toks_np.itemsize)
    print(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def write_evalfile(filename, datas):
    """
    Saves eval data as a .bin file, for reading in C.
    Used for multiple-choice style evals, e.g. HellaSwag and MMLU
    - First comes a header with 256 int32s
    - The examples follow, each example is a stream of uint16_t:
        - <START_EXAMPLE> delimiter of 2**16-1, i.e. 65,535
        - <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
        - <EXAMPLE_INDEX>, the index of the example in the dataset
        - <LABEL>, the index of the correct completion
        - <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
        - <NUM><CONTEXT_TOKENS>, where <NUM> is the number of tokens in the context
        - <NUM><COMPLETION_TOKENS>, repeated NUM_COMPLETIONS times
    """
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240522  # magic
    header[1] = 1  # version
    header[2] = len(datas)  # number of examples
    header[3] = 0  # reserved for longest_example_bytes, fill in later
    # now write the individual examples
    longest_example_bytes = 0  # in units of uint16s
    full_stream = []  # the stream of uint16s, we'll write a single time at the end
    assert len(datas) < 2**16, "too many examples?"
    for idx, data in enumerate(datas):
        stream = []
        # header of the example
        stream.append(2**16-1)  # <START_EXAMPLE>
        stream.append(0)  # <EXAMPLE_BYTES> (fill in later)
        stream.append(idx)  # <EXAMPLE_INDEX>
        stream.append(data["label"])  # <LABEL>
        ending_tokens = data["ending_tokens"]
        assert len(ending_tokens) == 4, "expected 4 completions for now? can relax later"
        stream.append(len(ending_tokens))  # <NUM_COMPLETIONS>
        # the (shared) context tokens
        ctx_tokens = data["ctx_tokens"]
        assert all(0 <= t < 2**16-1 for t in ctx_tokens), "bad context token"
        stream.append(len(ctx_tokens))
        stream.extend(ctx_tokens)
        # the completion tokens
        for end_tokens in ending_tokens:
            assert all(0 <= t < 2**16-1 for t in end_tokens), "bad completion token"
            stream.append(len(end_tokens))
            stream.extend(end_tokens)
        # write to full stream
        nbytes = len(stream)*2  # 2 bytes per uint16
        assert nbytes < 2**16, "example too large?"
        stream[1] = nbytes  # fill in the <EXAMPLE_BYTES> field
        longest_example_bytes = max(longest_example_bytes, nbytes)
        full_stream.extend(stream)
    # construct the numpy array
    stream_np = np.array(full_stream, dtype=np.uint16)
    # fill in the longest_example field
    assert 0 < longest_example_bytes < 2**16, f"bad longest_example"
    header[3] = longest_example_bytes
    # write to file (for HellaSwag val this is 10,042 examples, 3.6MB file)
    print(f"writing {len(datas):,} examples to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(stream_np.tobytes())

def download():
    """Downloads MMLU to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_filename = os.path.join(DATA_CACHE_DIR, "data.tar")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
        os.system(f"tar -xf {data_filename} -C {DATA_CACHE_DIR}")  # untar
        # creates a directory "data" inside it, with e.g. data/test/*csv
    else:
        print(f"{data_filename} already exists, skipping download...")

def iterate_examples():
    # there are 14,042 examples in total in the test set
    download()
    test_dir = os.path.join(DATA_CACHE_DIR, "data", "test")
    csv_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]
    for csv_file in csv_files:
        csv_path = os.path.join(test_dir, csv_file)
        # print(csv_path)
        df = pd.read_csv(csv_path, header=None)
        n = df.shape[0]
        for idx in range(n):
            example = {
                "question": df.iloc[idx, 0],
                "endings": [df.iloc[idx, 1], df.iloc[idx, 2], df.iloc[idx, 3], df.iloc[idx, 4]],
                "label": df.iloc[idx, 5],
            }
            yield example

def render_example(example, tokenizer):
    ctx = f"Question: {example['question']}\n\nAnswer:"
    ctx_tokens = tokenizer.encode(ctx)

    tok_rows = []
    mask_rows = []
    for end in example["endings"]:
        end_tokens = tokenizer.encode(" " + str(end))
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    label = "ABCD".index(example["label"])
    return tokens, mask, label

@torch.no_grad()
def evaluate(model, tokenizer, device):
    torch.set_float32_matmul_precision('high')
    model.to(device)

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in tqdm(iterate_examples(), desc="evaluating", total=14042):
        tokens, mask, label = render_example(example, tokenizer)
        tokens = tokens.to(device)
        mask = mask.to(device)

        if tokens.size(1) > model.config.block_size:
            tokens = tokens[:, -model.config.block_size:]
            mask = mask[:, -model.config.block_size:]

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

    return num_correct / num_total, num_correct_norm / num_total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="the device to use")
    parser.add_argument("-b", "--model_name", type=str, default="TuKoResearch/ConnectomeGPT100M", 
                        help="the model name from HuggingFace")
    parser.add_argument("-t", "--tokenizer_name", type=str, default="gpt2",
                        help="the tokenizer name from HuggingFace")
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    acc, acc_norm = evaluate(model, tokenizer, args.device)
    print(f"Model: {args.model_name} | Accuracy: {acc:.4f} | Accuracy_norm: {acc_norm:.4f}")
