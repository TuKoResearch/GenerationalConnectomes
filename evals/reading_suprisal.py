#!/usr/bin/env python
"""
Evaluates a single HuggingFace model on the behavioral reading-time surprisal correlation task.
Usage:
    python surprisal_eval.py --model_name <HF_repo_id> [--tokenizer_name gpt2] [--device cuda:0]
Prints the Pearson correlation (r) between GPT surprisal and reading times.
"""
import os
import json
import xarray as xr
import numpy as np
import torch
import argparse
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, GPT2TokenizerFast

# ------------------------------
# Constants
# ------------------------------
DATA_FILE = os.path.join('data', 'assy_Futrell2018.nc')

# ------------------------------
# Helper functions
# ------------------------------
def compute_token_surprisal(text, model, tokenizer, device):
    encoding = tokenizer(
        text,
        return_tensors='pt',
        return_offsets_mapping=True,
    )
    input_ids = encoding['input_ids'].to(device)
    offsets = encoding['offset_mapping'][0]
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits']
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    token_surprisals = -token_log_probs.squeeze(0)
    token_offsets = [(int(start), int(end)) for (start, end) in offsets[1:]]
    return token_surprisals.cpu().numpy(), token_offsets


def get_word_boundaries(words_list):
    boundaries = []
    pos = 0
    for word in words_list:
        start = pos
        end = start + len(word)
        boundaries.append((start, end))
        pos = end + 1
    return boundaries


def assign_tokens_to_words(token_offsets, word_boundaries):
    token_to_word = [-1] * len(token_offsets)
    for i, (t_start, t_end) in enumerate(token_offsets):
        if t_end <= t_start:
            continue
        t_mid = (t_start + t_end) / 2.0
        for j, (w_start, w_end) in enumerate(word_boundaries):
            if w_start <= t_mid < w_end:
                token_to_word[i] = j
                break
    word_to_tokens = [[] for _ in range(len(word_boundaries))]
    for tok_idx, word_idx in enumerate(token_to_word):
        if word_idx != -1:
            word_to_tokens[word_idx].append(tok_idx)
    return word_to_tokens


def compute_word_surprisal(token_surprisal, word_to_tokens):
    word_surprisal = []
    for toks in word_to_tokens:
        word_surprisal.append(np.sum(token_surprisal[toks]) if toks else np.nan)
    return word_surprisal


def evaluate_surprisal_correlation(model, tokenizer, device):
    # Load reading-time data
    data = xr.open_dataset(DATA_FILE)
    response = data.data.values
    valid = np.sum(~np.isnan(response), axis=0) > 6
    avg_rts = np.nanmean(response[:, valid], axis=1)
    words = data.word.values
    stories = data.story_id.values

    all_surprisal = []
    all_rts = []
    for story in np.unique(stories):
        idxs = np.where(stories == story)[0]
        words_list = [str(words[i]) for i in idxs]
        if not words_list:
            continue
        text = " ".join(words_list)
        boundaries = get_word_boundaries(words_list)
        token_surprisal, token_offsets = compute_token_surprisal(text, model, tokenizer, device)
        word_to_tokens = assign_tokens_to_words(token_offsets, boundaries)
        word_surprisal = compute_word_surprisal(token_surprisal, word_to_tokens)
        all_surprisal.extend(word_surprisal)
        all_rts.extend(avg_rts[idxs].tolist())

    all_surprisal = np.array(all_surprisal)
    all_rts = np.array(all_rts)
    mask = ~np.isnan(all_surprisal) & ~np.isnan(all_rts)
    r, p = pearsonr(all_surprisal[mask], all_rts[mask])
    return r, p


# ------------------------------
# Main CLI
# ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True,
                        help='HuggingFace model repo ID (e.g., TuKoResearch/ConnectomeGPT100M)')
    parser.add_argument('-t', '--tokenizer_name', type=str, default='gpt2',
                        help='HuggingFace tokenizer name or repo ID')
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help='Compute device (e.g., cuda:0 or cpu)')
    args = parser.parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_name)
    model.eval()
    model.to(args.device)

    # Evaluate correlation
    r, p = evaluate_surprisal_correlation(model, tokenizer, args.device)
    print(f"Model: {args.model_name} | Pearson r: {r:.4f} | p-value: {p:.4g}")
