#!/usr/bin/env python
"""
Score *chosen* vs *rejected* answers with a reward model and plot distributions
===============================================================================

Given a **JSONL** file produced by `generate_responses.py`, this script

1. Computes a scalar *reward* for every `chosen` answer and every `rejected`
   answer using a trained reward model checkpoint (SARM / baseline / plain).
2. Saves a **JSONL** alongside the original with two new fields:
   * `chosen_reward`
   * `rejected_reward`
3. Draws a density plot (or overlaid histograms) comparing the two reward
   distributions and saves it as PNG.

Example usage
-------------
```bash
python score_plot.py \
    --data_path gpt4o_responses.jsonl \
    --model_path /ckpt/llama3_rm-SARM \
    --plot_path reward_dist.png \
    --output_jsonl scored.jsonl \
    --sarm_base_model llama
```
"""

import argparse
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Iterator

import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local optional imports (only if available – for SARM forward)
try:
    from sae import get_last_assistant_masks
    from sarm_llama import LlamaSARM, LlamaBaseline
    from sarm_gemma2 import Gemma2SARM, Gemma2Baseline
except ImportError:
    get_last_assistant_masks = None
    LlamaSARM = LlamaBaseline = Gemma2SARM = Gemma2Baseline = None

# ---------------------------------------------------------------------------
# Utility --------------------------------------------------------------------

def stream_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_single_input(tokenizer, question: str, answer: str, max_length: int):
    """Return dict of tokenized tensors ready for model."""
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if tokenizer.bos_token:
        rendered = rendered.replace(tokenizer.bos_token, "")
    tok = tokenizer(rendered, truncation=True, max_length=max_length)
    inputs = {
        "input_ids": torch.tensor([tok["input_ids"]]),
        "attention_mask": torch.tensor([tok["attention_mask"]]),
    }
    if get_last_assistant_masks is not None:
        inputs["assistant_masks"] = torch.tensor([get_last_assistant_masks(tok["input_ids"])])
    return inputs

# ---------------------------------------------------------------------------
def parse_sarm_param(filename:str):
    latent = int(re.search(r"_Latent(\d+)_", filename).group(1))
    layer = int(re.search(r"_Layer(\d+)_", filename).group(1))
    k_value = int(re.search(r"_K(\d+)_", filename).group(1))

    assert "SARM" in filename or "Baseline" in filename 
    assert "TopK" in filename or "woTopK" in filename
    sarm_use_baseline = "Baseline" in filename
    sarm_use_topk = "TopK" in filename

    print(f"  SARM Parameters:  ".center(60, '='))
    print(f"Latent: {latent}")                
    print(f"Layer: {layer}")                  
    print(f"K: {k_value}")                    
    print(f"sarm_use_topk: {sarm_use_topk}")  

    ret = {
        "sae_latent_size": latent,
        "sae_hidden_state_source_layer": layer,
        "sae_k": k_value,
        'sarm_use_topk': sarm_use_topk,
    }
    if sarm_use_baseline: 
        ret.pop('sae_k')
        ret.pop('sarm_use_topk')
    return ret


def main():
    parser = argparse.ArgumentParser("Score chosen/rejected answers and plot distributions")
    parser.add_argument("--data_path", required=True, help="JSONL file with question, chosen, rejected")
    parser.add_argument("--model_path", required=True, help="Reward model checkpoint")
    parser.add_argument("--tokenizer_path", help="Separate tokenizer repo (defaults to model path)")
    parser.add_argument("--output_jsonl", default="scored.jsonl")
    parser.add_argument("--plot_path", default="reward_dist.png")
    parser.add_argument("--sarm_base_model", choices=["llama", "gemma2", "none"], default="none")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ----------------- load tokenizer -----------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = args.max_length

    # ----------------- load model ---------------------
    model_cls = None
    if args.sarm_base_model == "llama" and LlamaSARM is not None:
        model_cls = LlamaSARM
    elif args.sarm_base_model == "gemma2" and Gemma2SARM is not None:
        model_cls = Gemma2SARM
    if model_cls is None:
        model_cls = AutoModelForSequenceClassification

    sarm_params = parse_sarm_param(args.model_path)
    model = model_cls.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        **sarm_params
    ).to(args.device)
    model.eval()

    # ----------------- iterate rows -------------------
    chosen_scores: List[float] = []
    rejected_scores: List[float] = []

    out_path = Path(args.output_jsonl).expanduser().resolve()
    with out_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(stream_jsonl(Path(args.data_path)), desc="Scoring"):
            qid = row.get("question_id")
            question = row["question"]
            chosen_ans = row["chosen"]
            rejected_ans = row["rejected"]

            # score chosen
            with torch.no_grad():
                inp_c = build_single_input(tokenizer, question, chosen_ans, args.max_length)
                inp_c = {k: v.to(args.device) for k, v in inp_c.items()}
                out_c = model(**inp_c)
                score_c = out_c["logits"].squeeze().item()

                inp_r = build_single_input(tokenizer, question, rejected_ans, args.max_length)
                inp_r = {k: v.to(args.device) for k, v in inp_r.items()}
                out_r = model(**inp_r)
                score_r = out_r["logits"].squeeze().item()

            chosen_scores.append(score_c)
            rejected_scores.append(score_r)

            row.update({"chosen_reward": score_c, "rejected_reward": score_r})
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✓ Scores written to {out_path} – plotting …")

    # ----------------- plot distributions -------------
    plt.figure()
    # Use histogram density
    plt.hist(chosen_scores, bins=50, density=True, alpha=0.6, label="chosen (refusal)")
    plt.hist(rejected_scores, bins=50, density=True, alpha=0.6, label="rejected (answer)")
    plt.xlabel("Reward (logit)")
    plt.ylabel("Density")
    plt.title("Reward distribution: chosen vs rejected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=300)
    print(f"✓ Plot saved to {args.plot_path}")


if __name__ == "__main__":
    main()
