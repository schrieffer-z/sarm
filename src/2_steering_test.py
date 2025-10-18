import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Iterator

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def stream_jsonl(path: Path) -> Iterator[dict]:
    """Reads a .jsonl file line by line."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_conv(tok, q: str, a: str, max_len: int, device: str) -> dict:
    """Builds a conversation input for the model from a question and an answer."""
    messages = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tok(text, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    
def get_sequence_lengths(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Calculates the sequence lengths for a batch of input_ids, excluding padding."""
    sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
    sequence_lengths = sequence_lengths % input_ids.shape[-1]
    return sequence_lengths

def sample_data(data_path: Path, num_samples: int) -> list:
    """Randomly samples a specified number of data points."""
    all_data = list(stream_jsonl(data_path))
    if len(all_data) <= num_samples:
        return all_data
    return random.sample(all_data, num_samples)

def plot_kde(before: List[float], after: List[float], out_path: str):
    """Plots the Kernel Density Estimate for scores before and after steering."""
    xs = np.linspace(min(before + after), max(before + after), 512)
    kde_b, kde_a = gaussian_kde(before)(xs), gaussian_kde(after)(xs)

    plt.figure(figsize=(6, 4))
    plt.hist(before, bins=50, density=True, alpha=.4, label="before", color="tab:blue")
    plt.hist(after, bins=50, density=True, alpha=.4, label="after", color="tab:orange")
    plt.plot(xs, kde_b, lw=2, color="tab:blue")
    plt.plot(xs, kde_a, lw=2, color="tab:orange")
    plt.xlabel("Reward (logit)")
    plt.ylabel("Density")
    plt.title("Reward distribution: before vs after steering")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    p = argparse.ArgumentParser(description="Steer and score a SARM model using hooks.")
    p.add_argument("--data_path", required=True, help="JSONL with question/answer")
    p.add_argument("--num_example", type=int, default=450)
    p.add_argument("--model_path", required=True)
    p.add_argument("--tokenizer_path")
    p.add_argument("--steering_path", required=True, help="JSON file: {latent_id:[action,val], ...}")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--output_path", default="scored_steering.json")
    p.add_argument("--plot_path", default="reward_steering.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # --- Tokenizer ---
    tok_dir = args.tokenizer_path or args.model_path
    tok = AutoTokenizer.from_pretrained(tok_dir)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.truncation_side = "left"

    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(args.device).eval()

    # --- Steering and Hook Setup ---
    with open(args.steering_path, encoding="utf-8") as f:
        steer_after = {int(k): v for k, v in json.load(f).items()}
    
    # This dictionary will be modified in the loop to control the hook's behavior.
    current_steering_dict = {}
    captured_features = []

    def steering_pre_hook(module, model_input):
        """
        A forward pre-hook to intercept and modify the sae_features tensor
        before it is passed to the scoring head.
        """
        # The sae_features tensor is the first element of the input tuple.
        sae_features = model_input[0]

        if current_steering_dict:
            for latent, (action, val) in current_steering_dict.items():
                latent_idx = int(latent)
                if action == '+':
                    sae_features[:, :, latent_idx] += val
                elif action == '*':
                    # The original logic is equivalent to multiplication.
                    sae_features[:, :, latent_idx] *= val
                else:
                    raise ValueError(f'Unsupported steering action: {action}')
        
        # After modification (or not), capture the features for analysis.
        captured_features.append(sae_features.detach().clone())

    # Register a pre-hook. It runs *before* the module's forward pass.
    hook_handle = model.score.register_forward_pre_hook(steering_pre_hook)
    
    # --- Scoring Loop ---
    try:
        before_scores, after_scores = [], []
        output_data = {}
        
        sampled_data = sample_data(Path(args.data_path), args.num_example)
        for row in tqdm(sampled_data, desc="Scoring"):
            q, a = row["question"], row["answer"] # Adjusted keys to match common formats

            # --- Score Before Steering ---
            captured_features.clear()
            current_steering_dict = {} # No steering applied
            enc = build_conv(tok, q, a, args.max_length, args.device)
            with torch.no_grad():
                s_b = model(**enc).logits.item()
            
            latents_b = captured_features[0]
            seq_len_b = get_sequence_lengths(enc['input_ids'], tok.pad_token_id)
            last_token_latents_b = latents_b[torch.arange(latents_b.shape[0]), seq_len_b].squeeze()
            
            before_scores.append(s_b)

            # --- Score After Steering ---
            captured_features.clear()
            current_steering_dict = steer_after # Apply steering
            with torch.no_grad():
                s_a = model(**enc).logits.item()

            after_scores.append(s_a)
            
            # --- Analyze Activated Latents ---
            activated_latents = [
                latent for latent in steer_after.keys()
                if last_token_latents_b[latent].item() != 0
            ]
            
            output_data[row.get("id", len(output_data))] = {
                "reward": s_b,
                "reward_steered": s_a,
                "activated_latents": activated_latents,
                "question": q,
                "answer": a,
            }

        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"✓ results saved → {args.output_path}")

        plot_kde(before_scores, after_scores, args.plot_path)
        print(f"✓ plot saved    → {args.plot_path}")

    finally:
        # --- Cleanup ---
        hook_handle.remove()
        print("Forward pre-hook removed.")

if __name__ == "__main__":
    main()
