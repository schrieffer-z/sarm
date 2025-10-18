import argparse
import json
from pathlib import Path
from typing import Iterator

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def stream_jsonl(path: Path) -> Iterator[dict]:
    """
    Reads a .jsonl file line by line.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_conv(tok, q: str, a: str, max_len: int, device: str) -> dict:
    """
    Builds a conversation input for the model from a question and an answer.
    """
    # This chat template is based on Llama-3-Instruct format.
    # Adjust if your model uses a different template.
    messages = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return tok(
        text,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    ).to(device)

def get_sequence_lengths(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Calculates the sequence lengths for a batch of input_ids, excluding padding tokens.
    This replicates the logic from the original SARM model's forward pass.
    """
    # Find the first occurrence of the pad token.
    # The sequence length is the index of the token just before the final EOT token,
    # which is one position before the first padding token.
    sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
    # If no pad token is found, argmax returns 0, resulting in -1.
    # The modulo operation handles this case, correctly pointing to the last element.
    sequence_lengths = sequence_lengths % input_ids.shape[-1]
    return sequence_lengths

def main():
    """
    Main function to run the inference and scoring process.
    """
    p = argparse.ArgumentParser(description="Extract and score latent features from a SARM model.")
    p.add_argument("--data_path", type=str, required=True, help="Path to the input .jsonl data file.")
    p.add_argument("--tokenizer_path", type=str, help="Optional path to the tokenizer, defaults to model_path.")
    p.add_argument("--model_path", type=str, required=True, help="Path to the pretrained SARM model.")
    p.add_argument("--output_file", type=str, default="s.pt", help="Path to save the output tensor.")
    p.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length for the tokenizer.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    args = p.parse_args()

    # --- Tokenizer ---
    tok_dir = args.tokenizer_path or args.model_path
    tok = AutoTokenizer.from_pretrained(tok_dir)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # --- Model ---
    # Load the model using AutoModelForSequenceClassification.
    # `trust_remote_code=True` is essential to load the custom LlamaSARM architecture.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=1, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2", 
    ).to(args.device).eval()

    # --- Hook Setup ---
    # This list will store the tensor captured by the hook.
    captured_features = []

    def get_sae_latents_hook(module, model_input, model_output):
        """
        A forward hook that captures the final latent features from the SAE.
        We hook the SAE's internal encoder. The model's main forward pass then
        adds a bias and may apply a top-k activation. To get the final features
        that are sent to the scoring head, we must replicate that logic here.
        """
        # model_output is the raw tensor from the encoder's linear layer.
        # 1. Add the latent bias to get the pre-activations.
        pre_acts = model_output.detach() + model.sae.latent_bias

        # 2. Check if the model uses top-k activation and apply it if so.
        # This mirrors the logic in the LlamaSARM.forward() method.
        if model.sarm_use_activation:
            final_features = model.sae.get_latents(pre_acts)
        else:
            final_features = pre_acts
        
        captured_features.append(final_features)

    # Register the hook on the encoder layer within the SAE module.
    # This aligns with the concept of getting features from the SAE, but requires
    # replicating some logic from the model's forward pass inside the hook.
    hook_handle = model.sae.encoder.register_forward_hook(get_sae_latents_hook)

    # --- Scoring Loop ---
    try:
        sae_latent_size = model.config.sarm_param['sae_latent_size']
        chosen_latents_sum = torch.zeros(sae_latent_size, device="cpu")
        rejected_latents_sum = torch.zeros(sae_latent_size, device="cpu")

        for row in tqdm(stream_jsonl(Path(args.data_path)), desc="Scoring"):
            question, chosen, rejected = row["prompt"], row["chosen"], row["rejected"]

            # --- Process 'chosen' response ---
            captured_features.clear()
            enc_c = build_conv(tok, question, chosen, args.max_length, args.device)
            with torch.no_grad():
                model(**enc_c)
            
            # The hook has now captured the sae_features for the entire sequence.
            sae_features_c = captured_features[0]
            # Find the position of the last non-padding token.
            sequence_lengths_c = get_sequence_lengths(enc_c['input_ids'], tok.pad_token_id)
            # Extract the latent features for that specific token.
            last_token_latent_c = sae_features_c[
                torch.arange(sae_features_c.shape[0], device=args.device),
                sequence_lengths_c
            ].squeeze()
            chosen_latents_sum += last_token_latent_c.cpu()

            # --- Process 'rejected' response ---
            captured_features.clear()
            enc_j = build_conv(tok, question, rejected, args.max_length, args.device)
            with torch.no_grad():
                model(**enc_j)
            
            sae_features_j = captured_features[0]
            sequence_lengths_j = get_sequence_lengths(enc_j['input_ids'], tok.pad_token_id)
            last_token_latent_j = sae_features_j[
                torch.arange(sae_features_j.shape[0], device=args.device),
                sequence_lengths_j
            ].squeeze()
            rejected_latents_sum += last_token_latent_j.cpu()

        # --- Final Calculation ---
        # This formula is preserved from your original script.
        c, j = chosen_latents_sum, rejected_latents_sum
        denominator = c + j
        # Add a small epsilon to the denominator to avoid division by zero
        score = (c - j) / (denominator + denominator.mean() + 1e-9)
        
        torch.save(score, args.output_file)
        print(f"Latent scores saved to {args.output_file}")

    finally:
        # --- Cleanup ---
        # It's crucial to remove the hook when it's no longer needed to prevent memory leaks.
        hook_handle.remove()
        print("Forward hook removed.")


if __name__ == "__main__":
    main()

