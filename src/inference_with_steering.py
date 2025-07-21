'''

python src/inference_with_steering.py \
    --steering_path steering.json \
    --data_path data/16720/target.jsonl \
    --tokenizer_path models/Llama-3.1-8B-Instruct_sequence_Latent65536_Layer16_K192_1B-SARM-TopK-LastToken-1/   \
    --model_path models/Llama-3.1-8B-Instruct_sequence_Latent65536_Layer16_K192_1B-SARM-TopK-LastToken-1/checkpoint-150 \
    --sarm_base_model llama \
    --output_jsonl scored.jsonl \
    --device cuda:0

steering.json:
{
    "16720": ["*", 2], 
    "12222": ["*", 3],
    ...
}

'''
#!/usr/bin/env python

import argparse, torch, json, os, re
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Union, Tuple

import torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from sae import pre_process
from sarm_llama  import LlamaSARM
from sarm_gemma2 import Gemma2SARM



class LlamaSARM4Steering(LlamaSARM):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        steering: Optional[dict]=None,
        # Shuyi (aggregate latent)
        assistant_masks: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]


        # Shuyi
        h, _, _ = pre_process(hidden_states)
        sae_features = self.sae.pre_acts(h)
        sae_features_steering = sae_features
        if self.sarm_use_topk:
            sae_features = self.sae.get_latents(sae_features)
        if steering is not None:
            k = sae_features_steering
            sae_features_steering = self.sae.get_latents(sae_features_steering)
            for latent in steering.keys():
                action, val = steering[latent]
                if action == '+':
                    sae_features[:, :, latent] += torch.ones_like(sae_features[:, :, latent])*val
                elif action == '*':
                    sae_features[:, :, latent] += sae_features[:, :, latent]*(val-1)
                else:
                    raise ValueError(f'unsupported action {action}')

        logits = self.score(sae_features)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
        # Shuyi (查看last_token是否为<|eot_id|>)
        assert ((input_ids[torch.arange(batch_size, device=logits.device), sequence_lengths]!=torch.ones(batch_size, device=logits.device)*128009).sum() == 0).item()
        
        # Shuyi (联合训练)
        rec_loss = None
        if self.sarm_train_mode==2:
            if not self.sarm_use_topk:
                sae_features_t = self.sae.get_latents(sae_features)
            h_hat = self.sae.decode(sae_features_t)
            rec_loss = Masked_Normalized_MSE_loss(h, h_hat, assistant_masks)
        elif self.sarm_train_mode==3 and not self.sae_use_sequence_level:
            h_d = h.detach()
            _, h_hat = self.sae(h_d)
            rec_loss = Masked_Normalized_MSE_loss(h_d, h_hat, assistant_masks)        
        elif self.sarm_train_mode==3 and self.sae_use_sequence_level:
            h_d = h.detach()
            sequence_lengths_t = sequence_lengths.view(-1,1,1)
            last_token_mask = torch.zeros([h_d.shape[0] ,1 ,h_d.shape[1]], device=h_d.device)
            last_token_mask.scatter_(-1, sequence_lengths_t, torch.ones_like(sequence_lengths_t, dtype=last_token_mask.dtype))
            
            # h_d -> (bs, seq_len, d), last_token_mask -> (bs, 1, seq_len)
            h_d = torch.matmul(last_token_mask.to(h_d.dtype), h_d) 
            
            _, h_hat = self.sae(h_d)
            rec_loss = Normalized_MSE_loss(h_d, h_hat)       


        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]


        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)
        if rec_loss is not None:
            loss = rec_loss

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

# ---------- utils ------------------------------------------------------------
def stream_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def parse_sarm_param(name: str) -> Dict:
    latent  = int(re.search(r"_Latent(\d+)_", name).group(1))
    layer   = int(re.search(r"_Layer(\d+)_",  name).group(1))
    k       = int(re.search(r"_K(\d+)_",      name).group(1))
    kw = {"sae_latent_size": latent,
          "sae_hidden_state_source_layer": layer,
          "sae_k": k,
          "sarm_use_topk": "TopK" in name}
    if "Baseline" in name:
        kw.pop("sae_k"); kw.pop("sarm_use_topk")
    return kw

def build_conv(tok, q: str, a: str, max_len: int, device):
    text = tok.apply_chat_template(
        [{"role":"user","content":q},{"role":"assistant","content":a}],
        tokenize=False, add_generation_prompt=False)
    return tok(text, truncation=True, max_length=max_len,
               return_tensors="pt").to(device)

def plot_kde(before: List[float], after: List[float], out_path: str):
    xs = np.linspace(min(before+after), max(before+after), 512)
    kde_b, kde_a = gaussian_kde(before)(xs), gaussian_kde(after)(xs)

    plt.figure(figsize=(6,4))
    plt.hist(before, bins=50, density=True, alpha=.4, label="before",  color="tab:blue")
    plt.hist(after,  bins=50, density=True, alpha=.4, label="after",   color="tab:orange")
    plt.plot(xs, kde_b, lw=2, color="tab:blue")
    plt.plot(xs, kde_a, lw=2, color="tab:orange")
    # plt.fill_between(xs, kde_a, kde_b, where=kde_a>kde_b, alpha=.2,
    #                  color="tab:orange", label="after > before")
    plt.xlabel("Reward (logit)"); plt.ylabel("Density")
    plt.title("Reward distribution: before vs after steering")
    plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


# ---------- main -------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",      required=True, help="JSONL with question/answer")
    p.add_argument("--model_path",     required=True)
    p.add_argument("--tokenizer_path")
    p.add_argument("--sarm_base_model", choices=["llama","gemma2",None], default=None)
    p.add_argument("--steering_path",  required=True,
                   help="JSON file: {latent_id:[action,val], …}")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--output_jsonl", default="scored_steering.jsonl")
    p.add_argument("--plot_path",     default="reward_steering.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # ---------- tokenizer ----------
    tok_dir = args.tokenizer_path or args.model_path
    tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=False)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token":"[PAD]"})
    tok.truncation_side = "left"; tok.model_max_length = args.max_length

    # ---------- model ----------
    if args.sarm_base_model == "llama":
        Model = LlamaSARM4Steering
    else:
        raise ValueError(f'Unsupport model type {args.sarm_base_model}')
    sarm_param = parse_sarm_param(args.model_path)
    model = Model.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, **sarm_param
    ).to(args.device).eval()

    # ---------- steering dict ----------
    with open(args.steering_path, encoding="utf-8") as f:
        steer_after = {int(k): v for k, v in json.load(f).items()}
    # before = 同样 action，但数值统一替换为 1
    steer_before = {k: [v[0], 1.0] for k, v in steer_after.items()}

    def score(q, a, steer):
        enc = build_conv(tok, q, a, args.max_length, args.device)
        return model(**enc, steering=steer)["logits"].item()

    # ---------- 逐条计算 ----------
    before_scores, after_scores = [], []
    out_jsonl = Path(args.output_jsonl).resolve().open("w", encoding="utf-8")

    for row in tqdm(stream_jsonl(Path(args.data_path)), desc="Scoring"):
        q, a = row["question"], row["answer"]
        s_b = score(q, a, steer_before)
        s_a = score(q, a, steer_after)
        before_scores.append(s_b); after_scores.append(s_a)
        row.update({"reward_before": s_b, "reward_after": s_a})
        out_jsonl.write(json.dumps(row, ensure_ascii=False)+"\n")
    out_jsonl.close()
    print("✓ scored jsonl →", args.output_jsonl)

    # ---------- 绘图 ----------
    plot_kde(before_scores, after_scores, args.plot_path)
    print("✓ plot saved   →", args.plot_path)


if __name__ == "__main__":
    main()
