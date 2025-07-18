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
from typing import Dict, List, Iterator, Optional, Union, Tuple

import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

import argparse, json, os, re
from pathlib import Path
from typing import Dict, List, Iterator

import torch, numpy as np, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde          #  ← 用于连续拟合
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local optional imports (only if available – for SARM forward)
try:
    from sae import pre_process, Masked_Normalized_MSE_loss, Normalized_MSE_loss
    from sarm_llama import LlamaSARM, LlamaBaseline
    from sarm_gemma2 import Gemma2SARM, Gemma2Baseline
except ImportError:
    get_last_assistant_masks = None
    LlamaSARM = LlamaBaseline = Gemma2SARM = Gemma2Baseline = None
# ---------- util -------------------------------------------------------------
def stream_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def parse_sarm_param(filename:str):
    latent  = int(re.search(r"_Latent(\d+)_", filename).group(1))
    layer   = int(re.search(r"_Layer(\d+)_",  filename).group(1))
    k_value = int(re.search(r"_K(\d+)_",      filename).group(1))
    baseline = "Baseline" in filename
    topk     = "TopK" in filename
    params = {
        "sae_latent_size": latent,
        "sae_hidden_state_source_layer": layer,
        "sae_k": k_value,
        "sarm_use_topk": topk,
    }
    if baseline:                    # Baseline 不需要 k / topk
        params.pop("sae_k")
        params.pop("sarm_use_topk")
    return params


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
        if self.sarm_use_topk:
            sae_features = self.sae.get_latents(sae_features)
        if steering is not None:
            for latent in steering.keys():
                action, val = steering[latent]
                print(sae_features[:, :, latent])
                if action == '+':
                    sae_features[:, :, latent] += torch.ones_like(sae_features[:, :, latent])*val
                elif action == '*':
                    sae_features[:, :, latent] += sae_features[:, :, latent]*(val-1)

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

# ---------- KDE-绘图 ----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_combined(chosen, rejected, out_path):
    """Overlay hist+KDE for chosen & rejected, and shade KDE_chosen > KDE_rejected."""
    chosen = np.asarray(chosen, dtype=np.float32)
    rejected = np.asarray(rejected, dtype=np.float32)

    xs = np.linspace(min(chosen.min(), rejected.min()),
                     max(chosen.max(), rejected.max()), 512)
    kde_ch = gaussian_kde(chosen)(xs)
    kde_re = gaussian_kde(rejected)(xs)

    plt.figure(figsize=(6,4))

    # ① 直方图（密度）
    plt.hist(chosen,   bins=50, density=True, alpha=0.4, label="chosen",   color="tab:blue")
    plt.hist(rejected, bins=50, density=True, alpha=0.4, label="rejected", color="tab:orange")

    # ② KDE 曲线
    plt.plot(xs, kde_ch, lw=2, label="KDE chosen",   color="tab:blue")
    plt.plot(xs, kde_re, lw=2, label="KDE rejected", color="tab:orange")

    # ③ 阴影显示 kde_ch - kde_re > 0
    diff = kde_ch - kde_re
    plt.fill_between(xs, kde_ch, kde_re,
                     where=diff>0, interpolate=True,
                     color="tab:blue", alpha=0.2, label="chosen > rejected")

    plt.xlabel("Reward (logit)")
    plt.ylabel("Density")
    plt.title("Reward distributions: chosen vs rejected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# ---------- 主流程 ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tokenizer_path", help="若空则取 model_path 的父目录")
    ap.add_argument("--sarm_base_model", choices=["llama","gemma2",None], default=None)
    ap.add_argument("--output_jsonl", default="scored.jsonl")
    ap.add_argument("--plot_path",   default="plot.png")
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ---------- tokenizer ----------
    tok_path = args.tokenizer_path or os.path.dirname(os.path.normpath(args.model_path))
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token":"[PAD]"})
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = args.max_length

    # ---------- model ----------
    # 按需替换为 LlamaSARM / Gemma2SARM …（与之前逻辑相同，此处略）
    ModelCls = None
    if args.sarm_base_model == "llama" and LlamaSARM is not None:
        ModelCls = LlamaSARM4Steering
    elif args.sarm_base_model == "gemma2" and Gemma2SARM is not None:
        ModelCls = Gemma2SARM
    if ModelCls is None:
        ModelCls = AutoModelForSequenceClassification
    sarm_args = {}
    if args.sarm_base_model!=None:
        sarm_args = parse_sarm_param(args.model_path)

    model = ModelCls.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        **sarm_args
    ).to(args.device).eval()

    # ---------- 读数据 & 评分 ----------
    chosen_scores, rejected_scores = [], []
    out_path = Path(args.output_jsonl).expanduser().resolve()
    with out_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(stream_jsonl(Path(args.data_path)), desc="Scoring"):
            q, c_ans, r_ans = row["question"], row["chosen"], row["rejected"]

            def score_one(answer, steering):
                msgs=[{"role":"user","content":q},{"role":"assistant","content":answer}]
                text=tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                tok=tokenizer(text, truncation=True, max_length=args.max_length, return_tensors="pt")
                tok={k:v.to(args.device) for k,v in tok.items()}
                return model(steering=steering, **tok)["logits"].squeeze().item()
            steering1 = {
                16720:['*', 1]
            }
            steering2 = {
                16720:['*', 2]
            }
            s_c = score_one(c_ans, steering1)
            s_r = score_one(c_ans, steering2)
            chosen_scores.append(s_c); rejected_scores.append(s_r)
            row.update({"chosen_reward":s_c, "rejected_reward":s_r})
            fout.write(json.dumps(row,ensure_ascii=False)+"\n")
    print("✓ JSONL with scores saved at", out_path)

    # ---------- 单独绘制两张图 ----------
    plot_combined(chosen_scores, rejected_scores, args.plot_path)
    print(f"✓ Plots saved → {args.plot_path}")

if __name__=="__main__":
    main()
