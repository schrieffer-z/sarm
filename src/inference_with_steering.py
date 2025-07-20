'''

python score_steering_plot.py \
  --data_path qa.jsonl \
  --model_path /ckpt/Latent65536_Layer16_K192_SARM \
  --sarm_base_model llama \
  --steering_path steering.json \
  --plot_path reward_steer.png


'''
#!/usr/bin/env python
import argparse, json, os, re
from pathlib import Path
from typing import Dict, List, Iterator

import torch, numpy as np, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from sarm_llama  import LlamaSARM
    from sarm_gemma2 import Gemma2SARM
except ImportError:
    LlamaSARM = Gemma2SARM = None


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
    plt.fill_between(xs, kde_a, kde_b, where=kde_a>kde_b, alpha=.2,
                     color="tab:orange", label="after > before")
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
        Model = LlamaSARM
    elif args.sarm_base_model == "gemma2":
        Model = Gemma2SARM
    else:
        Model = AutoModelForSequenceClassification
    extra = parse_sarm_param(args.model_path) if args.sarm_base_model else {}
    model = Model.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, **extra
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
