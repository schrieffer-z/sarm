#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
interpret.py
------------
对已有 *context-json* 中的 latent 逐一调用 GPT / Azure-OpenAI，
让模型给出 “Explanation + Score(-2…2)” ，结果保存为 JSON。

运行示例
--------
python interpret.py \
  --context_path ../contexts/contexts_latent.json \
  --selected_latent_path ../latents/top600.json  \
  --api_base https://api.nuwaapi.com https://api.nuwaapi.com/v1 https://api.nuwaapi.com/v1/chat/completions\
  --api_key  sk-*** \
  --engine   gpt-4o \
  --context_per_latent 40 \
  --output_path  interp_results.json
"""
import argparse, json, re, time, random
from pathlib import Path
from typing  import Dict, List, Any

from tqdm import tqdm
from openai import OpenAI, AzureOpenAI

# ------------------- util ----------------------------------------------------
def extract_explanation_and_score(text: str):
    """从回复里抓 Explanation: …  和 Score: N （-2~2）"""
    exp = re.search(r'Explanation:\s*(.*?)(?:\n|$)', text, re.S).group(1).strip() \
          if re.search(r'Explanation:', text) else None
    m   = re.search(r'Score:\s*([-\-]?\d+)', text)
    score = int(m.group(1)) if m else None
    return exp, score

def load_json(p: str|Path) -> Any:
    with open(p, encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, p: str|Path):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ------------------- prompt --------------------------------------------------
PROMPT_HEAD = """A reward model outputs a single scalar estimating how well a model-generated\
 response aligns with human preferences for a given question.

Your task: judge whether the presence of **that context** makes the response \
more or less aligned with widely accepted social values and human preferences.

Scoring scale:
  2  : Strongly increases alignment
  1  : Moderately increases alignment
  0  : Neutral / no clear effect
 -1  : Moderately decreases alignment
 -2  : Strongly decreases alignment

Respond **exactly** in the following format:
Explanation: <one-sentence summary of the shared concept>
Score: <-2 | -1 | 0 | 1 | 2>

Consider the context snippets below in which the feature activates:
"""

def build_prompt(contexts: List[str]) -> str:
    body = "".join(f"Context: {c}\n\n" for c in contexts)
    return PROMPT_HEAD + body + "Now provide your two-line answer."

# ------------------- interpreter --------------------------------------------
def chat_completion(clients: list, prompt: str, engine: str,
                    max_retry: int = 2, timeout: int = 60):
    for client in clients:
        for _ in range(max_retry):
            try:
                ans = client.chat.completions.create(
                    model   = engine,
                    messages=[{"role":"system","content":"You are an assistant that analyses latent semantics."},
                              {"role":"user",  "content":prompt}],
                    temperature = 0,
                )
                return ans.choices[0].message.content.strip()
            except Exception:
                time.sleep(1+random.random())
    raise RuntimeError("All endpoints failed")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context_path",          required=True,
                    help="JSON with latent_context_map produced by apply step")
    ap.add_argument("--selected_latent_path",  required=True,
                    help="JSON list/dict of latents to interpret")
    ap.add_argument("--api_base",   nargs='+', required=True,
                    help="One or more OpenAI-compatible base URLs")
    ap.add_argument("--api_key",    required=True)
    ap.add_argument("--engine",     default="gpt-4o")
    ap.add_argument("--context_per_latent", type=int, default=40)
    ap.add_argument("--output_path", default="interpret_results.json")
    args = ap.parse_args()

    # -------- load data --------
    ctx_json   = load_json(args.context_path)
    latent_map = ctx_json["latent_context_map"]

    sel_lat    = load_json(args.selected_latent_path)
    if isinstance(sel_lat, dict): sel_lat = list(sel_lat.keys())
    sel_lat = [str(i) for i in sel_lat]

    # -------- clients ----------
    clients = [OpenAI(api_key=args.api_key, base_url=b, timeout=60, max_retries=2)
               for b in args.api_base]

    # -------- resume -----------
    out_path = Path(args.output_path)
    if out_path.exists():
        results = load_json(out_path)
    else:
        results = {"engine": args.engine, "features_scored": 0, "results": {}}

    # -------- iterate ----------
    for lat in tqdm(sel_lat, desc="Interpreting"):
        if lat not in latent_map:           continue
        if lat in results["results"] and results["results"][lat]["score"] is not None:
            continue

        ctxs = []
        for token_ctxs in latent_map[lat].values():
            ctxs += token_ctxs
        ctxs = sorted(ctxs, key=lambda x: x["activation"], reverse=True)[:args.context_per_latent]
        prompt = build_prompt([c["context"] for c in ctxs])

        try:
            resp = chat_completion(clients, prompt, args.engine)
            expl, sc = extract_explanation_and_score(resp)
        except Exception as e:
            expl, sc, resp = None, None, str(e)

        results["results"][lat] = {
            "score": sc,
            "explanation": expl,
            "contexts": [c["context"] for c in ctxs],
            "response": resp,
        }
        results["features_scored"] = sum(1 for v in results["results"].values()
                                         if v["score"] is not None)

        save_json(results, out_path)

    print(f"✓ finished. Saved → {out_path}")

if __name__ == "__main__":
    main()
