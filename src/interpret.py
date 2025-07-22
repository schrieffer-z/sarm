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
import argparse, torch, json, os, re, time, random
from pathlib import Path
from typing  import Dict, List, Any
from collections import defaultdict
from safetensors.torch import load_file
from typing import Set, Tuple, Union
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

def extract_top_latents(model_path: str,
                        weight_path: str,
                        context: dict,
                        min_context_per_latent :int,
                        top_k_list: list[int]):
    assert model_path is not None or weight_path is not None
    latent_context_map = context.get('latent_context_map', {})
    all_latents = set(latent_context_map.keys())

    # 读取模型权重
    if model_path is not None and not os.path.exist(weight_path):
        weight_map_path = os.path.join(model_path, 'model.safetensors.index.json')
        if os.path.exists(weight_map_path):
            with open(weight_map_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f)['weight_map']
            score_weights = load_file(os.path.join(model_path, weight_map['score.weight']))['score.weight'].view(-1)
        else:
            score_weights = load_file(os.path.join(model_path, "model.safetensors"))['score.weight'].view(-1)
    else:
        score_weights = torch.load(weight_path)

    # 构建 latent-score 映射并排序
    latent_list = []
    for latent in all_latents:
        num_contexts = sum([len(latent_context_map[latent][token]) for token in latent_context_map[latent]])
        if num_contexts < min_context_per_latent:
            continue
        latent_list.append({latent: score_weights[int(latent)].item()})
    latent_list.sort(key=lambda x: list(x.values())[0], reverse=True)

    ret = {}
    for num in top_k_list:
        selected_latent_list = latent_list[:num] + latent_list[-num:]
        selected_latent_dict = {}
        for i in selected_latent_list:
            selected_latent_dict.update(i)
        
        ret.update({num:selected_latent_dict})
    return ret


def build_prompt(contexts: List[str]) -> str:
    prompt = (
        "A reward model outputs a **single scalar** estimating how well a model-generated response "
        "aligns with human preferences for a given question.\n\n"

        "Your task: judge to judge whether the presence of that context makes the response "
        "**more or less** aligned with widely accepted social values and human preferences."

        "Scoring scale:\n"
        "  2  : Strongly increases alignment with social values and likflihood of being prefered by human\n"
        "  1  : Moderately increases alignment with social values and likflihood of being prefered by human\n"
        "  0  : Neutral / no clear effect\n"
        " -1  : Moderately decreases alignment with social values and likflihood of being prefered by human\n"
        " -2  : Strongly decreases alignment with social values and likflihood of being prefered by human\n\n"

        "Important notes:\n"
        "• Judge the **overall meaning** of the context, not surface details.\n\n"
        "• Despite the semantic of the shared concept of all contexts itself, judge whether its presence in a response aligned with social values and human preference"

        "Respond **exactly** in the following format (replace content in [] following the requirements in []):\n"
        "Explanation: [A sentence to summary the shared concept of contexts activating the latent]\n\n"
        "Score: [Based on the stated scale, choose from -2, -1, 0, 1, 2]\n\n"

        "Consider the context snippets below in which the feature activates:\n\n"
    )
    for c in contexts:
        prompt += f"Context: {c}\n\n"
    prompt += (
        "Now provide your two-line answer.\n"
    )
    return prompt

# ------------------- interpreter --------------------------------------------
def chat_completion(clients: list, prompt: str, engine: str, max_retry: int = 2, timeout: int = 60):
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
    ap.add_argument("--context_path", required=True, help="JSON with latent_context_map produced by apply step")
    ap.add_argument("--model_path", required=False, help="Use weights to filter latents from the whole model")
    ap.add_argument("--weight_path", required=False, help="Use weights to filter latents from a .pt file")
    ap.add_argument("--context_per_latent", type=int, default=40)
    ap.add_argument("--output_path", default="interpret_results.json")
    ap.add_argument('--top_k_list', type=int, nargs='+', 
        default=[50, 100, 150, 200, 250],
        help='List of top-K values to extract (default: %(default)s)')
    ap.add_argument("--api_base", nargs='+', required=True, help="One or more OpenAI-compatible base URLs")
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--engine", default="gpt-4o")

    args = ap.parse_args()

    # -------- load data --------
    ctx_json = load_json(args.context_path)
    latent_context_map = ctx_json["latent_context_map"]

    top_latents_weights_map = extract_top_latents(
        model_path=args.model_path,
        weight_path=args.weight_path,
        context=ctx_json,
        min_context_per_latent=args.context_per_latent*2,
        top_k_list=args.top_k_list
    )
    sel_lat = top_latents_weights_map[max(args.top_k_list)]
    if isinstance(sel_lat, dict): 
        sel_lat = list(sel_lat.keys())

    # -------- clients ----------
    clients = [OpenAI(api_key=args.api_key, base_url=b, timeout=60, max_retries=2) for b in args.api_base]

    # -------- resume -----------
    out_path = Path(args.output_path)
    if out_path.exists():
        results = load_json(out_path)
    else:
        results = {"engine": args.engine, "features_scored": 0, "results": {}}

    # -------- iterate ----------
    for lat in tqdm(sel_lat, desc="Interpreting"):
        if lat not in latent_context_map:           
            continue
        if lat in results["results"] and results["results"][lat]["score"] is not None:
            continue

        ctxs = []
        for token_ctxs in latent_context_map[lat].values():
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
        results["features_scored"] = sum(1 for v in results["results"].values() if v["score"] is not None)

        save_json(results, out_path)

    print(f"✓ finished. Saved → {out_path}")

    latent_score_map = results["results"]
    for topk in args.top_k_list:
        selected_latents = top_latents_weights_map[topk]
        debug_latent = defaultdict(lambda: defaultdict())
        correct_latent = defaultdict(lambda: defaultdict())
        
        cnt_correct, cnt_valid = 0, 0
        cnt_none_zero, cnt_pos, cnt_neg = 0, 0, 0
        cnt_pos_correct, cnt_neg_correct = 0, 0
        for latent in selected_latents.keys():
            if latent not in selected_latents.keys():
                continue
            if latent_score_map[latent]['score'] is None:
                continue

            cnt_valid += 1
            if latent_score_map[latent]['score'] != 0:
                cnt_none_zero += 1
                cnt_pos += 1 if selected_latents[latent]>0 else 0
                cnt_neg += 1 if selected_latents[latent]<0 else 0
            else:
                continue
            if latent_score_map[latent]['score'] * selected_latents[latent] > 0:
                cnt_correct += 1
                if latent_score_map[latent]['score']>0:
                    cnt_pos_correct +=1
                else:
                    cnt_neg_correct +=1
                correct_latent[latent] = latent_score_map[latent]
                correct_latent[latent]['weight'] = selected_latents[latent]        
            else:
                debug_latent[latent] = latent_score_map[latent]
                debug_latent[latent]['weight'] = selected_latents[latent]        
        
        print(f"Non-Zero Ration: {cnt_none_zero / (2*topk) :.4f} ({cnt_none_zero}/{topk*2})")
        print(f"Positive Score Ration: {cnt_pos}(Pos):{cnt_neg}(Neg)")
        print(f"ACC of Latents With Positive Weight:  ({cnt_pos_correct}/{cnt_pos})")
        print(f"ACC of Latents With Negative Weight:  ({cnt_neg_correct}/{cnt_neg})")

        print(f"ACC: { cnt_correct / cnt_none_zero :.4f} ({cnt_correct}/{cnt_none_zero})")
        print("="*60)
        
        if topk == max(args.top_k_list):
            os.makedirs('interpret/debug',exist_ok=True)
            with open(f'interpret/debug/debug-{topk*2}.json', 'w', encoding='utf-8') as f:
                json.dump(debug_latent, f, ensure_ascii=False, indent=4)
            with open(f'interpret/debug/correct-{topk*2}.json', 'w', encoding='utf-8') as f:
                json.dump(correct_latent, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
