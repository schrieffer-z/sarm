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
    """extract Explanation & Score"""
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


def build_prompt(contexts) -> str:
    prompt = (
        'We are analyzing the activation levels of features in a language model, where each feature activates certain sequence at the end of it.\n'
        'Each Sequence\'s activation value indicates its relevance to the feature, with higher values showing stronger association.\n'
        'Your task is to give this feature a monosemanticity score based on the following scoring rubric:\n'
        'Activation Consistency\n'
        '5: Clear pattern with no deviating examples\n'
        '4: Clear pattern with one or two deviating examples\n'
        '3: Clear overall pattern but quite a few examples not fitting that pattern\n'
        '2: Broad consistent theme but lacking structure\n'
        '1: No discernible pattern\n'
        'Consider the following activations for a feature in the language model.\n\n'
    )
    for c in contexts:
        prompt += f"Activation: {c[1]} | Context: {c[0]}\n\n"
    prompt += (
        'Provide your response in the following fixed format:\n'
        'Explanation: [Your brief explanation]\n'
        'Score: [5/4/3/2/1]\n'
        'Now provide your two-line answer.\n'
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
    ap.add_argument("--context_path", type=str, required=True, help="path of JSON with latent_context_map produced by apply.py")
    ap.add_argument('--latents_path', type=str, required=True, help='path of selected latent to be interpreted using collected latents')

    ap.add_argument("--context_per_latent", type=int, default=40)

    ap.add_argument("--api_base", nargs='+', required=True, help="One or more OpenAI-compatible base URLs")
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--engine", default="gpt-4o")

    ap.add_argument("--output_path", default="interpret_results.json")
    args = ap.parse_args()

    # -- load contexts & latens --
    ctx_json = load_json(args.context_path)
    latent_context_map = ctx_json["latent_context_map"]

    selected_latents = []
    with open(args.latents_path, "r") as f:
        selected_latents = [line.strip() for line in f]
    assert len(selected_latents)!=0 and isinstance(selected_latents[0], str)

    # -------- clients ----------
    clients = [OpenAI(api_key=args.api_key, base_url=b, timeout=60, max_retries=2) for b in args.api_base]

    # -------- resume -----------
    out_path = Path(args.output_path)
    if out_path.exists():
        results = load_json(out_path)
    else:
        results = {"engine": args.engine, "features_scored": 0, "results": {}}

    # -------- iterate ----------
    for lat in tqdm(selected_latents, desc="Interpreting"):
        if lat not in latent_context_map:           
            continue
        if lat in results["results"] and results["results"][lat]["score"] is not None:
            continue

        ctxs = []
        for token_ctxs in latent_context_map[lat].values():
            ctxs += token_ctxs
        ctxs = sorted(ctxs, key=lambda x: x["activation"], reverse=True)[:args.context_per_latent]
        prompt = build_prompt([(c["context"], c["activation"]) for c in ctxs])

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


if __name__ == "__main__":
    main()
