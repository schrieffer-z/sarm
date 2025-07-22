#!/usr/bin/env python3
# merge_contexts.py
"""
合并多份 latent_context_map-JSON

功能
----
* 多文件合并：latent ⟶ token ⟶ context 不重复
* 若同一 context 出现多次，仅保留 activation 更高的记录
* 过滤：activation > threshold 且每 (latent, token) 至少 `lines` 条
* 每个 token 至多保留 `max_per_token` 条激活最高的 context
* 自动重算 total_latents，其他元数据直接沿用

用法
----
python merge_contexts.py \
  --lines 10 --threshold 10 --max_per_token 256 \
  --output merged.json \
  split1.json split2.json split3.json ...
"""

import argparse, json, pathlib
from collections import defaultdict
from tqdm import tqdm


# ----------------- 核心合并逻辑 -----------------
def merge_json_files(file_paths, lines, threshold, max_per_token):
    merged_map = defaultdict(lambda: defaultdict(list))
    context_check = defaultdict(lambda: defaultdict(dict)) 
    
    for path in file_paths:
        with open(path, 'r') as f:
            data = json.load(f)
        for latent, tokens in tqdm(data["latent_context_map"].items(), total=len(data["latent_context_map"].items()), desc=f"Processing {path[-14:]}"):
            for token, entries in tokens.items():
                ck_map = context_check[latent][token]
                for entry in entries:
                    if entry['context'] in ck_map.keys():
                        exist_act = ck_map[entry['context']]
                        if entry['activation'] <= exist_act:
                            continue
                        merged_map[latent][token].remove({
                            'context': entry['context'], 
                            "activation": exist_act
                        })
                        merged_map[latent][token].append(entry)
                        ck_map[entry['context']] = entry['activation']
                    else:
                        ck_map[entry['context']] = entry['activation']
                        merged_map[latent][token].append(entry)
        
    x = defaultdict(lambda: defaultdict(list))
    for latent, tokens in tqdm(merged_map.items()):
        for token, entries in tokens.items():
            entries = [entry for entry in entries if entry['activation']>threshold]
            if len(entries) < lines:
                continue
            entries.sort(key=lambda x: x["activation"], reverse=True)
            x[latent][token] = entries[:max_per_token]
            
    
    result = {
        "total_latents": len(x),
        "threshold": threshold,
        "max_length": 1024,
        "max_per_token": max_per_token,
        "lines": lines,
        "latent_context_map": x,
    }

    return result


# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines",            type=int, default=10,   help="每 (latent,token) 至少保留多少条")
    ap.add_argument("--threshold",        type=float, default=5, help="activation > threshold 才保留")
    ap.add_argument("--max_per_token",    type=int, default=256,  help="每 token 最多保留多少条")
    ap.add_argument("--output",           required=True, help="输出 JSON 路径")
    ap.add_argument("files", nargs="+", help="待合并的 JSON 文件")
    args = ap.parse_args()

    result = merge_json_files(
        args.files,
        lines=args.lines,
        threshold=args.threshold,
        max_per_token=args.max_per_token
    )

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✓ merged {len(args.files)} files → {out_path}")

if __name__ == "__main__":
    main()
