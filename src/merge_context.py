import argparse, json, pathlib
from collections import defaultdict
from tqdm import tqdm


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines",            type=int, default=10,   help="least number of contexts (latent,token) ")
    ap.add_argument("--threshold",        type=float, default=5, help="activation's threshold ")
    ap.add_argument("--max_per_token",    type=int, default=256,  help="max number of contexts to keep")
    ap.add_argument("--output",           required=True, help="output path of JSON")
    ap.add_argument("files", nargs="+", help="input files in JSON")
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
