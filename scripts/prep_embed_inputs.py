#!/usr/bin/env python3
"""prep_embed_inputs.py — prepara jsonl de TEXTO para embed_docs.py.

Entradas:
  --holdout DIR   pares tokenizados (ctx/ok/bad) -> decodifica ctx+ok con
                  el tokenizer LLaDA -> docs de texto (para leakage check)
  --unam-md GLOB  markdown UNAM -> chunks ~3000 chars con tags
                  (lang=es, src=unam, domain=area de meta.jsonl)
  --unam-meta F   meta.jsonl del curador para area/facultad/titulo
  --tokenizer P   ruta tokenizer LLaDA (para decodificar holdout)
  --out DIR
"""
import argparse, glob, json, os, re


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower())[:60].strip("-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--unam-md", default=None, help="glob de .md curados")
    ap.add_argument("--unam-meta", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-chars", type=int, default=3000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.holdout and args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                            trust_remote_code=True)
        n = 0
        outp = os.path.join(args.out, "holdout_docs.jsonl")
        with open(outp, "w") as fo:
            for pf in sorted(glob.glob(os.path.join(args.holdout,
                                                    "pairs_L*.jsonl"))):
                lvl = os.path.basename(pf).replace("pairs_", "").replace(".jsonl", "")
                for line in open(pf):
                    d = json.loads(line)
                    txt = tok.decode(d["ctx"] + d["ok"],
                                     skip_special_tokens=True)
                    fo.write(json.dumps(
                        {"text": txt, "src": "holdout", "level": lvl},
                        ensure_ascii=False) + "\n")
                    n += 1
        print(f"[prep] holdout -> {outp} ({n} docs)", flush=True)

    if args.unam_md:
        meta = {}
        if args.unam_meta and os.path.exists(args.unam_meta):
            for line in open(args.unam_meta):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in (d.get("slug"), slug(d.get("title", "")),
                            d.get("uuid", "")[:8]):
                    if key:
                        meta[key] = d
        outp = os.path.join(args.out, "unam_docs.jsonl")
        n = 0
        with open(outp, "w") as fo:
            for md in sorted(glob.glob(args.unam_md)):
                base = os.path.basename(md)
                key = base.split("_", 1)[0] if "_" in base else None
                m = meta.get(key) or meta.get(slug(base.rsplit(".", 1)[0])) or {}
                try:
                    txt = open(md, encoding="utf-8").read()
                except OSError:
                    continue
                for ci in range(0, len(txt), args.chunk_chars):
                    chunk = txt[ci:ci + args.chunk_chars].strip()
                    if len(chunk) < 600:
                        continue
                    fo.write(json.dumps({
                        "text": chunk, "src": "unam", "lang": "es",
                        "domain": m.get("area", "unam"),
                        "facultad": m.get("facultad"),
                        "pid": f"unam:{base}:{ci//args.chunk_chars}"},
                        ensure_ascii=False) + "\n")
                    n += 1
        print(f"[prep] unam -> {outp} ({n} chunks)", flush=True)


if __name__ == "__main__":
    main()
