#!/usr/bin/env python3
"""embed_docs.py — embeddings densos de un corpus jsonl (e5-small, mean-pool).

Corre en cualquier nodo GPU del cluster: transformers 4.46 via
PYTHONPATH=/beegfs/a474r867/pylibs (compatible con torch 2.4 base).

Salida en --out-dir:
  emb.fp16.npy   memmap [N, 384] L2-normalizado
  idx.jsonl      una linea por doc: meta original + row
"""
import argparse, json, os, sys, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True,
                    help="jsonl(s) de entrada")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--meta-fields", default="pid,domain,src,lang",
                    help="campos del jsonl a copiar al idx")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(dev).eval()
    dim = model.config.hidden_size
    keep = [f.strip() for f in args.meta_fields.split(",") if f.strip()]

    # primera pasada: contar docs (para memmap exacto)
    n_docs = 0
    for path in args.jsonl:
        with open(path) as f:
            for _ in f:
                n_docs += 1
                if args.limit and n_docs >= args.limit:
                    break
        if args.limit and n_docs >= args.limit:
            break
    print(f"[embed] {n_docs} docs, dim={dim}, dev={dev}", flush=True)

    emb = np.memmap(os.path.join(args.out_dir, "emb.fp16.npy"),
                    dtype=np.float16, mode="w+", shape=(n_docs, dim))
    idx_path = os.path.join(args.out_dir, "idx.jsonl")
    idx_f = open(idx_path, "w")

    row = 0
    t0 = time.time()
    buf_txt, buf_meta = [], []

    def flush():
        nonlocal row
        if not buf_txt:
            return
        enc = tok(["passage: " + t for t in buf_txt], return_tensors="pt",
                  padding=True, truncation=True, max_length=args.max_len)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        emb[row:row + len(buf_txt)] = pooled.cpu().numpy().astype(np.float16)
        for mrec in buf_meta:
            mrec["row"] = row
            idx_f.write(json.dumps(mrec, ensure_ascii=False) + "\n")
            row += 1
        if row % (args.batch * 20) < args.batch:
            dt = time.time() - t0
            print(f"[embed] {row}/{n_docs} ({row/dt:.0f} docs/s)", flush=True)
        buf_txt.clear(); buf_meta.clear()

    for path in args.jsonl:
        src_name = os.path.basename(path)
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                txt = d.get(args.text_field)
                if not txt:
                    continue
                buf_txt.append(txt)
                meta = {k: d.get(k) for k in keep if k in d}
                meta["src_file"] = src_name
                buf_meta.append(meta)
                if len(buf_txt) >= args.batch:
                    flush()
                if args.limit and row >= args.limit:
                    break
        if args.limit and row >= args.limit:
            break
    flush()
    emb.flush(); idx_f.close()
    print(f"[embed] DONE {row} docs -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
