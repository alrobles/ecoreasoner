#!/usr/bin/env python3
"""pre_tokenize.py — tokeniza el corpus una sola vez y guarda a disco.

Lee 2_pretrain_3 (train_corpus_v3.jsonl), tokeniza con el tokenizer LLaDA (local, offline),
y guarda los IDs concatenados en un solo array plano .npy (int32) + metadata .json.
Esto elimina el re-tokenizado dentro de cada slurm (build_batches), que hoy
cuenta todo el corpus por rank (muy lento con 1M docs).

DATASET (ver docs/dataset-registry.md): lee 2_pretrain_3 -> escribe 2_ids_3.

Uso:
  python3 pre_tokenize.py --input data/train_corpus_v3.jsonl \
      --tokenizer <dir> --out data/train_ids_v3.npy [--workers 16] [--seq_len 768]

Output: 
  - <out>            : array int32 plano con todos los tokens (concatenados).
  - <out>.meta.json  : {n_tokens, seq_len, n_docs_usable, vocab_size}
"""
import argparse, json, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def encode_batch(args):
    lines, tok_path, tokenizer_type, max_len = args
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True,
                                        local_files_only=True)
    ids_all = []
    clipped = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            import json as _j
            t = _j.loads(line).get("text", "")
        except Exception:
            continue
        ids = tok.encode(t)[:max_len]
        if len(ids) >= 4:  # descartar demasiado cortos
            ids_all.append(ids)
        else:
            clipped += 1
    return ids_all, clipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=768)
    ap.add_argument("--tokenizer_type", default=None)
    args = ap.parse_args()

    t0 = time.time()
    # leer lineas
    n_lines = 0
    with open(args.input) as f:
        lines = f.readlines()
    n_lines = len(lines)
    print(f"[{time.strftime('%H:%M:%S')}] leidos {n_lines} lineas de {args.input}", flush=True)

    chunk = max(1, len(lines) // args.workers)
    chunks = [lines[i:i+chunk] for i in range(0, len(lines), chunk)]
    all_ids = []
    clipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(encode_batch, (ch, args.tokenizer, args.tokenizer_type, args.seq_len))
                for ch in chunks]
        for i, fu in enumerate(futs):
            idd, clip = fu.result()
            all_ids.extend(idd)
            clipped += clip
            print(f"[{time.strftime('%H:%M:%S')}] chunk {i+1}/{len(futs)} listo "
                  f"({sum(len(x) for x in all_ids)/1e6:.1f}M tok acum)", flush=True)

    # concatenar a array plano int32
    n_tok = sum(len(x) for x in all_ids)
    arr = np.empty(n_tok, dtype=np.int32)
    pos = 0
    for x in all_ids:
        arr[pos:pos+len(x)] = x
        pos += len(x)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, arr)
    meta = {
        "n_tokens": int(n_tok),
        "n_docs_usable": len(all_ids),
        "n_docs_clipped": int(clipped),
        "seq_len": args.seq_len,
        "vocab_size": None,  # se llena desde tokenizer
        "input": args.input,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # vocab size
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True,
                                            local_files_only=True)
        meta["vocab_size"] = tok.vocab_size
    except Exception as e:
        print("warn: no pude obtener vocab_size", e)
    meta_path = args.out + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    el = time.time() - t0
    print(f"DONE {n_tok/1e9:.3f}B tokens usable={len(all_ids)} docs in {el/60:.1f}min", flush=True)
    print(f"guardado: {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB int32) + meta: {meta_path}", flush=True)

if __name__ == "__main__":
    main()