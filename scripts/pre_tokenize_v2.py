#!/usr/bin/env python3
"""pre_tokenize_v2.py — tokeniza corpus con EOS y guarda .npz para packing v2.

Lee un corpus JSONL, tokeniza con el tokenizer LLaDA, añade EOS al final de
cada documento y guarda un .npz con:
  - ids:      array int32 plano con todos los tokens (incluyendo EOS).
  - lengths:  array int32 con la longitud de cada documento (incluyendo EOS).
  - eos_id:   token id del EOS.

Esto permite a train_mdlm_moe_v2.py hacer packing con fronteras de documento.

Uso:
  python3 pre_tokenize_v2.py --input data/train_skeleton.jsonl \
      --tokenizer <dir> --out data/train_ids_skeleton_v2.npz [--workers 16]
"""
import argparse, json, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer


def _clamp(ids, vocab, eos_id):
    """Asegura que todos los ids esten dentro del vocab del tokenizer."""
    if eos_id is not None and eos_id >= vocab:
        eos_id = 0
    out = []
    for x in ids:
        if x >= vocab:
            out.append(0)
        else:
            out.append(x)
    return out, eos_id


def encode_batch(args):
    lines, tok_path, max_len, field = args
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True,
                                        local_files_only=True)
    vocab = tok.vocab_size
    eos_id = tok.eos_token_id
    if eos_id is None:
        eos_id = tok.pad_token_id
    ids_all = []
    lengths = []
    clipped = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line).get(field, "")
        except Exception:
            continue
        ids = tok.encode(t, add_special_tokens=False)[:max_len - 1]
        if len(ids) >= 4:
            ids, eos_id = _clamp(ids, vocab, eos_id)
            ids.append(eos_id)
            ids_all.extend(ids)
            lengths.append(len(ids))
        else:
            clipped += 1
    return ids_all, lengths, clipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=768)
    ap.add_argument("--field", default="text", help="campo JSON con el texto")
    args = ap.parse_args()

    t0 = time.time()
    with open(args.input) as f:
        lines = f.readlines()
    n_lines = len(lines)
    print(f"[{time.strftime('%H:%M:%S')}] leidos {n_lines} lineas de {args.input}", flush=True)

    chunk = max(1, len(lines) // args.workers)
    chunks = [lines[i:i + chunk] for i in range(0, len(lines), chunk)]
    all_ids = []
    all_lengths = []
    clipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(encode_batch, (ch, args.tokenizer, args.seq_len, args.field))
                for ch in chunks]
        for i, fu in enumerate(futs):
            ids, lens, cl = fu.result()
            all_ids.extend(ids)
            all_lengths.extend(lens)
            clipped += cl
            print(f"[{time.strftime('%H:%M:%S')}] chunk {i + 1}/{len(futs)} listo "
                  f"({len(all_ids)/1e6:.1f}M tok acum)", flush=True)

    arr = np.array(all_ids, dtype=np.int32)
    lengths = np.array(all_lengths, dtype=np.int32)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True,
                                        local_files_only=True)
    vocab = tok.vocab_size
    eos_id = tok.eos_token_id
    if eos_id is None:
        eos_id = tok.pad_token_id
    if eos_id is not None and eos_id >= vocab:
        eos_id = 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, ids=arr, lengths=lengths, vocab_size=int(vocab),
             eos_id=int(eos_id))

    meta = {
        "n_tokens": int(arr.size),
        "n_docs_usable": len(lengths),
        "n_docs_clipped": int(clipped),
        "seq_len": args.seq_len,
        "vocab_size": tok.vocab_size,
        "eos_id": int(eos_id),
        "input": args.input,
        "field": args.field,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = args.out + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    el = time.time() - t0
    print(f"DONE {n_lines} lineas -> {len(lengths)} docs, {arr.size/1e9:.3f}B tokens "
          f"(incl. EOS) en {el/60:.1f}min", flush=True)
    print(f"guardado: {args.out} + {meta_path}", flush=True)


if __name__ == "__main__":
    main()
