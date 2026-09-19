#!/usr/bin/env python3
"""sft_pretok.py — pretokeniza el corpus chat-SFT para sft_mdlm.py.

Igual que pre_tokenize_v2.py pero por pares {prompt, response}: emite
  - ids:         prompt_ids + response_ids + EOS (plano int32)
  - lengths:     len total por par (incl. EOS)
  - resp_starts: offset (en tokens) donde empieza la response dentro del doc
                 = len(prompt_ids). El trainer enmascara solo [resp_start, len)
                 y aplica CE ahí; el prompt queda siempre visible.
Trunca el TOTAL a max_len-1 (si la response queda vacía tras truncar, se
descarta el par).

Uso:
  python3 sft_pretok.py --input chat_sft_v1.jsonl --tokenizer <dir> \
      --out sft_ids_v1.npz --workers 8 [--seq_len 768]
"""
import argparse, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer


def encode_batch(tup):
    recs, tok_path, max_len = tup
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True,
                                        local_files_only=True)
    eos = tok.eos_token_id
    if eos is None:
        eos = tok.pad_token_id
    if eos is not None and eos >= tok.vocab_size:
        eos = 0  # eos fuera de vocab del modelo (id 126081 > 126080) -> 0
    ids_all, lens, starts = [], [], []
    dropped = 0
    for r in recs:
        try:
            p = tok.encode(r["prompt"], add_special_tokens=False)
            s = tok.encode(r["response"], add_special_tokens=False)
        except Exception:
            dropped += 1
            continue
        ids = (p + s)[:max_len - 1]
        if len(ids) <= len(p) or len(ids) < 8:
            dropped += 1  # response vacía o doc degenerado tras truncar
            continue
        ids.append(eos)
        ids_all.extend(ids)
        lens.append(len(ids))
        starts.append(len(p))
    return ids_all, lens, starts, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_len", type=int, default=768)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=4000)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.input) if l.strip()]
    print(f"[sft_pretok] {len(recs)} pares")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True,
                                        local_files_only=True)
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    if eos is not None and eos >= tok.vocab_size:
        eos = 0
    chunks = [recs[i:i+args.chunk] for i in range(0, len(recs), args.chunk)]
    ids_all, lens, starts = [], [], []
    dropped = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for ids, l, s, d in ex.map(encode_batch,
                                   [(c, args.tokenizer, args.seq_len) for c in chunks]):
            ids_all.extend(ids); lens.extend(l); starts.extend(s); dropped += d
    np.savez_compressed(args.out,
                        ids=np.asarray(ids_all, dtype=np.int32),
                        lengths=np.asarray(lens, dtype=np.int32),
                        resp_starts=np.asarray(starts, dtype=np.int32),
                        eos_id=np.int64(eos))
    meta = {"n_pairs": len(lens), "n_tokens": len(ids_all), "dropped": dropped,
            "seq_len": args.seq_len, "eos_id": int(eos), "input": args.input}
    import json as j
    open(args.out + ".meta.json", "w").write(j.dumps(meta, indent=1))
    print(f"[sft_pretok] DONE {meta['n_pairs']} pares, {meta['n_tokens']/1e6:.1f}M tok, "
          f"dropped={dropped} -> {args.out}")


if __name__ == "__main__":
    main()
