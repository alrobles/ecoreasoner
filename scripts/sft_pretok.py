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
    ids_all, lens, starts, ends, nspans = [], [], [], [], []
    dropped = 0
    for r in recs:
        try:
            if "turns" in r:   # multi-turn: spans por cada respuesta
                segs, resp_idx = [], []
                for t in r["turns"]:
                    segs.append(tok.encode(f"[USER] {t['q']}\n[ASSISTANT]",
                                           add_special_tokens=False))
                    segs.append(tok.encode(f" {t['a']}\n",
                                           add_special_tokens=False))
                    resp_idx.append(len(segs) - 1)
                ids, sp = [], []
                for si, s in enumerate(segs):
                    if si in resp_idx:
                        sp.append((len(ids), len(ids) + len(s)))
                    ids += s
            else:              # single-turn: un span [len(prompt), L)
                p = tok.encode(r["prompt"], add_special_tokens=False)
                s = tok.encode(r["response"], add_special_tokens=False)
                ids = p + s
                sp = [(len(p), len(ids))]
            ids = ids[:max_len - 1]
            sp = [(a_, min(b_, len(ids))) for a_, b_ in sp if a_ < len(ids)]
            if not sp or len(ids) < 8 or all(b_ <= a_ for a_, b_ in sp):
                dropped += 1  # response vacía o doc degenerado tras truncar
                continue
            ids.append(eos)
            sp[-1] = (sp[-1][0], len(ids))   # ultimo span incluye EOS
            ids_all.extend(ids)
            lens.append(len(ids))
            nspans.append(len(sp))
            for a_, b_ in sp:
                starts.append(a_); ends.append(b_)
        except Exception:
            dropped += 1
            continue
    return ids_all, lens, starts, ends, nspans, dropped


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
    ids_all, lens, starts, ends, nspans = [], [], [], [], []
    dropped = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for ids, l, s, e, n, d in ex.map(encode_batch,
                                   [(c, args.tokenizer, args.seq_len) for c in chunks]):
            ids_all.extend(ids); lens.extend(l); starts.extend(s)
            ends.extend(e); nspans.extend(n); dropped += d
    np.savez_compressed(args.out,
                        ids=np.asarray(ids_all, dtype=np.int32),
                        lengths=np.asarray(lens, dtype=np.int32),
                        resp_starts=np.asarray(starts, dtype=np.int32),
                        resp_ends=np.asarray(ends, dtype=np.int32),
                        span_counts=np.asarray(nspans, dtype=np.int32),
                        eos_id=np.int64(eos))
    meta = {"n_pairs": len(lens), "n_tokens": len(ids_all), "dropped": dropped,
            "seq_len": args.seq_len, "eos_id": int(eos), "input": args.input}
    import json as j
    open(args.out + ".meta.json", "w").write(j.dumps(meta, indent=1))
    print(f"[sft_pretok] DONE {meta['n_pairs']} pares, {meta['n_tokens']/1e6:.1f}M tok, "
          f"dropped={dropped} -> {args.out}")


if __name__ == "__main__":
    main()
