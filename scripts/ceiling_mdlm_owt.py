#!/usr/bin/env python3
"""ceiling_mdlm_owt.py — techo MDLM-OWT (~130M) en la misma suite.

Uso:
  python3 scripts/ceiling_mdlm_owt.py \
      --pairs runs/pairs_hard/pairs_L0.jsonl \
      --pairs-tokenizer /beegfs/a474r867/ecoreasoner/tokenizer/LLaDA \
      --out ceiling_mdlm_owt.json \
      --device cuda

MDLM-OWT usa tokenizer gpt2 y AutoModelForMaskedLM. Los pares fueron
construidos con LLaDA tokenizer, asi que decodificamos a texto y
re-codificamos con gpt2.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_pairs(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not all(k in d for k in ("ctx", "ok", "bad")):
                continue
            out.append((d["ctx"], d["ok"], d["bad"]))
    return out


def _trunc(prompt, answer, max_seq):
    plen = len(prompt)
    alen = len(answer)
    total = plen + alen
    if total > max_seq:
        if alen > max_seq // 2:
            answer = answer[:max_seq // 2]
        prompt = prompt[-(max_seq - len(answer)):]
    return prompt, answer


def _score_pair(model, tokenizer, ctx_ids, cand_ids, mask_p, device, max_seq=1024):
    """Devuelve la masked LM loss del cand dado ctx. Menor = mejor."""
    ctx, cand = _trunc(list(ctx_ids), list(cand_ids), max_seq)
    seq = torch.tensor(ctx + cand, device=device, dtype=torch.long)
    cand_start = len(ctx)
    cand_len = len(cand)
    if cand_len == 0:
        return float("inf")
    n = max(1, int(mask_p * cand_len))
    # posiciones a enmascarar dentro del candidato
    perm = torch.randperm(cand_len)[:n]
    pos = cand_start + perm
    masked = seq.clone()
    mask_token = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.unk_token_id
    masked[pos] = mask_token
    labels = torch.full_like(seq, -100)
    labels[pos] = seq[pos]
    with torch.no_grad():
        out = model(masked.unsqueeze(0), labels=labels.unsqueeze(0))
        # out.loss ya promedia solo las posiciones con labels != -100
        return out.loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--pairs-tokenizer", required=True,
                    help="tokenizer usado para construir los pares (LLaDA)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--mask-p", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7331)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForMaskedLM

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[load] tokenizer pares: {args.pairs_tokenizer}")
    pairs_tok = AutoTokenizer.from_pretrained(args.pairs_tokenizer, trust_remote_code=True,
                                              local_files_only=True)

    print("[load] kuleshov-group/mdlm-owt ...")
    model_tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForMaskedLM.from_pretrained(
        "kuleshov-group/mdlm-owt",
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()

    pairs = _load_pairs(args.pairs, limit=args.limit or None)
    if not pairs:
        print("[fatal] sin pares", file=sys.stderr)
        return 1

    ok_wins, deltas = 0, []
    t0 = time.time()
    for i, (ctx_ids, ok_ids, bad_ids) in enumerate(pairs):
        ctx_text = pairs_tok.decode(ctx_ids, skip_special_tokens=True)
        ok_text = pairs_tok.decode(ok_ids, skip_special_tokens=True)
        bad_text = pairs_tok.decode(bad_ids, skip_special_tokens=True)

        ctx = model_tok.encode(ctx_text, add_special_tokens=False)
        ok = model_tok.encode(ok_text, add_special_tokens=False)
        bad = model_tok.encode(bad_text, add_special_tokens=False)

        try:
            l_ok = _score_pair(model, model_tok, ctx, ok, args.mask_p, args.device, args.max_seq)
            l_bad = _score_pair(model, model_tok, ctx, bad, args.mask_p, args.device, args.max_seq)
        except Exception as e:
            print(f"[warn] par {i} fallo: {e}")
            continue

        if l_ok < l_bad:
            ok_wins += 1
        deltas.append(l_bad - l_ok)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pairs)} acc parcial={ok_wins/(i+1):.4f}")

    acc = ok_wins / len(pairs) if pairs else 0.0
    report = {
        "model": "kuleshov-group/mdlm-owt",
        "pairwise_acc": round(acc, 4),
        "n_pairs": len(pairs),
        "mean_delta": round(float(sum(deltas) / len(deltas)) if deltas else 0.0, 5),
        "mask_p": args.mask_p,
        "elapsed_s": round(time.time() - t0, 2),
        "pairs_file": args.pairs,
    }
    print(json.dumps(report, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
