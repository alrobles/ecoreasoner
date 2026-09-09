#!/usr/bin/env python3
"""ceiling_llada8b.py — techo LLaDA-8B-Base en la misma suite de discriminacion.

Uso:
  python3 scripts/ceiling_llada8b.py \
      --pairs runs/pairs_hard/pairs_L0.jsonl \
      --pairs-tokenizer /beegfs/a474r867/ecoreasoner/tokenizer/LLaDA \
      --out ceiling_llada8b.json \
      --device cuda

Dependencias:
- pip install transformers>=4.38.2 (la tarjeta de LLaDA pide 4.38.2)
- modelo descargado en HF: GSAI-ML/LLaDA-8B-Base (~16GB bf16)
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch


def _add_llada_path():
    root = Path(__file__).resolve().parents[1]
    llada = root / "third_party" / "LLaDA"
    if llada.exists() and str(llada) not in sys.path:
        sys.path.insert(0, str(llada))


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
        # trunca prompt por la izquierda y answer por la derecha
        if alen > max_seq // 2:
            answer = answer[:max_seq // 2]
        prompt = prompt[-(max_seq - len(answer)):]
    return prompt, answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--pairs-tokenizer", required=True,
                    help="tokenizer usado para construir los pares (LLaDA)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--mc-num", type=int, default=32,
                    help="muestras Monte Carlo de get_log_likelihood")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cfg-scale", type=float, default=0.0)
    ap.add_argument("--mask-id", type=int, default=126336,
                    help="mask token id en LLaDA (default 126336)")
    args = ap.parse_args()

    _add_llada_path()
    from get_log_likelihood import get_log_likelihood
    from transformers import AutoTokenizer, AutoModel

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[load] tokenizer pares: {args.pairs_tokenizer}")
    pairs_tok = AutoTokenizer.from_pretrained(args.pairs_tokenizer, trust_remote_code=True,
                                              local_files_only=True)

    print("[load] LLaDA-8B-Base ...")
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "cuda" else None,
    )
    if args.device != "cuda":
        model.to(args.device)
    model.eval()

    # LLaDA-8B usa el mismo tokenizer que LLaDA (completo)
    model_tok = pairs_tok
    model_vocab = model_tok.vocab_size

    pairs = _load_pairs(args.pairs, limit=args.limit or None)
    if not pairs:
        print("[fatal] sin pares", file=sys.stderr)
        return 1

    ok_wins, deltas = 0, []
    t0 = time.time()
    for i, (ctx_ids, ok_ids, bad_ids) in enumerate(pairs):
        # decodificar ids de pares (saltar especiales) y re-codificar con el
        # tokenizer del modelo. Para LLaDA roundtrip es exacto.
        ctx_text = pairs_tok.decode(ctx_ids, skip_special_tokens=True)
        ok_text = pairs_tok.decode(ok_ids, skip_special_tokens=True)
        bad_text = pairs_tok.decode(bad_ids, skip_special_tokens=True)

        ctx = model_tok.encode(ctx_text, add_special_tokens=False)
        ok = model_tok.encode(ok_text, add_special_tokens=False)
        bad = model_tok.encode(bad_text, add_special_tokens=False)

        ctx, ok = _trunc(ctx, ok, args.max_seq)
        ctx, bad = _trunc(ctx, bad, args.max_seq)

        ctx_t = torch.tensor(ctx, device=model.device, dtype=torch.long)
        ok_t = torch.tensor(ok, device=model.device, dtype=torch.long)
        bad_t = torch.tensor(bad, device=model.device, dtype=torch.long)

        # get_log_likelihood devuelve log-prob (mayor = mejor). Convertimos a
        # negativo para que la comparacion sea consistente con denoise_loss.
        try:
            ll_ok = get_log_likelihood(model, ctx_t, ok_t,
                                       mc_num=args.mc_num,
                                       batch_size=args.batch_size,
                                       cfg_scale=args.cfg_scale,
                                       mask_id=args.mask_id)
            ll_bad = get_log_likelihood(model, ctx_t, bad_t,
                                        mc_num=args.mc_num,
                                        batch_size=args.batch_size,
                                        cfg_scale=args.cfg_scale,
                                        mask_id=args.mask_id)
        except Exception as e:
            print(f"[warn] par {i} fallo: {e}")
            continue

        score_ok = -ll_ok
        score_bad = -ll_bad
        if score_ok < score_bad:
            ok_wins += 1
        deltas.append(score_bad - score_ok)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pairs)} acc parcial={ok_wins/(i+1):.4f}")

    acc = ok_wins / len(pairs) if pairs else 0.0
    report = {
        "model": "GSAI-ML/LLaDA-8B-Base",
        "pairwise_acc": round(acc, 4),
        "n_pairs": len(pairs),
        "mean_delta": round(float(sum(deltas) / len(deltas)) if deltas else 0.0, 5),
        "mc_num": args.mc_num,
        "batch_size": args.batch_size,
        "cfg_scale": args.cfg_scale,
        "elapsed_s": round(time.time() - t0, 2),
        "pairs_file": args.pairs,
    }
    print(json.dumps(report, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
