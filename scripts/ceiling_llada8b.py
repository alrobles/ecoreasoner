#!/usr/bin/env python3
"""ceiling_llada8b.py — evalua LLaDA-8B-Base en el mismo suite_smoke.

Uso:
  python3 scripts/ceiling_llada8b.py \
      --pairs runs/pairs_hard/pairs_L0.jsonl \
      --config harness/configs/f0-span-esqueleto.yaml \
      --out ceiling_llada8b.json

Notas:
- LLaDA-8B requiere trust_remote_code=True.
- En pro6000 (96GB) con bf16 deberia caber; si no, bajar max_len y batch.
- Este es un stub; Hermes debe completar la funcion _score_pair y el sampler.
"""
import argparse
import json
import sys
from pathlib import Path

import torch


def _load_pairs(path, limit=None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                d = json.loads(line)
                yield d["ctx"], d["ok"], d["bad"]
            except json.JSONDecodeError:
                continue


def _score_pair(model, tokenizer, ctx, cand, device):
    """TODO (Hermes): implementar denoising loss de LLaDA para candidato.

    LLaDA-8B expone funciones de generacion y likelihood en su repo.
    Mirar ML-GSAI/LLaDA/get_log_likelihood.py y generate.py.
    """
    raise NotImplementedError(
        "ceiling_llada8b._score_pair: copiar/adaptar get_log_likelihood de "
        "third_party/LLaDA"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--config", default="harness/configs/f0-span-esqueleto.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=1,
                    help="batch de pares; 1 para no saturar VRAM")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    print("cargando LLaDA-8B-Base...")
    tok = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "cuda" else None,
    )
    model.eval()

    ok_wins = 0
    n = 0
    deltas = []
    device = torch.device(args.device)

    for ctx, ok, bad in _load_pairs(args.pairs, limit=args.limit or None):
        try:
            lok = _score_pair(model, tok, ctx, ok, device)
            lbad = _score_pair(model, tok, ctx, bad, device)
        except NotImplementedError:
            print("ERROR: implementar _score_pair antes de correr")
            return 1
        if lok < lbad:
            ok_wins += 1
        deltas.append(float(lbad - lok))
        n += 1

    acc = ok_wins / n if n else 0.0
    mean_delta = sum(deltas) / n if n else 0.0
    report = {
        "model": "GSAI-ML/LLaDA-8B-Base",
        "pairwise_acc": round(acc, 4),
        "mean_delta": round(mean_delta, 4),
        "n_pairs": n,
        "pairs_file": args.pairs,
    }
    print(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
