#!/usr/bin/env python3
"""ceiling_mdlm_owt.py — evalua kuleshov-group/mdlm-owt (~130M) en suite_smoke.

Uso:
  python3 scripts/ceiling_mdlm_owt.py \
      --pairs runs/pairs_hard/pairs_L0.jsonl \
      --out ceiling_mdlm_owt.json

TODO (Hermes): MDLM-OWT es un AutoModelForMaskedLM con tokenizer gpt2.
Completar _score_pair usando su API o la de third_party/mdlm.
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
    """TODO (Hermes): denoising loss del MDLM-OWT para candidato.

    MDLM-OWT se carga con transformers AutoModelForMaskedLM.
    Ejemplo:
      from transformers import AutoModelForMaskedLM, AutoTokenizer
      model = AutoModelForMaskedLM.from_pretrained('kuleshov-group/mdlm-owt')
      tokenizer = AutoTokenizer.from_pretrained('gpt2')
    """
    raise NotImplementedError(
        "ceiling_mdlm_owt._score_pair: adaptar log-likelihood del repo "
        "third_party/mdlm o usar masked LM loss con MASK token."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    print("cargando kuleshov-group/mdlm-owt...")
    # MDLM usa tokenizer gpt2 segun su HF card
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForMaskedLM.from_pretrained(
        "kuleshov-group/mdlm-owt",
        torch_dtype=torch.bfloat16,
    ).to(args.device)
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
        "model": "kuleshov-group/mdlm-owt",
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
