#!/usr/bin/env python3
"""split_jsonl.py — divide un JSONL en train/val con seed fija.

Uso:
  python3 scripts/split_jsonl.py \
      --input data/skeleton/train_skeleton.jsonl \
      --train-out data/skeleton/train_skeleton_train.jsonl \
      --val-out data/skeleton/train_skeleton_val.jsonl \
      --train-ratio 0.95 --seed 7331
"""
import argparse, random, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--train-ratio", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=7331)
    args = ap.parse_args()

    if not (0 < args.train_ratio < 1):
        raise SystemExit("--train-ratio debe estar entre 0 y 1")

    with open(args.input) as f:
        lines = [ln for ln in f if ln.strip()]
    n = len(lines)
    if n == 0:
        raise SystemExit("input vacio")

    idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    n_train = int(n * args.train_ratio)
    train_idx = set(idx[:n_train])

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.train_out, "w") as ftrain, open(args.val_out, "w") as fval:
        for i, line in enumerate(lines):
            if i in train_idx:
                ftrain.write(line)
            else:
                fval.write(line)

    print(f"split {n} lineas -> train {n_train} ({n_train/n:.0%}), "
          f"val {n - n_train} ({(n - n_train)/n:.0%})  seed={args.seed}")


if __name__ == "__main__":
    main()
