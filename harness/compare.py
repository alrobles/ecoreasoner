#!/usr/bin/env python3
"""compare.py — tabla A/B/N sobre runs/index.jsonl con veredicto GO/NO-GO
(Fase 3, F0).

El umbral GO lo fija el baseline: se considera GO si el pairwise_acc del run
supera el del baseline (--base tag) en al menos --margin (default 0.05).

Uso:
  python harness/compare.py --index runs/index.jsonl [--base f0-baseline] [--margin 0.05]
"""
import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="runs/index.jsonl")
    ap.add_argument("--base", default=None, help="tag baseline para el umbral GO")
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()

    if not Path(args.index).exists():
        print(f"[fatal] no existe {args.index}", file=sys.stderr)
        sys.exit(2)

    rows = [json.loads(l) for l in Path(args.index).read_text().splitlines() if l.strip()]
    if not rows:
        print("(sin runs en el índice)")
        return

    base_acc = None
    if args.base:
        base_acc = next((r.get("pairwise_acc") for r in rows if r.get("tag") == args.base), None)
        if base_acc is None:
            print(f"[warn] baseline '{args.base}' no está en el índice")

    print(f"{'tag':<22} {'acc':>6} {'delta':>9} {'status':<8}")
    print("-" * 50)
    for r in rows:
        acc = r.get("pairwise_acc")
        delta = r.get("mean_delta")
        acc_s = f"{acc:.3f}" if acc is not None else "  -"
        d_s = f"{delta:+.4f}" if delta is not None else "   -"
        status = "GO" if (acc is not None and base_acc is not None
                          and acc >= base_acc + args.margin) else \
                 ("NO-GO" if acc is not None and base_acc is not None else "")
        print(f"{r.get('tag','?'):<22} {acc_s:>6} {d_s:>9} {status:<8}")

    if base_acc is not None:
        print(f"\nbaseline {args.base}: acc={base_acc:.3f} | GO si acc >= {base_acc + args.margin:.3f}")


if __name__ == "__main__":
    main()