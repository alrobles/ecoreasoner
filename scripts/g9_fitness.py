#!/usr/bin/env python3
"""g9_fitness.py — tabla de fitness de la generacion G9 (dev v3_eval).

Para cada runs/g9-*/battery_logicdiff/dense/battery_L3.json:
  fitness = 0.5*L3 + 0.5*min(acc_number, acc_negation)   (convencion GA)
  fitness_l3 = L3                                       (dual, leccion nemotron)

Replica-seed: media de los brazos -sK del mismo gen (corrlo s1/s2).
Uso: python3 g9_fitness.py [--runs DIR]
"""
import argparse
import glob
import json
import os
import re


def load_arm(run_dir):
    f = os.path.join(run_dir, "battery_logicdiff", "dense", "battery_L3.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))["discrimination"]
    sub = d.get("l3_subtype_acc", {})
    num = sub.get("number", {}).get("acc")
    neg = sub.get("negation", {}).get("acc")
    l3 = d.get("pairwise_acc")
    if num is None or neg is None or l3 is None:
        return None
    return {"l3": l3, "num": num, "neg": neg, "n": d.get("n_pairs"),
            "fit": 0.5 * l3 + 0.5 * min(num, neg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/beegfs/a474r867/ecoreasoner/runs")
    ap.add_argument("--glob", default="g9-*",
                    help="patron de runs a evaluar (g9-*, g10-*)")
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(a.runs, a.glob))):
        tag = os.path.basename(d)
        done = os.path.exists(os.path.join(d, "training_complete.flag"))
        r = load_arm(d)
        rows.append((tag, done, r))
    print(f"{'run':16s} {'done':5s} {'L3':>6s} {'num':>6s} {'neg':>6s} "
          f"{'FIT':>6s}")
    arms = {}
    for tag, done, r in rows:
        if r:
            print(f"{tag:16s} {'Y' if done else 'run':5s} {r['l3']:.4f} "
                  f"{r['num']:.4f} {r['neg']:.4f} {r['fit']:.4f}")
            arm = re.sub(r"-s\d+$", "", tag)
            arms.setdefault(arm, []).append(r)
        else:
            print(f"{tag:16s} {'Y' if done else 'run':5s} "
                  f"{'--':>6s} {'--':>6s} {'--':>6s} {'--':>6s}")
    print("\n== media por brazo (seeds) ==")
    print(f"{'brazo':14s} {'L3':>6s} {'num':>6s} {'neg':>6s} {'FIT':>6s} {'n_r'}")
    for arm, rs in sorted(arms.items(),
                          key=lambda kv: -sum(r["fit"] for r in kv[1])
                          / len(kv[1])):
        m = {k: sum(r[k] for r in rs) / len(rs) for k in ("l3", "num", "neg", "fit")}
        print(f"{arm:14s} {m['l3']:.4f} {m['num']:.4f} {m['neg']:.4f} "
              f"{m['fit']:.4f} {len(rs)}")


if __name__ == "__main__":
    main()
