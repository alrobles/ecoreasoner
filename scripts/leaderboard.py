#!/usr/bin/env python3
"""leaderboard.py — cosecha baterias dense de runs EvoG0/B-series y rankea.

Fitness (protocolo EvoG0): 0.5*L3_acc + 0.5*min(number, negation).
El promedio L3 solo puede ganarlo direction_word; el maximin castiga la
frontera real. Seleccion: top-2 por fitness -> siguiente generacion.

Uso:
    python3 scripts/leaderboard.py --runs /beegfs/.../runs
    python3 scripts/leaderboard.py --runs runs --glob 'g0-*/battery_logicdiff/dense'
"""
import argparse, glob, json, os, sys


def harvest(runs_dir, pat):
    rows = []
    for bp in sorted(glob.glob(os.path.join(runs_dir, pat, "battery.json"))):
        try:
            d = json.load(open(bp))
        except Exception as e:
            print(f"[skip] {bp}: {e}", file=sys.stderr)
            continue
        run = bp.split("/runs/")[-1].split("/battery")[0] if "/runs/" in bp else bp
        lv = d.get("levels", {})
        subs = lv.get("L3", {}).get("l3_subtype_acc", {})
        num = subs.get("number", {}).get("acc")
        neg = subs.get("negation", {}).get("acc")
        l3 = lv.get("L3", {}).get("pairwise_acc")
        weak = [x for x in (num, neg) if x is not None]
        fit = None
        if l3 is not None and weak:
            fit = 0.5 * l3 + 0.5 * min(weak)
        rows.append({
            "run": run,
            "L0": lv.get("L0", {}).get("pairwise_acc"),
            "L1": lv.get("L1", {}).get("pairwise_acc"),
            "L2": lv.get("L2", {}).get("pairwise_acc"),
            "L3": l3,
            "n3": lv.get("L3", {}).get("n_pairs"),
            "number": num, "negation": neg,
            "direction": subs.get("direction_word", {}).get("acc"),
            "fitness": fit,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--glob", default="*/battery_logicdiff/dense")
    ap.add_argument("--out", default=None, help="jsonl de salida (opcional)")
    args = ap.parse_args()

    rows = harvest(args.runs, args.glob)
    rows.sort(key=lambda r: (r["fitness"] is None, -(r["fitness"] or 0)))

    hdr = f"{'run':<28} {'L0':>6} {'L1':>6} {'L2':>6} {'L3':>6} {'num':>6} {'neg':>6} {'dir':>6} {'FIT':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        f = lambda x: f"{x:.3f}" if isinstance(x, float) else "  -  "
        print(f"{r['run']:<28} {f(r['L0'])} {f(r['L1'])} {f(r['L2'])} "
              f"{f(r['L3'])} {f(r['number'])} {f(r['negation'])} {f(r['direction'])} "
              f"{f(r['fitness'])}")

    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"\n-> {args.out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
