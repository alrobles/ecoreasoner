#!/usr/bin/env python3
"""leaderboard.py — cosecha baterias dense de runs EvoG0/B-series y rankea.

Fitness (protocolo EvoG0): 0.5*L3_acc + 0.5*min(number, negation).
El promedio L3 solo puede ganarlo direction_word; el maximin castiga la
frontera real. Seleccion: top-2 por fitness -> siguiente generacion.

Uso:
    python3 scripts/leaderboard.py --runs /beegfs/.../runs
    python3 scripts/leaderboard.py --runs runs --glob 'g0-*/battery_logicdiff/dense'
"""
import argparse, glob, json, os, re, statistics, sys


def harvest(runs_dir, pat):
    rows = []
    for bp in sorted(glob.glob(os.path.join(runs_dir, pat, "battery.json"))):
        try:
            d = json.load(open(bp))
        except Exception as e:
            print(f"[skip] {bp}: {e}", file=sys.stderr)
            continue
        run = bp.split("/battery_logicdiff/")[0].rstrip("/").split("/")[-1]
        ppl = None
        pj = os.path.join(runs_dir, run, "ppl_proxy.json")
        if os.path.exists(pj):
            try:
                ppl = json.load(open(pj)).get("ppl_proxy")
            except Exception:
                pass
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
            "fitness": fit, "ppl": ppl,
        })
    return rows


def group_replicas(rows):
    """Agrupa corridas `nombre-sK` (misma config, seed k) bajo `nombre`.
    La fila de grupo = MEDIA entre seeds (seleccion por media, no best-of-N);
    `sd` = desviacion del fitness entre seeds. `n3` = n de pares (identico
    entre replicas; se toma el maximo)."""
    groups = {}
    for r in rows:
        groups.setdefault(re.sub(r"-s\d+$", "", r["run"]), []).append(r)
    agg = []
    for base, mem in groups.items():
        a = {"run": base, "n_seeds": len(mem)}
        for k in ("L0", "L1", "L2", "L3", "number", "negation", "direction",
                  "fitness", "ppl"):
            v = [m[k] for m in mem if m[k] is not None]
            a[k] = statistics.mean(v) if v else None
        a["n3"] = max((m["n3"] for m in mem if m["n3"] is not None), default=None)
        fv = [m["fitness"] for m in mem if m["fitness"] is not None]
        a["sd"] = statistics.pstdev(fv) if len(fv) > 1 else (0.0 if fv else None)
        a["members"] = mem
        agg.append(a)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--glob", default="*/battery_logicdiff/dense")
    ap.add_argument("--out", default=None, help="jsonl de salida (opcional)")
    args = ap.parse_args()

    rows = harvest(args.runs, args.glob)
    agg = group_replicas(rows)
    agg.sort(key=lambda r: (r["fitness"] is None, -(r["fitness"] or 0)))

    hdr = f"{'run':<28} {'n':>2} {'L0':>6} {'L1':>6} {'L2':>6} {'L3':>6} {'num':>6} {'neg':>6} {'dir':>6} {'FIT':>6} {'sd':>5} {'PPL':>7}"
    print(hdr); print("-" * len(hdr))
    f = lambda x: f"{x:.3f}" if isinstance(x, float) else "  -  "
    fp = lambda x: f"{x:.1f}" if isinstance(x, float) else "   -   "
    for a in agg:
        print(f"{a['run']:<28} {a['n_seeds']:>2} {f(a['L0'])} {f(a['L1'])} {f(a['L2'])} "
              f"{f(a['L3'])} {f(a['number'])} {f(a['negation'])} {f(a['direction'])} "
              f"{f(a['fitness'])} {f(a['sd'])} {fp(a['ppl'])}")
        for m in a["members"] if a["n_seeds"] > 1 else []:
            print(f"  · {m['run']:<25} {'':>2} {f(m['L0'])} {f(m['L1'])} {f(m['L2'])} "
                  f"{f(m['L3'])} {f(m['number'])} {f(m['negation'])} {f(m['direction'])} "
                  f"{f(m['fitness'])} {'':>5} {fp(m['ppl'])}")

    if args.out:
        with open(args.out, "w") as fh:
            for a in agg:
                a2 = {k: v for k, v in a.items() if k != "members"}
                fh.write(json.dumps(a2) + "\n")
        print(f"\n-> {args.out} ({len(agg)} configs, {len(rows)} runs)")


if __name__ == "__main__":
    main()
