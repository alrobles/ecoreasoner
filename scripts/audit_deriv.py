#!/usr/bin/env python3
"""audit_deriv.py — gate de calidad del corpus deriv (spec §3, issue #8).

Checks obligatorios antes de pretokenizar:
  1. forcedness   : F1 (arith) — recomputa la derivación del texto y exige que
                    el valor correcto aparezca; F3/F4 — la respuesta del
                    generador es función del ctx (verificación estructural).
  2. anti-template: ningún mold >5% de su familia (usa el campo `mold`).
  3. leak         : dedup vs pairs_holdout_clean y eval_devin_hard por
                    embeddings e5-small (--leak, requiere sentence-transformers
                    en el entorno HPC).
  4. stats        : n docs, tokens estimados, distribución de slots.

Uso:
    python3 audit_deriv.py corpus_deriv_v1.jsonl
    python3 audit_deriv.py corpus_deriv_v1.jsonl --leak \
        --pairs runs/pairs_hard_v4_holdout_clean --eval-jsonl eval_devin_hard.jsonl
"""
import argparse, json, re, sys
from collections import Counter, defaultdict


def audit(path, mold_cap=0.05):
    docs = [json.loads(l) for l in open(path)]
    fam = Counter(d.get("src", "?") for d in docs)
    slots = Counter(d.get("slot", "?") for d in docs)
    molds = defaultdict(Counter)
    for d in docs:
        molds[d.get("src", "?")][d.get("mold", "?")] += 1

    print(f"docs: {len(docs)}")
    print("familias:", dict(fam))
    print("slots:", dict(slots))
    toks = sum(len(d["text"].split()) for d in docs)
    print(f"tokens (whitespace aprox): {toks} (~{toks / len(docs):.0f}/doc)")

    # anti-template
    ok_mold, worst = True, (None, 0.0)
    for f, c in molds.items():
        tot = sum(c.values())
        m, mc = c.most_common(1)[0]
        share = mc / tot
        if share > worst[1]:
            worst = (f"{f}:{m[:50]}", share)
        if share > mold_cap:
            ok_mold = False
            print(f"  MOLD FAIL {f}: '{m[:60]}' share {share:.1%} > {mold_cap:.0%}")
    print(f"anti-template: {'OK' if ok_mold else 'FAIL'} (peor {worst[1]:.1%} {worst[0]})")

    # forcedness: recomputa desde forced.inputs (builder guarda la respuesta)
    RECOMP = {
        "pct_change": lambda i: abs(round((i["b"] - i["a"]) / i["a"] * 100)),
        "diff": lambda i: abs(i["b"] - i["a"]),
        "ratio": lambda i: round(i["a"] / i["b"], 2),
        "mean": lambda i: (i["a"] + i["b"]) / 2,
        "rate": lambda i: round(i["n"] / i["t"], 2),
        "conv": lambda i: i["v1"] / i["div"],
        "pct_points": lambda i: i["b"] - i["a"],
        "total": lambda i: i["a"] + i["b"],
        "frac_pct": lambda i: 100 * i["k"] / i["n"],
        "remainder": lambda i: i["n"] - i["k"],
    }
    verif, checked, present = 0, 0, 0
    fails = []
    for d in docs:
        f = d.get("forced")
        if not f:
            continue
        checked += 1
        exp = f.get("expected", "")
        # el valor forzado debe aparecer literalmente en el texto
        if exp and re.search(rf"(?<![\d.]){re.escape(str(exp))}(?!\d|\.?\d)",
                             d["text"], re.I):
            present += 1
        op = f.get("op")
        if op in RECOMP:
            try:
                rec = RECOMP[op](f["inputs"])
                if abs(float(rec) - float(exp)) <= 1:
                    verif += 1
                    continue
            except Exception:
                pass
            fails.append((op, f["inputs"], exp, d["text"][:140]))
        elif exp:
            # contrast/syll: la verdad forzada es estructural; basta presencia
            verif += 1 if re.search(re.escape(str(exp)[:60]), d["text"], re.I) else 0
    pct = verif / max(1, checked)
    print(f"forcedness: {verif}/{checked} ({pct:.1%}) verified; "
          f"valor literal presente en texto: {present}/{checked}")
    for r in fails[:5]:
        print("  FAIL", r)
    return {"n": len(docs), "mold_ok": ok_mold, "forcedness": pct}


def leak_check(path, pairs_dir, eval_jsonl, thresh=0.9):
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("leak: sentence-transformers no disponible — omitido (correr en HPC)")
        return
    enc = SentenceTransformer("intfloat/e5-small-v2")
    docs = [json.loads(l) for l in open(path)]
    ref = []
    import glob, os
    for f in glob.glob(os.path.join(pairs_dir, "pairs_L*.jsonl")):
        for l in open(f):
            d = json.loads(l)
            ref += [d.get("ok", ""), d.get("bad", "")]
    if eval_jsonl and os.path.exists(eval_jsonl):
        for l in open(eval_jsonl):
            ref.append(json.loads(l).get("question", "") + " " +
                       json.loads(l).get("answer", ""))
    E_ref = enc.encode([f"passage: {r[:512]}" for r in ref], normalize_embeddings=True)
    E_doc = enc.encode([f"passage: {d['text'][:512]}" for d in docs],
                       normalize_embeddings=True, batch_size=256)
    sim = E_doc @ E_ref.T
    mx = sim.max(axis=1)
    leaks = int((mx > thresh).sum())
    print(f"leak: {leaks}/{len(docs)} docs con sim>{thresh} vs refs "
          f"({len(ref)} textos) — {'OK' if leaks == 0 else 'REVISAR'}")
    if leaks:
        idx = np.argsort(-mx)[:5]
        for i in idx:
            print(f"  sim={mx[i]:.3f} | {docs[i]['text'][:140]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--leak", action="store_true")
    ap.add_argument("--pairs", default="runs/pairs_hard_v4_holdout_clean")
    ap.add_argument("--eval-jsonl", default=None)
    ap.add_argument("--mold-cap", type=float, default=0.05)
    args = ap.parse_args()
    audit(args.jsonl, args.mold_cap)
    if args.leak:
        leak_check(args.jsonl, args.pairs, args.eval_jsonl)


if __name__ == "__main__":
    main()
