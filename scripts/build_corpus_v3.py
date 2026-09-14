#!/usr/bin/env python3
"""build_corpus_v3.py — corpus v3: merge + dedup + balanceo por dominio.

Capas:
  - skeleton_v2 filtrado por audit (keep_idx menos leak_idx)
  - augv2 + synth_logic (tag src=synth)
  - unam chunks (lang=es, domain=unam_<area>)

Balanceo: cap por dominio (--domain-cap, default 40000) — reduce dominancia
de medgen sin inventar docs en dominios raros. Re-muestreo deterministico
(--seed) dentro de cada dominio excedente.

Salida: corpus_v3.jsonl con {pid, domain, text, etapas, src, lang} +
reporte de distribucion.
"""
import argparse, json, os, random
from collections import Counter

import numpy as np


def stream_jsonl(path, text_field="text"):
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--audit-dir", default=None,
                    help="dir con keep_idx.npy/leak_idx.npy (dedup+leakage)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="jsonl extra con field text (ya taggeados)")
    ap.add_argument("--unam", default=None,
                    help="unam_docs.jsonl de prep_embed_inputs")
    ap.add_argument("--domain-cap", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    keep = leak = None
    if args.audit_dir:
        kp = os.path.join(args.audit_dir, "keep_idx.npy")
        lp = os.path.join(args.audit_dir, "leak_idx.npy")
        keep = set(np.load(kp).tolist()) if os.path.exists(kp) else None
        leak = set(np.load(lp).tolist()) if os.path.exists(lp) else set()

    out_f = open(args.out, "w")
    stats = Counter(); dropped = Counter()

    # --- capa 1: skeleton filtrado ---
    n = 0
    for i, d in enumerate(stream_jsonl(args.skeleton)):
        if keep is not None and i not in keep:
            dropped["dedup"] += 1; continue
        if i in leak:
            dropped["leak"] += 1; continue
        d.setdefault("src", "skeleton"); d.setdefault("lang", "en")
        out_f.write(json.dumps(d, ensure_ascii=False) + "\n")
        stats[d.get("domain", "?")] += 1; n += 1
    print(f"[v3] skeleton: {n} docs ({dropped['dedup']} dedup, "
          f"{dropped['leak']} leak)", flush=True)

    # --- capa 2: extras (synth/aug) ---
    for path in args.extra:
        m = 0
        for d in stream_jsonl(path):
            d.setdefault("src", "synth"); d.setdefault("lang", "en")
            d.setdefault("domain", "synth")
            out_f.write(json.dumps(d, ensure_ascii=False) + "\n")
            stats[d["domain"]] += 1; m += 1
        print(f"[v3] extra {os.path.basename(path)}: {m}", flush=True)

    # --- capa 3: unam ---
    if args.unam and os.path.exists(args.unam):
        m = 0
        for d in stream_jsonl(args.unam):
            dom = d.get("domain") or "unam"
            d["domain"] = f"unam_{dom}" if not str(dom).startswith("unam") else dom
            out_f.write(json.dumps(d, ensure_ascii=False) + "\n")
            stats[d["domain"]] += 1; m += 1
        print(f"[v3] unam: {m} chunks", flush=True)

    out_f.close()

    # --- balanceo por dominio (segunda pasada, in-place reescritura) ---
    over = {d: c for d, c in stats.items() if c > args.domain_cap}
    if over:
        rng_keep = {}
        for d, c in over.items():
            # muestreo sin reemplazo deterministico por doc
            sel = set(rng.sample(range(c), args.domain_cap))
            rng_keep[d] = (sel, 0)
        tmp = args.out + ".tmp"
        seen = Counter()
        with open(args.out) as fi, open(tmp, "w") as fo:
            for line in fi:
                d = json.loads(line)
                dom = d.get("domain", "?")
                if dom in rng_keep:
                    sel, cnt = rng_keep[dom]
                    if cnt not in sel:
                        rng_keep[dom] = (sel, cnt + 1)
                        dropped[f"cap_{dom}"] += 1
                        continue
                    rng_keep[dom] = (sel, cnt + 1)
                fo.write(line); seen[dom] += 1
        os.replace(tmp, args.out)
        stats = seen
    total = sum(stats.values())
    report = {"total_docs": total, "domain_dist": dict(stats.most_common()),
              "domain_cap": args.domain_cap,
              "dropped": dict(dropped)}
    with open(args.out + ".report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[v3] total {total} docs -> {args.out}", flush=True)
    for d, c in stats.most_common(15):
        print(f"  {d}: {c} ({100*c/total:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
