#!/usr/bin/env python3
"""extract_eco_v7.py — FASE 3: extraer el subcorpus ECO del v7 curado.

Filtra train_corpus_v7_curated.jsonl por dominios ecológicos (eco, phylo, bioc,
climate, conserv, marine, plant, palaeo, microbio — NO genética médica/genom)
y escribe docs/results/eco_v7_subset.jsonl para el tagger de especies.

Uso: python3 scripts/extract_eco_v7.py --in <v7.jsonl> --out <eco_subset.jsonl>
"""
import argparse, json, sys

ECO_DOMAINS = {"eco", "phylo", "bioc", "climate", "conserv", "marine",
               "plant", "palaeo", "microbio"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = kept = 0
    with open(a.inp) as fi, open(a.out, "w") as fo:
        for line in fi:
            n += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("domain") in ECO_DOMAINS:
                fo.write(json.dumps({"text": d.get("text", ""), "pmid": d.get("pmid"),
                                     "domain": d.get("domain")}, ensure_ascii=False) + "\n")
                kept += 1
            if n % 200000 == 0:
                print(f"  {n} leidos, {kept} eco", flush=True)
    print(f"DONE: {n} leidos, {kept} eco ({(100*kept/max(n,1)):.1f}%)")
    return 0

if __name__ == "__main__":
    sys.exit(main())