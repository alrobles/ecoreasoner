#!/usr/bin/env python3
"""consolidate_b1_traces.py — fusiona trazas de destilacion B1 (b1 + litdump).

Fuentes:
  - data/sci_v2_b1.jsonl        (1711 trazas, metadatos ricos: pmid/doi/journal/mesh/usage)
  - data/sci_v2_existing.jsonl  (4287 trazas litdump, MAYORIA sin pmid)

Reglas:
  - Trazas CON pmid real -> dedup por pmid (b1 gana, tiene metadatos ricos).
  - Trazas SIN pmid       -> se conservan TODAS (no colapsar en clave vacia).
  - Shuffle con seed fijo (42) para reproducibilidad.

Uso:
  python3 consolidate_b1_traces.py [--b1 data/sci_v2_b1.jsonl]
          [--lit data/sci_v2_existing.jsonl] [--out data/sci_v2_B1_consolidated.jsonl]
"""
import argparse, json, random
from collections import Counter

def read_jsonl(path):
    for line in open(path):
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except Exception:
                pass

def nopmid(d):
    return not str(d.get("pmid") or "").strip()

def norm(d, src):
    return {
        "pmid": str(d.get("pmid") or ""),
        "title": d.get("title", ""),
        "journal": d.get("journal", ""),
        "pub_year": d.get("pub_year"),
        "search_query": d.get("search_query", ""),
        "context": (d.get("context") or "").strip(),
        "reasoning": (d.get("reasoning") or "").strip(),
        "code": (d.get("code") or "").strip(),
        "code_valid": bool(d.get("code_valid")),
        "teacher": d.get("teacher") or d.get("model") or "litdump",
        "source": src,
        "source_file": d.get("source_file", ""),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", default="/beegfs/a474r867/ecoreasoner/data/sci_v2_b1.jsonl")
    ap.add_argument("--lit", default="/beegfs/a474r867/ecoreasoner/data/sci_v2_existing.jsonl")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/sci_v2_B1_consolidated.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    b1 = list(read_jsonl(args.b1))
    lit = list(read_jsonl(args.lit))
    print(f"b1: {len(b1)} | litdump: {len(lit)}", flush=True)

    # 1) con pmid -> dedup (b1 gana)
    by_pmid = {}
    for d in lit:
        if nopmid(d):
            continue
        by_pmid.setdefault(str(d["pmid"]), norm(d, "lit"))
    for d in b1:
        if nopmid(d):
            continue
        by_pmid[str(d["pmid"])] = norm(d, "b1")

    merged = list(by_pmid.values())

    # 2) sin pmid -> conservar todas
    for d in lit:
        if nopmid(d):
            merged.append(norm(d, "lit-nopmid"))

    random.seed(args.seed)
    random.shuffle(merged)

    n_with = sum(1 for m in merged if m["pmid"])
    print(f"total: {len(merged)} | con pmid: {n_with} | sin pmid: {len(merged)-n_with}", flush=True)
    print("code_valid:", dict(Counter(m["code_valid"] for m in merged)), flush=True)
    print("por fuente:", dict(Counter(m["source"] for m in merged)), flush=True)

    with open(args.out, "w") as f:
        for m in merged:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"escrito: {args.out}", flush=True)

if __name__ == "__main__":
    main()