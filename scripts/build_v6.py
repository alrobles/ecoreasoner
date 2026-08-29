#!/usr/bin/env python3
"""
build_v6.py — Construye train_corpus_v6.jsonl: v5 + full-text PMC 2025-2026.

Fuentes:
  1. train_corpus_v5.jsonl  (tag "v5", PRIORIDAD ALTA — ya tiene domain fino
     + fine-label; 1,905,424 docs)
  2. fulltext_2526/*.jsonl  (tag "pmc-2526", prioridad baja — 439K full-text
     nuevos 2025-2026 descargados del S3 OA)

Dedup por PMCID (campo comun fiable: en v5 los PMC traen pmcid y los abstracts
pmid; en el fetch los docs traen pmid REAL + pmcid). Regla: si un pmcid ya esta
en v5, gana v5 (ya etiquetado); el fetch SOLO anade pmcids ausentes.

Compatibilidad aguas abajo: pre_tokenize.py lee text/pmid/domain/year/source.

IMPORTANTE (regla de labor):
  - Se ejecuta SOLO via Slurm (CPU, sin GPU), NUNCA en login node.
  - No borra nada: escribe train_corpus_v6.jsonl + v6_report.json.
  - NO toca v5 ni el fetch (solo lectura).
"""
from __future__ import annotations
import argparse, json, glob, os, random, sys, time
from collections import Counter

# ──────────────────────────── guardas ────────────────────────────
if not os.environ.get("SLURM_JOB_ID"):
    sys.exit("ERROR: build_v6.py SOLO puede ejecutarse via Slurm (SLURM_JOB_ID ausente).")

# ──────────────────────────── taxonomia (igual v5) ────────────────────────────
def ilike_domain(text: str) -> str:
    t = text.lower()
    pats = {
        "eco":   ["ecolog", "biodivers", "species distribution", "maxent", "niche",
                  "occupancy", "habitat", "conservation", "invasion", "community ecology",
                  "population dynamic", "ecosystem"],
        "phylo": ["phylogen", "phylogeograph", "evolut", "natural selection", "adaptation",
                  "speciation", "divergence", "comparative method", "biogeograph",
                  "molecular evolution"],
        "genom": ["genom", "transcriptom", "population genetics", "gwas", "genome assembl",
                  "gene expression", "metagenom", "whole-genome", "exome", "sequencing"],
        "bioc":  ["bioinformatic", "machine learning", "deep learning", "neural network",
                  "computational", "modeling", "simulation", "statistical model",
                  "bayesian", "algorithm"],
        "microbio": ["microbi", "bacteria", "archaea", "virus", "pathogen", "infection",
                     "antimicrobial", "probiotic", "gut flora", "microorganism"],
        "medgen": ["disease", "clinical", "medical genetics", "variant", "mutation",
                   "hereditary", "cancer", "genetic disorder", "therapeutic", "patient"],
        "climate": ["climate change", "global warming", "temperature", "precipitation",
                    "carbon", "co2", "greenhouse", "climate model", "warming", "sea level"],
        "conserv": ["protected area", "management", "restoration", "endangered",
                    "threatened", "policy", "sustainable", "natural resource",
                    "wildlife management"],
        "plant":  ["plant", "botan", "flora", "crop", "agricultur", "forest", "tree",
                   "pollination", "vegetation", "herbivor"],
        "palaeo": ["human evolution", "archaeolog", "paleo", "fossil", "hominin",
                   "anthropolog", "primate evolution", "paleoecolog", "quaternary"],
        "marine": ["marine", "ocean", "coastal", "coral", "fish", "sea", "aquatic",
                   "freshwater", "plankton", "fishery"],
        "soil":   ["soil", "sediment", "nutrient cycling", "decomposition", "mycorrhiz",
                   "rhizosphere", "biogeochemistr", "nitrogen", "carbon cycling"],
    }
    best, best_score = "bioc", 0
    for dom, pats_d in pats.items():
        score = sum(1 for p in pats_d if p in t)
        if score > best_score:
            best, best_score = dom, score
    return best

def doc_key(d):
    """Clave de dedup: pmcid si existe, si no pmid (o 'pmid'=PMCID de v5)."""
    k = d.get("pmcid") or d.get("pmid")
    return str(k).strip().upper() if k else None

def read_jsonl(path, src_tag):
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except Exception: continue
        if not d.get("text") or len(str(d.get("text", ""))) < 50: continue
        doc = {
            "text": d["text"],
            "pmid": str(d.get("pmid") or d.get("pmcid") or ""),
            "domain": d.get("domain"),
            "year": d.get("year"),
            "source": src_tag,
        }
        if d.get("pmcid"): doc["pmcid"] = str(d["pmcid"])
        if d.get("license"): doc["license"] = d["license"]
        yield doc

def main():
    ap = argparse.ArgumentParser(description="Construye train_corpus_v6.jsonl (v5 + fulltext 2025-2026)")
    ap.add_argument("--v5", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_v5.jsonl")
    ap.add_argument("--fetch-dir", default="/beegfs/a474r867/ecoreasoner/data/fulltext_2526")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_v6.jsonl")
    ap.add_argument("--report", default="/beegfs/a474r867/ecoreasoner/data/v6_report.json")
    ap.add_argument("--max-per-domain", type=int, default=300000)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t0 = time.time()
    fetch_files = sorted(glob.glob(os.path.join(a.fetch_dir, "*.jsonl")))
    if not os.path.exists(a.v5):
        sys.exit(f"ERROR: no existe {a.v5}")
    if not fetch_files:
        sys.exit(f"ERROR: no hay shards fetch en {a.fetch_dir}")
    # v5 primero (alta prioridad), luego fetch (baja)
    paths_sources = [(a.v5, "v5")] + [(f, "pmc-2526") for f in fetch_files]
    print(f"[build_v6] fuentes: v5 + {len(fetch_files)} shards fetch = {len(paths_sources)}", flush=True)
    print(f"[build_v6] max_per_domain={a.max_per_domain}", flush=True)

    # pase 1: prioridad por clave (v5 gana) + conteo de claves
    priority = {}   # key -> tag ganador
    n_v5 = n_fetch = 0
    for path, tag in paths_sources:
        for d in read_jsonl(path, tag):
            k = doc_key(d)
            if k is None: continue
            if tag == "v5":
                n_v5 += 1
                priority.setdefault(k, tag)   # v5 gana si aparece
            else:
                n_fetch += 1
                if k not in priority:
                    priority[k] = tag          # solo si v5 no lo tiene
    print(f"[build_v6] pase prioridad OK: v5 docs={n_v5}, fetch docs={n_fetch}, claves unicas={len(priority)}", flush=True)

    # pase 2: emitir ganadores con balance
    seen = set(); out_docs = []
    n_dup = 0; dom_counts = Counter(); src_counts = Counter(); year_none = 0
    for path, tag in paths_sources:
        for d in read_jsonl(path, tag):
            k = doc_key(d)
            if k is None or priority.get(k) != tag:
                n_dup += 1
                continue
            if k in seen:
                n_dup += 1
                continue
            seen.add(k)
            if not d["domain"]:
                d["domain"] = ilike_domain(d["text"])
            if d["year"] is None:
                year_none += 1
            if a.max_per_domain and dom_counts[d["domain"]] >= a.max_per_domain:
                n_dup += 1
                continue
            dom_counts[d["domain"]] += 1
            src_counts[d["source"]] += 1
            out_docs.append(d)
    print(f"[build_v6] emitidos {len(out_docs)} docs unicos (dup/descartados {n_dup}, sin year {year_none})", flush=True)

    if a.shuffle:
        random.seed(a.seed); random.shuffle(out_docs)

    with open(a.out, "w", encoding="utf-8") as f:
        for d in out_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    report = {
        "version": 6,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": {"v5": a.v5, "fetch_dir": a.fetch_dir, "fetch_shards": len(fetch_files)},
        "output": a.out,
        "max_per_domain": a.max_per_domain,
        "docs_total": len(out_docs),
        "docs_from_v5": src_counts.get("v5", 0),
        "docs_new_fetch": src_counts.get("pmc-2526", 0),
        "dups_removed": n_dup,
        "year_missing": year_none,
        "domains": dict(dom_counts),
        "sources": dict(src_counts),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(a.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[build_v6] DONE {len(out_docs)} docs (nuevos {src_counts.get('pmc-2526',0)}) "
          f"en {time.time()-t0:.0f}s -> {a.out}", flush=True)
    print(json.dumps({"domains": report["domains"], "sources": report["sources"]}, indent=2))

if __name__ == "__main__":
    main()