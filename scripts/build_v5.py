#!/usr/bin/env python3
"""
build_v5.py — Construye train_corpus_v5.jsonl (Opción A: corpus limpio).

Reemplaza a v4 (concat crudo sin dedup). Objetivos:
  1. Dedup GLOBAL por pmid, priorizando full-text sobre abstract.
     Regla: si un pmid aparece en v3 y en PMC-v4, gana el full-text (más rico).
  2. Asignar domain a TODOS los docs (los PMC v4 no traen domain -> inferir por ILIKE).
  3. Balance por dominio opcional (--max-per-domain) para que eco no quede en minoría.
  4. Reporte JSON de construcción (docs, dups eliminados, % por domain/source).

Compatibilidad aguas abajo:
  - `pre_tokenize.py` lee `d.get("text", "")` -> el v5 usa {"text", "pmid", "domain",
    "year", "source"} igual que v3, con los PMC con domain inferido.

IMPORTANTE (regla de labor):
  - Este script se ejecuta SOLO vía Slurm (CPU, sin GPU), NUNCA en login node.
  - No borra nada: escribe train_corpus_v5.jsonl + v5_report.json en paralelo.
  - NO toca el v4 ni el training activo (moe-v4-bw entrena sobre v4).

Uso (Slurm CPU, no login):
  python3 build_v5.py \
      --v3  /beegfs/a474r867/ecoreasoner/data/train_corpus_v3.jsonl \
      --pmc /beegfs/a474r867/ecoreasoner/data/pmc_corpus_v4 \
      --out /beegfs/a474r867/ecoreasoner/data/train_corpus_v5.jsonl \
      --report /beegfs/a474r867/ecoreasoner/data/v5_report.json \
      [--max-per-domain 300000] [--workers 16] [--shuffle] [--seed 42]
"""
from __future__ import annotations
import argparse, json, glob, os, random, re, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

# ──────────────────────────── guardas ────────────────────────────
# Regla: nada de ejecución en login node. Slurm define SLURM_JOB_ID.
if not os.environ.get("SLURM_JOB_ID"):
    sys.exit("ERROR: build_v5.py SOLO puede ejecutarse via Slurm (SLURM_JOB_ID ausente). "
             "Lanza con sbatch build_v5.slurm, no en login node.")

# ──────────────────────────── helpers ────────────────────────────
def ilike_domain(text: str) -> str:
    """Infiere el dominio científico por patrones (taxonomía 12 dominios).

    Basado en el probe real sobre train_corpus_v5.jsonl (2026-08-28): el corpus
    tiene genética médica, microbioma, plantas, clima, conservación... que el
    esquema de 4 dominios original (eco/phylo/genom/bioc de mine_pubmed_duckdb)
    NO capturaba (caían en 'genom'/'bioc' por defecto).
    El orden de evaluación importa: se puntúa cada patrón y gana el de más hits.
    """
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
    best, best_score = "bioc", 0   # default razonable (bioinformática/computacional)
    for dom, pats_d in pats.items():
        score = sum(1 for p in pats_d if p in t)
        if score > best_score:
            best, best_score = dom, score
    return best

def read_jsonl(path, src_tag, meta=None):
    """Generador streaming: doc -> {"text","pmid","domain","year","source", [+license,+pmcid]}."""
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except Exception: continue
        pmid = d.get("pmid") or d.get("pmcid")
        if not pmid or not d.get("text"): continue
        doc = {
            "text": d["text"],
            "pmid": str(pmid),
            "domain": d.get("domain"),
            "year": d.get("year"),
            "source": src_tag,
        }
        # preservar metadata extra de provenance si existe (PMC v4 trae license)
        if d.get("license"): doc["license"] = d["license"]
        if d.get("pmcid"): doc["pmcid"] = str(d["pmcid"])
        yield doc

# ──────────────────────────── dedup + balance ────────────────────────────
def dedup_docs(paths_sources, priority, max_per_domain, extra_field=None):
    """Pase unico streaming con dedup global por pmid.
    paths_sources: list[(path, tag)] priorizada (full-text primero).
    priority: dict pmid -> tag ganador (de mayor a menor prioridad).
    Devuelve (docs_ordenados, reporte_parcial) con balance max_per_domain.
    """
    seen_pmid = set()
    dom_counts = Counter()
    src_counts = Counter()
    out_docs = []
    dups = 0
    # 1) pase de prioridad: para cada pmid, recordar el tag de mayor prioridad
    #    (el que gana si aparece en varias fuentes)
    for path, tag in paths_sources:
        for d in read_jsonl(path, tag):
            p = d["pmid"]
            cur = priority.get(p)
            if cur is None:
                priority[p] = tag
            else:
                # orden de prioridad: v3 < pmc-v4 (full-text gana)
                rank = {"v3": 1, "pmc-v4": 2}.get(tag, 0)
                if rank > {"v3": 1, "pmc-v4": 2}.get(cur, 0):
                    priority[p] = tag
    # 2) pase de emisión: emitir SOLO los docs cuyo tag == priority[pmid] (el ganador)
    for path, tag in paths_sources:
        for d in read_jsonl(path, tag):
            p = d["pmid"]
            if priority.get(p) != tag:
                continue  # pierde contra un full-text del mismo pmid en otra fuente
            if p in seen_pmid:
                dups += 1
                continue
            seen_pmid.add(p)
            # dominio: inferir si falta
            if not d["domain"]:
                d["domain"] = ilike_domain(d["text"])
            # balance por dominio
            if max_per_domain and dom_counts[d["domain"]] >= max_per_domain:
                dups += 1  # fuera de cuota -> cuenta como descartado por balance
                continue
            dom_counts[d["domain"]] += 1
            src_counts[d["source"]] += 1
            out_docs.append(d)
    return out_docs, {"dups": dups, "domains": dict(dom_counts), "sources": dict(src_counts)}

def main():
    ap = argparse.ArgumentParser(description="Construye train_corpus_v5.jsonl (Opción A limpia)")
    ap.add_argument("--v3", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_v3.jsonl")
    ap.add_argument("--pmc", default="/beegfs/a474r867/ecoreasoner/data/pmc_corpus_v4")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_v5.jsonl")
    ap.add_argument("--report", default="/beegfs/a474r867/ecoreasoner/data/v5_report.json")
    ap.add_argument("--max-per-domain", type=int, default=0, help="0=sin balance")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t0 = time.time()
    # fuentes: PMC-v4 primero (full-text gana por prioridad), luego v3
    pmc_files = sorted(glob.glob(os.path.join(a.pmc, "shard_*.jsonl")))
    if not pmc_files:
        sys.exit(f"ERROR: no hay shards PMC en {a.pmc}")
    paths_sources = [(f, "pmc-v4") for f in pmc_files] + [(a.v3, "v3")]

    print(f"[build_v5] fuentes: {len(pmc_files)} shards PMC + v3 = {len(paths_sources)}", flush=True)
    print(f"[build_v5] max_per_domain={a.max_per_domain} workers={a.workers}", flush=True)

    # pase 1: prioridad por pmid (barrido streaming de todas las fuentes)
    # (el dedup_docs ya lo hace internamente con su propio priority, pero aqui
    #  lo hacemos explícito para reportar)
    priority: dict[str, str] = {}
    # pase de prioridad rápido (sin emitir): marcar ganador por pmid
    for path, tag in paths_sources:
        for d in read_jsonl(path, tag):
            p = d["pmid"]
            cur = priority.get(p)
            rank_new = {"pmc-v4": 2, "v3": 1}.get(tag, 0)
            rank_old = {"pmc-v4": 2, "v3": 1}.get(cur, 0) if cur else 0
            if rank_new > rank_old:
                priority[p] = tag
    print(f"[build_v5] pase prioridad OK: {len(priority)} pmids unicos vistos", flush=True)

    # pase 2: emitir ganadores con balance
    seen = set()
    out_docs_all = []
    n_dup = 0
    dom_counts = Counter(); src_counts = Counter()
    for path, tag in paths_sources:
        for d in read_jsonl(path, tag):
            p = d["pmid"]
            if priority.get(p) != tag:    # pierde contra full-text en otra fuente
                n_dup += 1
                continue
            if p in seen:
                n_dup += 1
                continue
            seen.add(p)
            if not d["domain"]:
                d["domain"] = ilike_domain(d["text"])
            if a.max_per_domain and dom_counts[d["domain"]] >= a.max_per_domain:
                n_dup += 1  # fuera de cuota
                continue
            dom_counts[d["domain"]] += 1
            src_counts[d["source"]] += 1
            out_docs_all.append(d)
    print(f"[build_v5] emitidos {len(out_docs_all)} docs unicos (dup/descartados {n_dup})", flush=True)

    # shuffle (opcional)
    if a.shuffle:
        random.seed(a.seed)
        random.shuffle(out_docs_all)

    # escribir
    with open(a.out, "w", encoding="utf-8") as f:
        for d in out_docs_all:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # reporte
    report = {
        "version": 5,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": {"v3": a.v3, "pmc_dir": a.pmc, "pmc_shards": len(pmc_files)},
        "output": a.out,
        "max_per_domain": a.max_per_domain,
        "docs_total": len(out_docs_all),
        "dups_removed": n_dup,
        "domains": dict(dom_counts),
        "sources": dict(src_counts),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(a.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[build_v5] DONE {len(out_docs_all)} docs, {n_dup} dup eliminados "
          f"en {time.time()-t0:.0f}s -> {a.out}", flush=True)
    print(f"[build_v5] reporte: {a.report}")
    print(json.dumps({"domains": report["domains"], "sources": report["sources"]}, indent=2))

if __name__ == "__main__":
    main()