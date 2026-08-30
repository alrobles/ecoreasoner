#!/usr/bin/env python3
"""
build_arxiv_corpus.py — Une los shards de full-text arXiv descargados
(data/arxiv/fulltext/fulltext_c*.jsonl) en train_corpus_phys.jsonl con el
formato del pipeline (text/pmid/domain/year/source), dedup por arxiv_id y
filtro de longitud.

Compatible aguas abajo con pre_tokenize.py (lee text) y con el merge v6/v7
(dedup por pmcid/pmid).

Ejecutar SOLO via Slurm.
"""
from __future__ import annotations
import argparse, glob, json, os, random, sys, time
from collections import Counter

if not os.environ.get("SLURM_JOB_ID"):
    sys.exit("ERROR: build_arxiv_corpus.py SOLO via Slurm.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/beegfs/a474r867/ecoreasoner/data/arxiv/fulltext")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/train_corpus_phys.jsonl")
    ap.add_argument("--report", default="/beegfs/a474r867/ecoreasoner/data/arxiv_phys_report.json")
    ap.add_argument("--min-text", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t0 = time.time()
    files = sorted(glob.glob(os.path.join(a.src, "fulltext_c*.jsonl")))
    if not files:
        sys.exit(f"ERROR: no hay shards en {a.src}")
    print(f"[arxiv-corpus] {len(files)} shards, leyendo...", flush=True)

    seen = set(); out = []
    n_short = n_dup = n_bad = 0
    doms = Counter(); yrs = Counter(); srcs = Counter(); lics = Counter()
    for fp in files:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                try: d = json.loads(ln)
                except Exception:
                    n_bad += 1; continue
                txt = d.get("text","")
                if len(txt) < a.min_text:
                    n_short += 1; continue
                aid = d.get("arxiv_id","")
                if not aid or aid in seen:
                    n_dup += 1; continue
                seen.add(aid)
                # pmid = arxiv_id (compatible con el dedup del pipeline v6)
                doc = {
                    "text": txt,
                    "pmid": aid,
                    "arxiv_id": aid,
                    "title": d.get("title",""),
                    "abstract": d.get("abstract",""),
                    "license": d.get("license",""),
                    "year": d.get("year"),
                    "domain": d.get("domain",""),
                    "source": "arxiv",
                }
                out.append(doc)
                doms[d.get("domain","")] += 1
                yrs[str(d.get("year"))] += 1
                srcs["arxiv"] += 1
                lics[d.get("license","")] += 1
    print(f"[arxiv-corpus] leidos, emitidos {len(out)} (dup {n_dup}, short {n_short}, bad {n_bad}) "
          f"[{time.time()-t0:.0f}s]", flush=True)

    random.seed(a.seed); random.shuffle(out)
    with open(a.out, "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    toks = sum(len(d["text"])//4 for d in out)
    report = {
        "version": "phys-1",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output": a.out,
        "docs_total": len(out),
        "approx_tokens": toks,
        "dups_removed": n_dup,
        "short_removed": n_short,
        "domains": dict(doms),
        "years": dict(yrs),
        "licenses": dict(lics),
        "elapsed_s": round(time.time()-t0,1),
    }
    with open(a.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[arxiv-corpus] DONE {len(out)} docs, ~{toks/1e6:.0f}M tok -> {a.out}", flush=True)
    print("domains:", dict(doms), flush=True)
    print("years:", dict(sorted(yrs.items())), flush=True)
    print("licenses:", dict(lics), flush=True)

if __name__ == "__main__":
    main()