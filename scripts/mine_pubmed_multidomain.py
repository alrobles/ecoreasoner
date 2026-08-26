#!/usr/bin/env python3
"""mine_pubmed_multidomain.py — mina PubMed (SQLite FTS5) en dominios múltiples
para el corpus dLLM EcoReasoner. Salida: .jsonl con {text} por doc (mask+denoise).

Fuente: /beegfs/a474r867/litdump/pubmed/pubmed_full.db  (FTS5 index articles_fts)
Formato de salida: líneas {"text": "...", "domain": "...", "pmid": ..., "year": ...}

Uso:
  python3 mine_pubmed_multidomain.py --db PATH --out eco_corpus.jsonl \
      --max_per_domain 200000 [--domains eco,phylo,genom] [--min_abstract_len 200]

Estratificada multi-dominio: ecología, biología evolutiva, genómica, bio computacional.
Cada dominio tiene un set de queries FTS5 (boolean OR) y una fracción de muestreo.
"""
import argparse, json, random, sqlite3, sys, time
from pathlib import Path

# Dominios [id, label, queries FTS5]
DOMAINS = [
    {"id":"eco",  "label":"ecology_sdm", "queries":[
        "ecolog* OR biodivers* OR niche OR \"species distribution\" OR MaxEnt OR occupancy",
        "species OR population OR habitat OR \"community ecology\" OR conservation OR invasion",
        "\"climate change\" species OR \"remote sensing\" vegetation OR \"land use\" biodiversity",
    ]},
    {"id":"phylo", "label":"evolution_phylo", "queries":[
        "phylogen* OR evolut* OR \"natural selection\" OR adaptation OR speciation OR divergence",
        "phylogeograph* OR \"molecular evolution\" OR \"comparative method\" OR biogeograph*",
    ]},
    {"id":"genom", "label":"genomics", "queries":[
        "genom* OR transcriptom* OR \"population genetics\" OR GWAS OR \"genome assembly\"",
        "\"gene expression\" OR metabolom* OR proteom* OR \"structural variant\"",
    ]},
    {"id":"bioc", "label":"bio_computational", "queries":[
        "bioinformatic* OR \"machine learning\" biolog* OR modeling biolog* OR simulation biolog*",
        "\"systems biology\" OR \"network biology\" OR \"deep learning\" biolog* OR algorithm",
    ]},
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/beegfs/a474r867/litdump/pubmed/pubmed_full.db")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/eco_corpus.jsonl")
    ap.add_argument("--max_per_domain", type=int, default=200000)
    ap.add_argument("--min_abstract_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domains", default="eco,phylo,genom,bioc")
    args = ap.parse_args()

    random.seed(args.seed)
    wanted = [d for d in DOMAINS if d["id"] in args.domains.split(",")]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] conectando {args.db} ...", flush=True)
    c = sqlite3.connect(args.db)
    # FTS5 table is `articles_fts` (with content table articles)
    # Use MATCH with content-table join: SELECT ... FROM articles_fts f JOIN articles a USING(pmid)
    n_total = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for d in wanted:
            label = d["label"]
            # collect pmids for all queries via ORDER BY rank LIMIT (relevancia, eficiente)
            pmids = []
            for q in d["queries"]:
                try:
                    rows = c.execute(
                        "SELECT pmid FROM articles_fts WHERE articles_fts MATCH ? "
                        "ORDER BY rank LIMIT ?",
                        (q, args.max_per_domain*2)
                    ).fetchall()
                    for r in rows:
                        if r[0] not in pmids:
                            pmids.append(r[0])
                except Exception as e:
                    print(f"  query ERR {q!r}: {e}", flush=True)
            random.shuffle(pmids)
            take = pmids[: args.max_per_domain]
            print(f"[{label}] {len(pmids)} unicos -> tomando {len(take)}", flush=True)
            # fetch rows in chunks (avoid huge IN)
            chunk = 500
            for i in range(0, len(take), chunk):
                ids = take[i:i+chunk]
                ph = ",".join("?"*len(ids))
                rows = c.execute(
                    f"SELECT pmid, year, title, abstract FROM articles WHERE pmid IN ({ph})", ids
                ).fetchall()
                for pmid, year, title, abstract in rows:
                    abstract = (abstract or "").strip()
                    title = (title or "").strip()
                    if len(abstract) < args.min_abstract_len:
                        continue
                    text = (title + "\n" + abstract).strip()
                    f.write(json.dumps({"text": text, "domain": label, "pmid": pmid,
                                        "year": year}, ensure_ascii=False) + "\n")
                    n_total += 1
            # progress
            print(f"  [{label}] total acumulado {n_total} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE: {n_total} docs -> {out_path}", flush=True)

if __name__ == "__main__":
    main()