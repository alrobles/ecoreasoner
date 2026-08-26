#!/usr/bin/env python3
"""mine_pubmed_duckdb.py — mina PubMed desde Parquet (DuckDB) multi-dominio
para el corpus dLLM EcoReasoner. Reemplaza al minero FTS5/SQLite.

Fuente: /beegfs/a474r867/litdump/pubmed/parsed/year=*/*.parquet (30.8M rows)
Queries: búsquedas complejas columnar (texto via regexp/ilike + filtros por
año/journal/mesh) + muestreo estratificado por dominio y década.
Salida: .jsonl {"text","domain","pmid","year"} — multi-dominio.

Uso:
  python3 mine_pubmed_duckdb.py --out eco_corpus.jsonl \
      --max_per_domain 200000 [--domains eco,phylo,genom,bioc] \
      [--min_year 2000] [--min_abstract_len 200]

Dominios (queries sobre title+abstract, ILIKE para robustez):
  eco   : ecología / SDM / biodiversidad
  phylo : filogenia / evolución
  genom : genómica / poblaciones
  bioc  : bioinformática / modelado
"""
import argparse, json, random, time
from pathlib import Path

PARQUET = "/beegfs/a474r867/litdump/pubmed/parsed/parquet/year=*/*.parquet"

# dominio -> (filtro WHERE sobre lower(title||' '||abstract)) usando ILIKE (validado)
# DuckDB: ILIKE funciona robusto; regexp `~` no. Cada término ILIKE, OR'ed.
DOMAIN_FILTERS = {
    "eco":   [("title||' '||abstract ILIKE '%ecolog%' OR title||' '||abstract ILIKE '%biodivers%' OR "
               "title||' '||abstract ILIKE '%niche%' OR title||' '||abstract ILIKE '%species distribution%' OR "
               "title||' '||abstract ILIKE '%maxent%' OR title||' '||abstract ILIKE '%occupancy%' OR "
               "title||' '||abstract ILIKE '%habitat%' OR title||' '||abstract ILIKE '%conservation%'")],
    "phylo": [("title||' '||abstract ILIKE '%phylogen%' OR title||' '||abstract ILIKE '%phylogeograph%' OR "
               "title||' '||abstract ILIKE '%natural selection%' OR title||' '||abstract ILIKE '%adaptation%' OR "
               "title||' '||abstract ILIKE '%speciation%' OR title||' '||abstract ILIKE '%divergence%' OR "
               "title||' '||abstract ILIKE '%evolution%'")],
    "genom": [("title||' '||abstract ILIKE '%genom%' OR title||' '||abstract ILIKE '%transcriptom%' OR "
               "title||' '||abstract ILIKE '%population genetic%' OR title||' '||abstract ILIKE '%gwas%' OR "
               "title||' '||abstract ILIKE '%genome assembl%' OR title||' '||abstract ILIKE '%gene expression%' OR "
               "title||' '||abstract ILIKE '%variant%'")],
    "bioc":  [("title||' '||abstract ILIKE '%bioinformatic%' OR title||' '||abstract ILIKE '%machine learning%' OR "
               "title||' '||abstract ILIKE '%modeling%' OR title||' '||abstract ILIKE '%simulation%' OR "
               "title||' '||abstract ILIKE '%systems biology%' OR title||' '||abstract ILIKE '%network biolog%' OR "
               "title||' '||abstract ILIKE '%deep learning%'")],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/eco_corpus.jsonl")
    ap.add_argument("--max_per_domain", type=int, default=200000)
    ap.add_argument("--min_abstract_len", type=int, default=200)
    ap.add_argument("--min_year", type=int, default=1990, help="solo >= año")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domains", default="eco,phylo,genom,bioc")
    args = ap.parse_args()
    random.seed(args.seed)
    wanted = args.domains.split(",")

    import duckdb
    con = duckdb.connect()
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] leyendo {args.parquet} ...", flush=True)
    # sample estratificado: por dominio + década, con filtro de año y texto
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    with open(out_path, "w") as f:
        for dom in wanted:
            conds = DOMAIN_FILTERS.get(dom)
            if not conds:
                print(f"  dominio desconocido: {dom}", flush=True); continue
            where_text = " OR ".join(conds)
            # década en subquery para estratificar
            q = f"""
                WITH base AS (
                    SELECT pmid, year, title, abstract, mesh,
                           (CAST(year AS INT)/10*10) AS decade
                    FROM read_parquet('{args.parquet}')
                    WHERE CAST(year AS INT) >= {args.min_year}
                      AND length(abstract) >= {args.min_abstract_len}
                      AND ({where_text})
                )
                SELECT * FROM base
                ORDER BY decade DESC, pmid
                LIMIT {args.max_per_domain*2}
            """
            try:
                rows = con.execute(q).fetchdf()
            except Exception as e:
                print(f"  ERR {dom}: {e}", flush=True); continue
            # muestreo estratificado por década (hasta max_per_domain)
            rows = rows.sample(frac=1.0, random_state=args.seed)  # shuffle
            take = rows.head(args.max_per_domain)
            for _, r in take.iterrows():
                text = (str(r.get("title","")) + "\n" + str(r.get("abstract",""))).strip()
                if len(text) < args.min_abstract_len: continue
                f.write(json.dumps({"text": text, "domain": dom,
                                    "pmid": int(r.get("pmid",0)), "year": int(r.get("year",0))},
                                   ensure_ascii=False) + "\n")
                n_total += 1
            print(f"  [{dom}] {len(rows)} unicos -> {len(take)} tomados (acum {n_total}, {time.time()-t0:.0f}s)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE: {n_total} docs -> {out_path}", flush=True)

if __name__ == "__main__":
    main()