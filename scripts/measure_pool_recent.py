#!/usr/bin/env python3
"""measure_pool_recent.py — mide candidatos eco por ano RECIENTE (2020-2025)
y cuantos ya cubre el corpus v5 (por pmcid) para dimensionar un shard nuevo
de full-text PMC sin duplicar. Corre como Slurm CPU."""
import duckdb, time, json, sys
sys.stdout.reconfigure(line_buffering=True)
con = duckdb.connect()
P = '/beegfs/a474r867/litdump/pubmed/parsed/parquet/year=*/*.parquet'
CONDS_ECO = """title||' '||abstract ILIKE '%ecolog%' OR title||' '||abstract ILIKE '%biodivers%' OR
title||' '||abstract ILIKE '%niche%' OR title||' '||abstract ILIKE '%species distribution%' OR
title||' '||abstract ILIKE '%maxent%' OR title||' '||abstract ILIKE '%occupancy%' OR
title||' '||abstract ILIKE '%habitat%' OR title||' '||abstract ILIKE '%conservation%'"""

t0 = time.time()
# Candidatos eco post-2019 con abstract>=200, por ano
df = con.execute(f"""
  SELECT CAST(year AS INT) AS yr,
         COUNT(*) AS eco_cand,
         COUNT(DISTINCT CAST(pmid AS VARCHAR)) AS eco_pmids
  FROM read_parquet('{P}')
  WHERE CAST(year AS INT) BETWEEN 2020 AND 2025
    AND length(abstract)>=200 AND ({CONDS_ECO})
  GROUP BY 1 ORDER BY 1
""").df()
print("== candidatos eco por ano (parquet) ==")
print(df.to_string())
print("scan s:", round(time.time()-t0,1), flush=True)

# Cuantos de esos tienen PMCID (full-text PMC disponible)
import pandas as pd
t1 = time.time()
# PMC-ids.csv.gz TIENE HEADER (12 cols): Journal,ISSN,eISSN,Year,Vol,Issue,Page,
# DOI,PMCID,PMID,Manuscript,ReleaseDate. PMCID=col index 8, PMID=col index 9.
pmc = pd.read_csv('/beegfs/a474r867/litdump/pubmed/PMC-ids.csv.gz', compression='gzip',
                  dtype=str, keep_default_na=False)
print("PMC-ids leido s:", round(time.time()-t1,1), "filas:", len(pmc),
      "cols:", list(pmc.columns), flush=True)
pmc_set = set(pmc['PMID'].str.strip())
print("pmc_set size:", len(pmc_set), "ejemplos:", list(pmc_set)[:3], flush=True)
t0 = time.time()
pids_by_year = con.execute(f"""
  SELECT CAST(year AS INT) AS yr, CAST(pmid AS VARCHAR) AS pmid
  FROM read_parquet('{P}')
  WHERE CAST(year AS INT) BETWEEN 2020 AND 2025
    AND length(abstract)>=200 AND ({CONDS_ECO})
""").df()
print("extract pids s:", round(time.time()-t0,1), flush=True)
pids_by_year['has_pmc'] = pids_by_year['pmid'].isin(pmc_set)
g = pids_by_year.groupby('yr').agg(
    eco_pmids=('pmid','nunique'),
    with_pmc=('pmid', lambda s: s[s.isin(pmc_set)].nunique())).reset_index()
print("== eco_pmids y con PMC por ano ==")
print(g.to_string())
print("TOTAL eco 2020-2025:", int(pids_by_year['pmid'].nunique()),
      "con PMC:", int(pids_by_year[pids_by_year['has_pmc']]['pmid'].nunique()), flush=True)

# PMCs que ya estan en el corpus v5 (por pmcid) -> margen real
v5_pmc = set()
with open('/beegfs/a474r867/ecoreasoner/data/train_corpus_v5.jsonl', encoding='utf-8') as f:
    for i, ln in enumerate(f):
        if i >= 2000000: break
        try:
            o = json.loads(ln)
            pmcid = o.get('pmcid') or o.get('pmid','')
            if isinstance(pmcid, str) and pmcid.startswith('PMC'):
                v5_pmc.add(pmcid.upper())
        except Exception:
            pass
print("PMCs distintos en v5 (muestra 2M docs):", len(v5_pmc), flush=True)
print("DONE")
