#!/usr/bin/env python3
"""measure_pool.py — mide el pool minable de PubMed para EcoReasoner.
Cuenta candidatos eco-relevantes por año y cuántos tienen PMCID (→ full-text PMC disponible).
Salida consola (no escribe corpus). Corre como Slurm CPU para no saturar login.
"""
import duckdb, time
con = duckdb.connect()
P = '/beegfs/a474r867/litdump/pubmed/parsed/parquet/year=*/*.parquet'
CONDS_ECO = """title||' '||abstract ILIKE '%ecolog%' OR title||' '||abstract ILIKE '%biodivers%' OR
title||' '||abstract ILIKE '%niche%' OR title||' '||abstract ILIKE '%species distribution%' OR
title||' '||abstract ILIKE '%maxent%' OR title||' '||abstract ILIKE '%occupancy%' OR
title||' '||abstract ILIKE '%habitat%' OR title||' '||abstract ILIKE '%conservation%'"""
t0 = time.time()
q = f"""
SELECT (CAST(year AS INT)>=2000) AS post2000,
       length(abstract)>=200 AS hasabs,
       COUNT(*) AS n
FROM read_parquet('{P}')
WHERE ({CONDS_ECO})
GROUP BY 1,2 ORDER BY 1 DESC,2 DESC
"""
print(con.execute(q).df().to_string())
print('tiempo scan:', round(time.time()-t0,1), 's', flush=True)

# cuantos de esos (post2000+hasabs) tienen PMCID -> full-text PMC disponible
t0 = time.time()
pids = con.execute(f"""
    SELECT DISTINCT CAST(pmid AS VARCHAR) AS pmid
    FROM read_parquet('{P}')
    WHERE CAST(year AS INT)>=2000 AND length(abstract)>=200 AND ({CONDS_ECO})
""").fetchnumpy()['pmid']
print('pmids eco candidatos post2000+abs>200:', len(pids), flush=True)
print('tiempo extract:', round(time.time()-t0,1), 's', flush=True)
open('/beegfs/a474r867/ecoreasoner/data/_cand_pmids.txt','w').write('\n'.join(pids))
print('lista escrita en _cand_pmids.txt', flush=True)
print('DONE', flush=True)