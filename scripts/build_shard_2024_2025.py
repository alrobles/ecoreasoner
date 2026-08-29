#!/usr/bin/env python3
"""build_shard_2024_2025.py — dimensiona y prepara el shard nuevo de full-text PMC
para 2024-2025 (y 2026 parcial) NO cubiertos por el corpus v5.

Salidas (solo dimensiona + escribe listas; NO descarga):
  - _new_pmids_2024_2025.txt  : PMIDs nuevos con PMCID, 2024-2026
  - _cand_pmc_2024_2025.pkl   : dict pmid->pmcid para esos
"""
import pandas as pd, json, pickle, time, glob, sys
sys.stdout.reconfigure(line_buffering=True)

t0 = time.time()
# 1) PMCIDs ya en v5 = los de pmc_corpus_v4 (fuente pmc-v4 del corpus v5)
have = set()
for sh in sorted(glob.glob('/beegfs/a474r867/ecoreasoner/data/pmc_corpus_v4/shard_*.jsonl')):
    with open(sh, encoding='utf-8') as f:
        for ln in f:
            try:
                o = json.loads(ln)
                c = o.get('pmcid') or ''
                if isinstance(c, str) and c.startswith('PMC'):
                    have.add(c.upper())
            except Exception:
                pass
print(f"PMCIDs ya en v5 (pmc_corpus_v4): {len(have)}  [{round(time.time()-t0,1)}s]", flush=True)

# 2) PMC-ids (con header): PMCID col index 8, PMID col 9, Year col 3
t1 = time.time()
pmc = pd.read_csv('/beegfs/a474r867/litdump/pubmed/PMC-ids.csv.gz', compression='gzip',
                  dtype=str, keep_default_na=False)
pmc['Year'] = pd.to_numeric(pmc['Year'], errors='coerce')
print(f"PMC-ids: {len(pmc)} filas  [{round(time.time()-t1,1)}s]", flush=True)

# 3) Filtrar 2025-2026 (los 2 anios mas recientes con margen: v5 ya cubre 2024-2025 bien),
# PMID no vacio, PMCID valido, y NO en v5
sel = pmc[(pmc['Year'] >= 2025) & (pmc['Year'] <= 2026) &
          (pmc['PMID'].str.strip() != '') & (pmc['PMCID'].str.startswith('PMC'))].copy()
sel['PMCID'] = sel['PMCID'].str.upper()
sel = sel[~sel['PMCID'].isin(have)]
print(f"PMCs 2025-2026 en PMC-ids: {len(pmc[(pmc['Year']>=2025)&(pmc['Year']<=2026)&(pmc['PMID'].str.strip()!='')&(pmc['PMCID'].str.startswith('PMC'))])}")
print(f"  -> nuevos vs v5: {len(sel)}  [{round(time.time()-t1,1)}s]", flush=True)

# distribucion por ano del margen
print("por ano:")
print(sel.groupby('Year').size().to_string())

# 4) escribir listas
pmids = sel['PMID'].str.strip().drop_duplicates().tolist()
pmid2pmcid = dict(zip(sel['PMID'].str.strip(), sel['PMCID']))
with open('/beegfs/a474r867/ecoreasoner/data/_new_pmids_2024_2025.txt', 'w') as f:
    f.write('\n'.join(pmids))
with open('/beegfs/a474r867/ecoreasoner/data/_cand_pmc_2024_2025.pkl', 'wb') as f:
    pickle.dump(pmid2pmcid, f)
print(f"escritos: {len(pmids)} pmids, {len(pmid2pmcid)} en pkl", flush=True)
print("DONE")
