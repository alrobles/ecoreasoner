# ROADMAP — Massive PMC OA ingest → Parquet → DuckDB → RAG

> ReumanLab · EcoReasoner · 2026-08-26
> Goal: download the **PMC Open Access Subset**, filter by time/domain,
> store in **Parquet (columnar, fast queries)**, and serve as the scientific agent's
> **RAG**. Design decision: use the **heuristic C approach** (calibrate PMCID↔year)
> for the "last 10 years" filter with measurable volume BEFORE bulk download.

---

## 0. Key technical decision (algorithm C — PMCID↔year)

**Why:** PMCIDs are not contiguous and the bucket is keyed by `PMCID.ver/`. PMC
is published chronologically → high PMCID ≈ recent article (monotonic). To filter
"last 10 years" without downloading ~8M full objects, we calibrate PMCID→year using
the official **S3 Inventory** (daily CSV of all `metadata/*.json`):

1. Download the inventory (`s3://pmc-oa-opendata/inventory-reports/.../metadata/`).
2. The `metadata/PMCID.ver.json` files carry **citation (year), license_code, title, doi,
   is_pmc_openaccess, is_manuscript, is_retracted** — filtering happens here, not on the text.
3. Fit the PMCID→year curve (or use year directly from citation) and estimate the volume/size of
   the last 10 years (≥2016) before downloading text.
4. Enrich with the NCBI Entrez token if a PMID→year crosswalk is needed.

`scripts/pmc/calibrar_pmcid_anio.py` implements the base sampling.
**Official datum (pmcaws, 2026):** ~**8M PMC article versions**; the JSON has `license_code`
(CC BY / CC BY-NC / `TDM` for author manuscripts with reusable full-text).

## 1. Technologies (researched and confirmed)

| Layer | Technology | Why | Verified status |
|---|---|---|---|
| OA source | PMC S3 `pmc-oa-opendata` (world-readable, no auth) | 1.5–5M OA articles (CC/reuse) | ✅ listable (ListBucketResult), .json+ .txt per PMCID |
| Metadata source | S3 inventory reports (official, `inventory-reports/`) | filter by date/license/PMCID | ✅ documented (pmcaws) |
| Columnar storage | **Parquet** (pyarrow 21/24) | columnar, compressed, fast footer metadata | ✅ kuhpc pyarrow 21, local 24 |
| Queries | **DuckDB** (1.4.4) | SQL over Parquet, vectorized, zero-copy Arrow | ✅ kuhpc duckdb 1.4.4 |
| Streaming (optional) | Apache NiFi/MiNiFi + Kafka | for massive real-time ingest (exactly your idea) | referenced (ADR/Arrow, ~1M rows/s) |
| Future RAG | DuckDB/Parquet → vector (retrieval) | full papers queryable by the agent | design |

**Why DuckDB+Arrow (not Spark alone):** zero-copy Arrow integration, direct
Parquet-in-S3 queries, millions of rows/s merged on a single node; sufficient for our
volume (GB to TB). NiFi/Kafka only if we need production streaming (batch is enough).

## 2. Proposed architecture

```
PMC S3 (pmc-oa-opendata)
   │  [ingestor: downloads .txt per PMCID ± .json date/license]
   ▼
staging/ raw jsonl (pmcid, ver, text, year, license)
   │  [processor: filters year≥cutoff, CC license, dedup, tokenize]
   ▼
Parquet columnar  (partitioned by year; 10-year/domain filter)
   │                 (pyarrow / DuckDB COPY)
   ├──► dLLM corpus (training)            ← current mine
   └──► RAG index (future agent)  ← DuckDB query / retrieval

"Industrial" pipeline: S3 → ingest shards (slurm multi-proc) → staging → parquet → duckdb.
Optional NiFi/Kafka for continuous streaming (not required for batch).
```

## 3. Milestones (measurable)

### M1 — PMCID↔year calibration (volume estimation) — ✅ COMPLETED
- [x] `calibrar_pmcid_anio.py` / `m1_inventory_calibrar.py` runs on kuhpc with real S3 Inventory.
- [x] S3 Inventory downloaded: **~3M metadata/articles** in the PMC OA subset.
- [x] Uniform random sample of 400 metadata → **98.8% ≥2016** (last 10 years ≈ entire subset).
- [x] Year range covered and PMCID→year curve fitted (range 1000-2026 in samples).
- [x] Licenses: CC BY (majority), CC BY-NC/ND, TDM, CC0 → filterable.
- **Criterion met (measurable):** ~3.0M articles ≥2016, estimated **~150-170 GB of text** (at ~54k chars/doc) → `scripts/pmc/m1_resultado.json`.

### M2 — S3→Parquet ingestor (demo shard) — ✅ COMPLETED
- [x] `m2_ingest_demo.py` downloads metadata+text and writes Parquet.
- [x] Demo shard **200 PMCIDs** → `pmc_demo.parquet` in **125s**, validated `DuckDB COUNT(*)=200`, columns (year, license, text) present.
- [x] Demo licenses: CC BY (186), CC0 (8), TDM (5), None (1) → CC BY dominant.
- **Criterion met:** DuckDB-queryable Parquet; **~1.6s/article** (gross ≈ text download) ⇒ 3M articles ≈ **~58 days on 1 thread** → NOT viable sequential, requires shard parallelization (slurm).

Scope note: ingest of the **last 5 years first** is expected (cheaper; ~99% of the subset is ≥2021), then expand.

### F3 — Full 2024-2026 ingest (first real chunk) — IN PROGRESS
- [x] Validated probe: numeric range 11M-13M = **1,802,253 PMCIDs** from inventory (proxy 2024-2026).
- [x] 500-PMCID probe → **469 valid 2024 rows** (94% density), ~0.6s/article, Parquet OK DuckDB COUNT=469.
- [x] Worker with incremental flush every 5000 (crash-tolerant).
- [ ] Slurm array **~55 shards** (~33K articles each ≈ ~5.5h) → **~6h turnaround** for 1.8M articles (~160MB-1GB per shard ≈ ~50-90GB total).
- [ ] Filter by year≥2024 and license (CC BY/CC0/TDM/NC); optional ecological domain post-hoc.
- **Criterion:** ~1.5-1.8M full papers in Parquet on beegfs, verifiable with DuckDB COUNT.
Scope note: if a good ceiling is reached in a few days, stop (user authorized).

### M4 — DuckDB lake / query service — [ ]
- [ ] Parquet catalog registered in DuckDB (virtual `PMC` table with year, text, license).
- **Criterion:** example query `SELECT * FROM PMC WHERE year>=2016 AND text ILIKE '%species%' LIMIT 10` answers <1s.

### F5 — RAG prototype (optional, future) — [ ]
- [ ] Extension: embeddings over Parquet → retrieval (DuckDB FTS or vector).
- **Criterion:** a full-paper query relevant to a concept returns the doc >100 tokens.

### F6 — dLLM integration — [ ]
- [ ] The Parquet corpus (10 years, optionally ecological) becomes JSONL for the dLLM v4 training.
- **Criterion:** v4 incorporates the new full papers; token/volume measure higher than v3 (97k PMC).

## 4. Dependencies / resources

- kuhpc: pyarrow 21, duckdb 1.4.4 ✓. HTTP S3 access (no auth) ✓.
- Storage: /beegfs (~860TB free) — fits (tens–hundreds of GB of real text).
- Inodes: use large Parquet files (not 1M mini) to avoid burning inodes.

## 5. Risks / mitigations

| Risk | Mitigation |
|---|---|
| PMCID↔year model: not exact | heuristic algorithm + report estimate with margin; the real cutoff is validated with a sample |
| Underestimated volume | filter by date with S3 inventory (official) before bulk |
| licenses | respect them (CC-BY/CC0/commercial vs NC), filter by `license` |
| DuckDB/ram on nodes | parquet per shard, don't open everything in memory; partial columnar query |
| ITRM cost | free (world-readable S3); only cluster traffic |

---

_Written as part of the build documentation. Next step: run M1 (calibration) on the cluster._