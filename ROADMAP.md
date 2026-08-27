# ROADMAP — Ingesta masiva PMC OA → Parquet → DuckDB → RAG

> ReumanLab · EcoReasoner · 2026-08-26
> Objetivo: bajar el **Open Access Subset de PMC**, filtrar por tiempo/dominio,
> almacenar en **Parquet (columnar, consultas rápidas)**, y servir como **RAG** del agente
> científico. Decisión de diseño: usar el enfoque **heurístico C** (calibrar PMCID↔año)
> para el filtro "últimos 10 años" con volumen medible ANTES de descargar en bulto.

---

## 0. Decisión técnica clave (algoritmo C — PMCID↔año)

**Por qué:** los PMCIDs no son contiguos y el bucket se ordena por clave `PMCID.ver/`. PMC
se publica cronológicamente → PMCID alto ≈ artículo reciente (monotónico). Para filtrar
"últimos 10 años" sin bajar ~8M objetos completos, calibramos PMCID→año usando el **S3
Inventory oficial** (CSV diario de todos los `metadata/*.json`):

1. Bajar el inventory (`s3://pmc-oa-opendata/inventory-reports/.../metadata/`).
2. Los `metadata/PMCID.ver.json` traen **citation (año), license_code, title, doi,
   is_pmc_openaccess, is_manuscript, is_retracted** — el filtro se hace sobre esto, no el texto.
3. Ajustar la curva PMCID→año (o año directo desde citation) y estimar volumen/tamaño de
   los últimos 10 años (≥2016) antes de descargar texto.
4. Enriquecer con el token Entrez de NCBI si hace falta cruzar PMID→año.

`scripts/pmc/calibrar_pmcid_anio.py` implementa el muestreo base.
**Dato oficial (pmcaws, 2026):** ~**8M PMC article versions**; el JSON tiene `license_code`
(CC BY / CC BY-NC / `TDM` para author manuscripts con full-text reusable).

## 1. Tecnologías (investigadas y confirmadas)

| Capa | Tecnología | Por qué | Estado verificado |
|---|---|---|---|
| Fuente OA | PMC S3 `pmc-oa-opendata` (world-readable, sin auth) | 1.5–5M artículos OA (CC/reuso) | ✅ listable (ListBucketResult), .json+ .txt por PMCID |
| Fuente metadata | S3 inventory reports (oficial, `inventory-reports/`) | filtra por fecha/licencia/PMCID | ✅ documentado (pmcaws) |
| Almacen. columnar | **Parquet** (pyarrow 21/24) | columnar, comprimido, footer metadata rápido | ✅ kuhpc pyarrow 21, local 24 |
| Consultas | **DuckDB** (1.4.4) | SQL sobre Parquet, vectorizado, zero-copy con Arrow | ✅ kuhpc duckdb 1.4.4 |
| Streaming (opcional) | Apache NiFi/MiNiFi + Kafka | para ingesta masiva en tiempo real (exactamente tu idea) | referenciado (ADR/Arrow, ~1M filas/s) |
| RAG futuro | DuckDB/Parquet → vector (retrieval) | papers completos consultables por el agente | diseño |

**Por qué DuckDB+Arrow (no solo Spark):** hay integración Arrow zero-copy, query directa sobre
Parquet en S3, merge de millones de filas/s en un solo nodo; suficiente para nuestro
volumen (GB a TB). NiFi/Kafka solo si necesitamos streaming en producción (batch basta).

## 2. Arquitectura propuesta

```
PMC S3 (pmc-oa-opendata)
   │  [ingestor: descarga .txt por PMCID ± .json fecha/licencia]
   ▼
staging/ raw jsonl (pmcid, ver, text, year, license)
   │  [procesador: filtra año≥corte, licencia CC, dedup, tokeniza]
   ▼
Parquet columnar  (particionado por año; filtro 10 años / dominio)
   │                 (pyarrow / DuckDB COPY)
   ├──► dLLM corpus (entrenamiento)        ← mina actual
   └──► RAG index (futuro agente)  ← DuckDB query / retrieval

Pipeline "industrial" : S3 → ingest shards (slurm multi-proc) → staging → parquet → duckdb.
Opcional NiFi/Kafka si se quiere streaming continuo (no requerido para batch).
```

## 3. Milestones (medibles)

### M1 — Calibración PMCID↔año (estimación de volumen) — ✅ COMPLETADO
- [x] `calibrar_pmcid_anio.py` / `m1_inventory_calibrar.py` corre en kuhpc con S3 Inventory real.
- [x] S3 Inventory descargado: **~3M metadata/artículos** en el PMC OA subset.
- [x] Muestra aleatoria uniforme de 400 metadata → **98.8% ≥2016** (últimos 10 años ≈ todo el subset).
- [x] Rango de años cubierto y curva PMCID→año ajustada (rango 1000-2026 en muestras).
- [x] Licencias: CC BY (mayoría), CC BY-NC/ND, TDM, CC0 → filtrable.
- **Criterio cumplido (medible):** ~3.0M artículos ≥2016, estimación **~150-170 GB de texto** (a ~54k chars/doc) → `scripts/pmc/m1_resultado.json`.

### M2 — Ingestor S3→Parquet (shard demo) — ✅ COMPLETADO
- [x] `m2_ingest_demo.py` descarga metadata+texto y escribe Parquet.
- [x] Shard demo **200 PMCIDs** → `pmc_demo.parquet` en **125s**, validado `DuckDB COUNT(*)=200`, columnas (year, license, text) presentes.
- [x] Licencias del demo: CC BY (186), CC0 (8), TDM (5), None (1) → CC BY dominante.
- **Criterio cumplido:** Parquet consultable por DuckDB; **~1.6s/artículo** (grosso = descarga texto) ⇒ 3M artículos ≈ **~58 días en 1 hilo** → NO viable secuencial, exige paralelizar por shards (slurm).

Nota de alcance: se espera ingesta de **últimos 5 años primero** (más barato; ~99% del subset es ≥2021) y luego ampliar.

### F3 — Ingesta completa 2024-2026 (primer chunk real) — EN PROGRESO
- [x] Probe validado: rango num 11M-13M = **1,802,253 PMCIDs** del inventory (proxy 2024-2026).
- [x] Probe 500 PMCIDs → **469 filas válidas 2024** (94% densidad), ~0.6s/artículo, Parquet OK DuckDB COUNT=469.
- [x] Worker con flush incremental cada 5000 (tolerante a cortes).
- [ ] Slurm array **~55 shards** (~33K artículos/cada ≈ ~5.5h) → **~6h turnaround** para 1.8M artículos (~160MB-1GB por shard ≈ ~50-90GB total).
- [ ] Filtrar por año≥2024 y licencia (CC BY/CC0/TDM/NC); dominio ecológico opcional post-hoc.
- **Criterio:** ~1.5-1.8M papers completos en Parquet en beegfs, validable con DuckDB COUNT.
Nota de alcance: si se alcanza buen techo en pocos días, se detiene (el usuario lo autorizó).

### M4 — DuckDB lake / query service — [ ]
- [ ] Catálogo Parquet registrado en DuckDB (tabla virtual `PMC` con año, texto, licencia).
- **Criterio:** query de ejemplo `SELECT * FROM PMC WHERE year>=2016 AND text ILIKE '%species%' LIMIT 10` responde <1s.

### F5 — RAG prototipo (opcional, futuro) — [ ]
- [ ] Extensión: embeddings sobre Parquet → retrieval (DuckDB FTS o vector).
- **Criterio:** consulta de paper completo relevante al concepto retorna el doc >100 tokens.

### F6 — Integración dLLM — [ ]
- [ ] El corpus Parquet (10 años, ecológico opcional) se convierte a JSONL para el training dLLM v4.
- **Criterio:** v4 incorpora los papers completos nuevos; medida de tokens/volumen superior al v3 (97k PMC).

## 4. Dependencias / recursos

- kuhpc: pyarrow 21, duckdb 1.4.4 ✓. S3 acceso HTTP (sin auth) ✓.
- Almacenamiento: /beegfs (~860TB libre) — cabe (decenas–cientos GB de text real).
- Inodes: usar archivos Parquet grandes (no 1m mini) para no quemar inodes.

## 5. Riesgos / mitigaciones

| Riesgo | Mitigación |
|---|---|
| PMCID↔año del modelo: no es exacto | algoritmo heurístico + reportar estimación con margen; el corte real se valida con muestra |
| Volumen subestimado | filtrar por fecha con S3 inventory (oficial) antes que bulk |
| licencias | respetar (CC-BY/CC0/commercial vs NC), filtro por `license` |
| DuckDB/ram en nodos | parquet por shard, no abrir todo en memoria; consulta columnar parcial |
| Coste ITRM | gratuito (S3 world-readable); solo traffic del cluster |

---

_Redactado como parte de la documentación del montaje. Siguiente paso: ejecutar M1 (calibración) en el cluster._