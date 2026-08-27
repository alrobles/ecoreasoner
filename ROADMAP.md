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

### M1 — Calibración PMCID↔año (estimación de volumen) — [objetivo]
- [ ] `calibrar_pmcid_anio.py` corre en kuhpc, **≥300 muestras** (PMCID→año real).
- [ ] **✓** Rango de años cubierto y curva PMCID→año ajustada.
- [ ] Estima el **número de PMCIDs** de los últimos 10 años (≥2016) y su **tamaño** (GB).
- **Criterio (medible):** reportar estimación con margen (ej. "~N artículos, ~X GB, ±20%").

### M2 — Ingestor S3→Parquet (primero un shard demo) — [VALIDAR]
- [ ] Script `ingest_pmc_to_parquet.py` descarga PMCID+texto y escribe Parquet.
- [ ] Ejecutar **1 shard demo** (ej. 200 PMCIDs) → Parquet local/beegfs, validar con DuckDB.
- **Criterio:** contar filas en el Parquet con DuckDB `SELECT count(*)` OK, y columnas (year, license, text) presentes.

### F3 — Ingesta completa últimos 10 años (filtro dominio) — [ ]
- [ ] Slurm array (shards paralelos) en kucpc baja los PMCIDs del corte PMCID→10 años.
- [ ] Filtrar por año y licencia (CC-BY/CC0/NC) y por dominio ecológico/biología (conceptos malla) si se quiere.
- **Criterio:** **~decenas de miles - cientos de miles** papers completos en Parquet, tamaño conocido, en beegfs.

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