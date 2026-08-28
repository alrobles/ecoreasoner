# EcoReasoner — Revisión del minado de texto + diseño corpus v5

> Autor: Hermes · 2026-08-28 · Estado: diseño (sin ejecutar, requiere OK)
> Propósito: revisión del trabajo de minado hecho + plan del corpus v5,
> que reemplaza a v4 como corpus de pretrain limpio.

---

## 1. Revisión del minado de texto (todo lo hecho)

### 1.1 Genealogía de los corpus

| Versión | Cómo se construyó | Docs | Tokens | Fuentes |
|---|---|---|---|---|
| v1 | abstracts pubs | 44,045 | ~165M | PubMed FTS |
| v2 | `merge_corpus.py`: abstracts + full-text PMC por pmid (full gana, skip abstract dup) | 1,009,725 | ~1.4B | PubMed + PMC (~97K full) |
| v3 | = v2 + 1,728 EcoEvoRxiv fulltext (append, no dup por pmid) | 1,011,449 | ~1.7B | +EcoEvoRxiv |
| **v4 (actual)** | `concat_corpus_v4.py`: **v3 + 1.74M PMC fulltext nuevos SIN dedup** | **2,749,947** | **1.697B** | +1.74M PMC |

### 1.2 El pipeline de minado (scripts reales en `scripts/`)

- `port_pubmed_parquet.py`: portó SQLite PubMed (30.8M rows) → Parquet particionado por año (`/beegfs/a474r867/litdump/pubmed/parsed/parquet/`, 9.9G, 1456 parquet).
- `mine_pubmed_duckdb.py` / `mine_pubmed_multidomain.py`: minan de Parquet/SQLite por dominio con ILIKE (eco/phylo/genom/bioc), `max_per_domain`, estratificado por década.
- `pmc_fetch_array.slurm` + `fetch_pmc_shard.py`: descargan full-text PMC desde S3 OA (~25 docs/s, 85,875 nuevos + 11,137 previos = 97,012).
- `concat_corpus_v4.py`: concat v3 + PMC v4, shuffle, **sin dedup**.
- `merge_corpus.py`: el correcto (dedup por pmid, full gana), usado para v2.
- `pre_tokenize.py`: jsonl → `train_ids_*.npy` (token-only) + meta.

### 1.3 Hallazgos del v4 (verificados hoy, 2026-08-28)

1. **El v4 acumula duplicados**: `concat_corpus_v4.py` deliberadamente NO deduplica
   ("no hay dedupe por clave porque v3 ya está fusionado y los PMC son full-text nuevos").
   Verificado por muestra de 1M docs: pmids_unicos=197,847, con_dup=2,138 (~1.1%).
   Además v3 (que ya tenía skip-dups) se concatenó con 1.74M PMC, pero **no se verificó
   solapamiento v3↔PMC por pmid** (el mismo paper puede estar en ambos con pmid distinto).
2. **Los PMC v4 NO llevan `domain`**: shard_0.jsonl = `{text, pmcid, year, license}`.
   En la muestra de 1M, 96,118 docs (9.6%) tienen `domain=None` — son los PMC.
3. **Source contamina el balance**: muestra 1M = 100% source `v3` (el concat pone
   `source=v3` a TODO, y `pmc-v4` solo en los nuevos, pero la muestra cayó toda en v3).
   Los dominios: bioc 244K, genom 241K, phylo 236K, eco 182K + None 96K (ver punto 2).
4. **Tokenización v4 = 2.75M docs, 1.697B tok** (`train_ids_v4.npy.meta.json`):
   el concat casi triplicó los docs pero los tokens crecieron poco (− de 1.7B), porque
   los abstracts son cortos y el full-text PMC es largo. El `n_docs_clipped=0`.
5. **El pool de minado sigue vivo**: Parquet PubMed = 30.81M filas, 9.9G; SQLite FTS
   97GB. Hay margen para minar MÁS abstracts por dominio si hace falta.

### 1.4 Estado del training activo

- `moe-v4-bw` (27477651, Blackwell 2×pro6000) **entrena sobre v4 AHORA**: step ~2260
  de 6000, loss 6.14 (bajando). Si cambiamos el corpus a v5 ⇒ este job queda obsoleto
  (aprendió de v4) ⇒ hay que decidir si se deja terminar o se cancela y se relanza
  sobre v5. **Regla: no tocar el corpus sin checkpoint evaluable del v4.**

---

## 2. Diseño del corpus v5 (limpio, la base para el futuro)

### 2.1 Objetivo

Un corpus de pretrain **limpio y balanceado** que reemplace a v4:
- **Dedup global** por pmid (y por similitud de título para cross-source).
- **Dominio asignado a TODOS los docs** (incluidos PMC, hoy sin domain).
- **Balance de dominios** razonable (eco/phylo/genom/bioc + fulltext).
- Sin duplicados → tokenización eficiente (no 2.75M docs para 1.7B tok).

### 2.2 Recetas (opciones)

#### Opción A — "v5-limpio" (recomendada, barata, no toca modelo)
1. **Dedup global por pmid**: un solo pase con un set de pmids vistos.
   - Fuentes: v3 (1.01M) + PMC v4 (1.74M) + abstracts nuevos (si se mina más).
   - Regla de prioridad: **full-text gana** sobre abstract del mismo pmid; si el mismo
     pmid aparece en v3 y en PMC-v4, gana el full-text v4 (más rico).
2. **Asignar dominio a PMC**: los PMC v4 no tienen domain. Opciones:
   - (a) Inferir por texto con el mismo ILIKE de dominios (rápido, ~1 h CPU).
   - (b) Heredar del pmid si el abstract del mismo pmid en FTS/parquet tiene domain.
   - (a) es más simple y no requiere join.
3. **Balancear dominios**: con `--max_per_domain` (p.ej. 300K/dominio) para que
   ninguno domine. Los full-text largos pueden tener cuota menor (son caros).
4. **Repetir la lógica de `merge_corpus.py`** (full gana, skip abstract dup) pero a
   escala v5 y con los 3 pasos arriba.
5. **Tokenizar** → `train_ids_v5.npy` (~6-7G) + meta. **NO tocar v4 mientras
   moe-v4-bw entrena**; el v5 se construye en paralelo y se activa cuando el v4
   termine/checkpoint.

#### Opción B — "v5-minado+" (más ambiciosa, añade pool)
1. Minar MÁS abstracts del pool Parquet (30.81M filas, hay margen) por dominio,
   subiendo `max_per_domain` para cubrir mejor los dominios minoritarios (eco).
2. Opcional: cruzar contra `PMC-ids.csv.gz` para traer MÁS full-text PMC de los que
   ya hay (el pool S3 está explotado al ~25% de fail, hay margen).
3. Aplicar el mismo dedup/dominio/balance que la Opción A.

**Recomendación**: empezar por **A** (limpia lo que ya hay, barato, sin GPU), y si el
análisis del corpus lo pide, hacer **B** después. El v5 NO debe crecer en docs sin
controlar la duplicación — la regla es "menos docs, más limpios, mejor balanceados".

### 2.3 Criterios de éxito del v5 (medibles)

- Dedup: **0 duplicados de pmid** en el corpus final (verificable por scan).
- Dominio: **100% de docs con domain** (0 con None), balance ~25-30% por dominio
  (o la cuota que se decida).
- Tokens: `n_docs_usable ≈ n_docs`, `n_docs_clipped=0`, y la tokenización eficiente
  (sin 2.75M docs para 1.7B tok).
- Source: limpio y consistente (v3/pmc-v4/ecoEvoRxiv), sin contaminar.

### 2.4 Qué NO hacer

- **No** lanzar el v5 mientras moe-v4-bw entrena (evitar condiciones de carrera en
  `train_ids_v4.npy` o en el checkpoint). Construir en paralelo OK, activar después.
- **No** sobre-minar sin controlar el balance de dominios (el v4 ya tiene bioc/genom
  dominando; eco es el dominio clave del laboratorio y está en minoría).
- **No** tocar `pre_tokenize.py` a la ligera: cada cambio de corpus exige re-tokenizar.

---

## 3. Siguiente paso concreto (barato, sin GPU)

1. Escribir `dedup_v5.py` (o `build_v5.py`) que:
   - Lee v3 + PMC v4 (streaming, sin cargar todo en RAM).
   - Dedup por pmid con prioridad full-text.
   - Asigna dominio por ILIKE a los que no tienen.
   - Aplica `--max_per_domain` opcional.
   - Escribe `train_corpus_v5.jsonl` + un **reporte** (`v5_report.json`) con:
     docs totales, duplicados eliminados, % por dominio, % por source.
2. Correr en login node (CPU, ~30-60 min, sin GPU).
3. Revisar el reporte; si el balance/cobertura es bueno, tokenizar con
   `pre_tokenize.py` → `train_ids_v5.npy` (Slurm CPU, ~30-60 min).
4. Decidir con el usuario cuándo activar v5 vs terminar el v4 actual.

> Regla de labor: nada se ejecuta sin OK explícito. Este documento es el diseño;
> el paso 3.1 es un script de construcción (no destructivo, escribe archivo nuevo).