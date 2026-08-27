# EcoReasoner — Dataset Registry (nomenclatura canónica)

> Fuente de verdad para los datasets del dLLM-MoE "Agentic Scientist" (EcoReasoner).
> Este documento define el **código canónico** de cada dataset y a qué **etapa** del
> pipeline pertenece, para que no haya ambigüedad cuando distintas cadenas corren en
> paralelo (pretrain, destilación, eval, overlap de teachers).
>
> Fecha: 2026-08-27 · Estado: v1 (aplicables desde ahora)

## Esquema de nombre canónico

Cada dataset tiene un **código estable** compuesto por `ETAPA__TIPO` y una **ruta real**
en `/beegfs/a474r867/ecoreasoner/`. El código es lo que se usa en scripts y en este
registro; la ruta puede cambiar si migramos de directorio, el código no.

```
  ETAPA  __  TIPO  __  VERSION
   │            │         │
   A            pretrain(S)  v3
   T            distill   v4r3
   E            eval      ...
```

- **ETAPA**: `1` = ingesta/raw, `2` = pretrain corpus, `3` = destilación, `4` = eval/benchmark, `P` = prompts, `X` = auxiliar/inventario.
- **TIPO**: description corta (pretrain, distill, traj, prompt, eval, license, inv, meta).
- **VERSION**: números de versión (sin `v` para menos ruido).

## Registro actual (por etapa)

### ETAPA 1 — FUENTES (ingesta, sin entrenar directamente)

| Código | Ruta | Registro actual | desc |
|---|---|---|---|
| `1_pmc_full` | `fulltext_corpus.jsonl` | 596M · ~ f | PMC fulltext bruto (vía map_pmc_fulltext) |
| `1_pmc_c0..c3` | `fulltext_corpus_c{0..3}.jsonl` | 1.2G×4 | Shards PMC (fetch_pmc_shard) |
| `1_pubmed` | `eco_corpus.jsonl` | 67M ~? | Abstracts PubMed (miner duckdb) |
| `1_pub_v2` | `eco_corpus_v2.jsonl` | 1.6G, 1,000,000 docs | Abstracts PubMed multi-dominio |
| `1_evorxiv_c0..c3` | `ecoevorxiv_fulltext_c{0..3}.jsonl` | ~30M×4 | Fulltext EcoEvoRxiv shards |
| `1_eco` | `ecoe_corpus.jsonl` | (obsoleto) | precursor de eco_corpus_v2 |
| `1_full` | `fulltext_corpus_all.jsonl` | 5.1G, 97,012 docs | Unión shards PMC |

- Los shards `_c0.._c3` se unen a `fulltext_corpus_all.jsonl` (ETAPA 1).

### ETAPA 2 — CORPUS DE PRETRAIN (input directo de Fase A)

|Odigo | Tipo | V (reg) | tamaño | token/pal | desc |
|---|---|---|---|---|---|
| `2_pretrain_2` | merge abstracts+full+evorxiv | v2 | 6.5G | 1.0M docs | corpus v2 (merge_corpus) |
| `2_pretrain_3` | **corpus v3 activo** | v3 | 6.6G, 1,011,449 docs | ~1.7B tok | v2 + 1,728 EcoEvoRxiv; **≈ este es el que entrena** (Fase A 27324545) |
| `2_ids_3` | `train_ids_v3.npy` | token-only | 1.4G, 363,234,015 tok | | **pre-tokenizado de 2_pretrain_3** (evita re-tokenizar). usado por train_*.py `--data_cache` |
| `2_ids_3_meta` | `train_ids_v3.npy.meta.json` | meta | | | metadata del npy (n_tokens, seq_len, vocab) |

- **Regla**: `2_pretrain_3` es el corpus de pretrain **actual**; `2_ids_3` es su tokenización **única**. Las cadenas de Fase B y E usan estos nombres.

### ETAPA 3 — DESTILACIÓN (input de Fase B/C)

|Código | Tipo | Arch | desc |
|---|---|---|---|
| `3_toolcall` | prompts canónicos tool-call | `prompts_toolcall_canonical.jsonl` | 30 prompts de tool-call de referencia |
| `3_distill_0` | trayectorias gen_distill_data | `distill_data.jsonl` | primera gen distill (teacher loop) |
| `3_distill_r1` | trayectorias v4 round 1 | `distill_v4_round1.jsonl` | round 1 (v4) |
| `3_distill_r2` | trayectorias v4 round 2 | `distill_v4_round2.jsonl` | round 2 |
| `3_distill_r3` | **trayectorias v4 round 3 (activo)** | `distill_v4_round3.jsonl` | round 3 (terminó 49 tray; ver logs) |

- **teachers goal** actuales (tier 2/3): `deepseek-v4-flash` (r30r08n01:42321), `qwen3.6:35b` (r08r28n01:54249, r08r30n01:44003), `glm-4.7-flash:q4` (r22r10n01:60431).

### ETAPA 4 — EVAL / BENCHMARK

|Código | Tipo | Archivo | desc |
|---|---|---|---|
| `4_eval_sci` | sci eval | `sci_v2*.jsonl`, `sci_v1.jsonl` | prompts eval de razonamiento científico (tool-agnostic) |
| `4_bench` | benc | `benchmarks/`, `outputs/*/activation_report.json` | resultados de benchmark de bloco |

### T — PROMPTS de toolcall (no son trayectorias)

| Código | Archivo | desc |
|---|---|---|
| `T_toolcall` | `prompts_toolcall_canonical.jsonl` | (vuelca con 3_tool_0) |

### X — INVENTARIO / PROVENANCE

| Código | Archivo | desc |
|---|---|---|
| `X_inv_*` | `*.inv.gz` | inventarios S3 del bucket PMC |
| `X_prov` | `provenance_manifest.jsonl` | license/origin de cada doc |
| `X_lic_pmc` | `pmc_license_map.jsonl`, `ecorevorxiv_license_map.jsonl`, `ecorevorxiv_osf_map.jsonl` | maps de licencia |

## Fuentes de verdad en los scripts

- `pre_tokenize.py` usa `2_pretrain_3` → escribe `2_ids_3`.
- `train_mdlm_moe*.py` usan `--data_cache 2_ids_3` (`.npy`).
- `gen_distill_data.py` consume `3_tool_0` y escribe `3_distill_rN`.
- `merge_corpus.py` genera `2_pretrain_N` desde `1_pub_v2` + `1_pmc_all`.
- `analyze_activation*.py` lee un ckpt de `outputs/` (modelo, no dataset).

## Reglas para no confundirlos

1. **Nombra por ETAPA primero** (`2_`, `3_`, ...) al crear datasets nuevos.
2. **nunca** reescribas un nombre antiguo en un script nuevo. Usa el código de aquí.
3. El **dataset "activo"** de cada etapa está marcado con **`(activo)`** arriba.
4. Un `npy` siempre es una tokenización **de** un jsonl `2_pretrain_N`: ambas comparten versión.

> NOTA: los tamaños y contadores de la tabla son del 2026-08-27 y se revalidan con
> `python3 scripts/dataset_catalog.py` (que lee el registro vivo desde el disco).