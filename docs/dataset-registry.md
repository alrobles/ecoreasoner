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

### ETAPA 1.5 — CONSOLIDADO DEDUPLICADO (papers_db, 18-09)

| Código | Ruta | Registro | desc |
|---|---|---|---|
| `1x_papersdb` | `data/papersdb/papers_db.jsonl` | **3,141,540 docs** · ~10.5B tok est. | **DB canónica deduplicada** de todos los corpus: key = pmid/pmcid/arxiv/doi o hash texto; prioridad fulltext>abstract. kind: 288,256 fulltext · 2,810,499 abstract · 40,756 synth · 2,029 unam-en. lang: 3.12M en / 23.4K es. 4.86M duplicados eliminados (v5 ⊂ v6, eco_corpus_v2 ⊂ v6). |
| `1x_skeldb` | `data/papersdb/skeleton_db.jsonl` | **1,036,993 esqueletos** | key→esqueleto (etapas≥3). src: 263,829 reused_v4en + 773,164 extracted (abstract-struct → IMRaD → phrases). |
| `1x_stats` | `data/papersdb/corpus_stats.json` | — | stats canónicas (`scripts/consolidate_papers.py stats`) |

- Dominios principales: bioc 380K, phylo 320K, genom 302K, medgen 302K,
  microbio 263K, eco 260K, phys-* 169K, climate 80K, marine 86K.
- **Regla**: papers_db es la fuente de verdad de texto; skeleton_db la capa
  estructural. Ninguno es corpus de entrenamiento directo — la elegibilidad
  pasa por filtrado/curriculum aparte (lección Nemotron: volumen≠señal).

### ETAPA 2 — CORPUS DE PRETRAIN (input directo de Fase A)

|Odigo | Tipo | V (reg) | tamaño | token/pal | desc |
|---|---|---|---|---|---|
| `2_pretrain_2` | merge abstracts+full+evorxiv | v2 | 6.5G | 1.0M docs | corpus v2 (merge_corpus) |
| `2_pretrain_3` | **corpus v3 activo** | v3 | 6.6G, 1,011,449 docs | ~1.7B tok | v2 + 1,728 EcoEvoRxiv; **≈ este es el que entrena** (Fase A 27324545) |
| `2_ids_3` | `train_ids_v3.npy` |  token-only | 1.4G, 363,234,015 tok | | **pre-tokenizado de 2_pretrain_3** (evita re-tokenizar). usado por train_*.py `--data_cache` |
| `2_ids_3_meta` | `train_ids_v3.npy.meta.json` | meta | | | metadata del npy (n_tokens, seq_len, vocab) |
| `2_pretrain_4en` | `data/corpus_v4_en.jsonl` | v4en (activo GA G8-G11) | 1.0G, **306,614 docs** | ~130M tok | corpus v3_en + UNAM traducido (sin ES crudo — lección g8-c1). Esqueletos con etapas `[OBSERVACION]..[CONCLUSION]` |
| `2_ids_4en` | `data/train_ids_v4en.npz` |  token-only | 531M | | pretok de `2_pretrain_4en` (`pre_tokenize_v2.py`, seq 768, ids+lengths+eos) |
| `2_hneg_pairs` | `data/hneg_pairs.jsonl` | aux gen-G11 | 70M, **20,444 docs** | | pair-docs minados de `emb_v1/audit_skeleton_v2/knn_edges.tsv` (sim≥0.93 dedup, diff mutable verificado, ≤768 tok/doc; `build_hneg_pairs.py`) |
| `2_pretrain_4hneg` | `data/corpus_v4hneg.jsonl` | v4hneg (exp. G11) | 1.08G, **327,058 docs** | 132.5M tok | `2_pretrain_4en` + `2_hneg_pairs` |
| `2_ids_4hneg` | `data/train_ids_v4hneg.npz` |  token-only | 531M, 132,540,055 tok | | pretok de `2_pretrain_4hneg`, 0 docs clipped, eos_id=0 |
| `2_pretrain_5pdb` | `data/corpus_v5_pdb.jsonl` | v5pdb (run 1B) | **3,445,211 docs** | 12.0B tok (medido) | papers_db EN completo (285K fulltext + 2.79M abstract + 40.7K synth + 2K unam-en) + 327K esqueletos v4hneg; `build_papersdb_corpus.py` |
| `2_ids_5pdb` | `data/train_ids_v5pdb.npz` |  token-only | | | pretok `2_pretrain_5pdb` --split-long: **17,384,808 docs, 11,964,472,366 tok**, 47.9GB, eos_id=0, 4597 clipped |
| `2_sft_chat1` | `data/chat_sft_v1.jsonl` | sft chat v1 | 38M, **36,677 pares** | 8.2M tok | sciq 11,679 + smoltalk 25,000 → `{prompt,response}` `[USER]/[ASSISTANT]`; `build_sft_chat.py` |
| `2_ids_sft1` | `data/sft_ids_v1.npz` |  token-only | | | pretok de `2_sft_chat1` con `resp_starts` (frontera prompt/response); `sft_pretok.py` |
| `2_qa_pass` | `data/qa_passages.jsonl` | pasajes QA | **150,620 pasajes** | | ventanas ~280-520 palabras de `2_pretrain_5pdb` (150k docs), LaTeX limpiado preservando math inline; `qa_extract.py` |
| `2_qa_raw` | `data/qa_raw/qa_raw_shard*.jsonl` | QA crudo teacher | ~600k pares (est) | | Qwen2.5-14B-Instruct vLLM, 4 pares/pasaje, JSON estricto; `qa_gen.py`+`qa_gen.slurm` |
| `2_sft_qa1` | `data/qa_sft_v1.jsonl` | sft QA v1 | ~550k pares (est) | | filtro groundedness (recall≥0.5 + números presentes en pasaje) + dedup → `{prompt,response}`; `qa_filter.py` |

- **Regla**: `2_pretrain_3` fue el corpus Fase A; el GA evolutivo (G8+) entrena
  sobre `2_pretrain_4en`/`2_ids_4en`. `2_pretrain_4hneg` es el corpus del
  experimento G11 (último gen de datos). `2_pretrain_5pdb`/`2_ids_5pdb` es el
  corpus del pretrain ~1B (v5-1b). `2_sft_chat1`/`2_ids_sft1` alimenta el
  chat-SFT (`sft_mdlm.py`). Las cadenas de Fase B y E usan estos nombres.

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

| Código | Tipo | Archivo | desc |
|---|---|---|---|
| `4_eval_sci` | sci eval | `sci_v2*.jsonl`, `sci_v1.jsonl` | prompts eval de razonamiento científico (tool-agnostic) |
| `4_bench` | benc | `benchmarks/`, `outputs/*/activation_report.json` | resultados de benchmark de bloco |
| `4_bench_afrisci` | bench exter | **`gimmy256/African-science`** (HF) | ⭐ cuestionario benchmark africano: ciencia en contexto africano (malaria, TB, HIV, biodiversidad Albertine Rift/Congo, clima Sahel/Lago Victoria, energía/recursos, instituciones Makerere/KEMRI/IITA, STEM educ). Instrucción-tuning web-grounded, generado gemini-2.5-flash, registrado para benchmark (user 2026-08-28) |
| `4_bench_nemotron_sci2` | bench exter | **`nvidia/Nemotron-SFT-Science-v2`** (HF) | ⭐ cuestionario benchmark ciencia: 2.84M filas / 49GB, dominios Physics/Biology/Chemistry, formatos Synthetic-MCQ + RQA + Vendor + SO-MCQ, soluciones generadas con GPT-OSS/Kimi-K2/DeepSeek-V3.2/V4-Pro, CC BY-SA 4.0. Registrado para benchmark (user 2026-08-28) |

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
## Purga de modelos de origen chino (2026-09-20)

Política: prohibido usar modelos de origen chino. Eliminados ~145GB de pesos
(Qwen2.5-14B, Qwen3.5-35B-A3B, Tencent HY-MT 1.8B/7B, LLaDA-8B/LLaDA-MoE-7B
weights, BAAI bge-small, DeepSeek refs) y el output QA derivado de Qwen
(`qa_raw_QUAR_qwen`, ~65k pares). ~40 scripts que dependían de esos pesos
llevan header `DEPRECATED 2026-09-20`.

**Dependencias no eliminables sin reentrenar** (decisión pendiente):
- Tokenizer LLaDA-8B (vocab 126080): horneado en pesos v5-1b, `2_ids_5pdb`,
  `2_ids_sft1` y todos los ckpts. Solo quedan archivos de tokenizer (~6MB).
- 2,029 docs `unam-en` dentro de `2_pretrain_5pdb`/`2_ids_5pdb`: texto
  traducido por HY-MT (Tencent), 0.06% del corpus, ya consumido por el run.

**Teachers compliant activos**: OLMo-2-13B-Instruct (Ai2, Apache 2.0) para
bulk QA (`qa_gen.slurm`, array 29921943); Devin/SWE-2-Max local para el
tier elite curado. Embedder permitido: multilingual-e5-small (Microsoft).

## Tier elite Devin/SWE-2-Max (2026-09-20)

Experimento multi-agente/multi-máquina: 40 sesiones `devin -p` (swe-2-max,
free promo) en 4-5 máquinas reumanlab, 430 pasajes de `2_qa_pass` con
provenance por pid. 860 candidatos → 747 validados por `qa_filter`
(87%). Lección: `--permission-mode dangerous` necesario para que las
sesiones lean líneas largas sin bloquearse en non-interactive;
throughput ~50 pares/min en flota limpia.

| id | archivo | n | notas |
|---|---|---|---|
| `2_qa_devin_elite` | `data/qa_devin_elite.jsonl` | **747** | pares con provenance completa (pid, type, machine, session, round, passage) |
| `2_eval_devin` | `data/eval_devin_hard.jsonl` | **500** | held-out eval curado: 242 numerical + 162 multihop + 81 negation — NUNCA entrenar |
| `2_sft_devin_supp` | `data/qa_sft_devin_supp.jsonl` | **247** | supplement SFT formato `[USER]/[ASSISTANT]`, src=devin-elite:<pid> |
