# EcoReasoner — Reporte técnico para paper (2026-08-28)

> Consolidación del trabajo de datos + arquitectura + training listo para citar
> en el writeup del dLLM-MoE "Agentic Scientist". Todos los números son medidas
> reales tomadas del clúster (KU HPC) en 2026-08-28, no estimaciones.
> Autor: A.L. Robles Fernández (lab ReumanLab) · Estado: reporte de datos/entrenamiento

---

## 1. Pipeline de corpus (la base del pretrain)

### 1.1 Genealogía

| Versión | Docs | Tokens | Construcción |
|---|---|---|---|
| v1 | 44,045 | ~165M | abstracts PubMed FTS |
| v2 | 1,009,725 | ~1.4B | abstracts + full-text PMC, merge por pmid (full gana) |
| v3 | 1,011,449 | ~1.7B | = v2 + 1,728 EcoEvoRxiv fulltext |
| v4 | 2,749,947 | 1.697B | = v3 + 1.74M PMC full-text, **concat sin dedup** (decisión de ese día) |
| **v5** | **1,905,424** | **1.118B** | = v3 + PMC con **dedup global por pmid**, 12 dominios gruesos, **etiqueta fina MeSH** |

### 1.2 Hallazgo clave del v4 → v5

El v4 acumuló duplicados y desetiqueta:
- Duplicados por pmid: ~52K (2,697,557 pmids únicos de 2,749,947 docs = 1.9%)
- 96K docs (9.6%) con `domain=None` (los PMC)
- Desbalance: bioc/genom dominaban, eco en minoría (182K de 1M)
- Tokenización ineficiente: 2.75M docs para solo 1.697B tok

**El v5 arregla todo con 3 mecanismos** (cuantificados):
1. **Dedup global por pmid** con prioridad full-text (PMC gana a abstract del mismo paper): 1,549,947 descartados (dups reales + fuera de cuota de balance)
2. **Balance por dominio** (`--max-per-domain`): 4 dominios grandes a 300K exactos
3. **Etiqueta fina MeSH** (sección 3): cobertura 100% de `domain_fine`

### 1.3 El "ruido" del corpus (observación metodológica)

Los full-text PMC traen prefijos "JOURNAL INFORMATION... NLM Title Abbreviation..."
al inicio. Para pretrain es ruido tolerable (el modelo aprende a ignorarlo), pero
para calidad de etiquetado conviene conocerlo. No se limpió en v5 (decisión: no
tocar text; solo añadir metadata).

---

## 2. Etiqueta fina con MeSH (el contributo metodológico de hoy)

### 2.1 Por qué MeSH

El corpus es PubMed/PMC de NIH. El esquema de 4 dominios original
(eco/phylo/genom/bioc, ILIKE) capturaba mal la realidad: un probe sobre 30K docs
mostró que **genética médica = 46.8%** del corpus, indistinguible de "genom".
La ontología real de NIH (MeSH) da la etiqueta fina correcta.

### 2.2 Cobertura real (medida)

- **36.35%** de los docs tienen MeSH (692,591 de 1,905,424) — los abstracts v3
  (82.7% con MeSH); los **PMC full-text son de revistas no-MEDLINE → 0% MeSH**
- El resto (64%) se etiqueta por **clasificación de texto** con la misma taxonomía
  → cobertura final de `domain_fine` = **100%**
- PMCIDs → PMID via `PMC-ids.csv.gz`: **96%** traducidos (1,864,678 de 1,070,774)

### 2.3 Taxonomía fina (~40 dominios, 3 familias)

- **ECO (19)**: sdm, community_ecology, population_ecology, conservation,
  climate_ecology, landscape_ecology, macroecology, evolutionary,
  phylogeography, metagenomics, microbiology, plant_biology, marine,
  soil_ecology, paleoecology, animal_behavior, ecoevo, disease_ecology,
  ecotoxicology
- **BIOMED (16)**: cardiovascular, oncology, infectious, immunology, neurology,
  endocrinology, respiratory, renal, public_health, pharmacology,
  medical_genetics, psychiatry, nutrition, surgery, geriatrics, pediatrics
- **GENERAL (5)**: molecular_biology, cell_biology, genetics_general,
  biochemistry, methods_stats

### 2.4 Regla de señal relativa (control de falsos positivos)

El substring matcher ingenuo produce falsos positivos absurdos (verificado):
- paper de salud materna en Kenia → "population_ecology" ✗
- fisiología piel/temperatura → "climate_ecology" ✗
- tuberculosis → "sdm" (por "species specificity") ✗

**Fix**: marcadores clínicos fuertes (`Humans, Female, Male, Adult, clinical,
patient, disease, therapeutic, drug therapy, treatment outcome, health`) pesan a
biomedicina; eco domina solo con ≥2 hits y ≥ que biomed. **Validación**: de los
etiquetados eco, solo 3.7% tienen ≥3 marcadores clínicos (falsos positivos
controlados).

### 2.5 Distribución fina final (medida, top 15)

| dominio | docs | | dominio | docs |
|---|---|---|---|---|
| molecular_biology | 567,587 | | psychiatry | 30,019 |
| methods_stats | 210,175 | | respiratory | 25,725 |
| infectious | 128,187 | | genetics_general | 24,958 |
| oncology | 102,315 | | cardiovascular | 24,647 |
| other | 94,374 | | medical_genetics | 23,593 |
| neurology | 85,547 | | plant_biology | 21,326 |
| public_health | 79,141 | | metagenomics | 20,525 |
| surgery | 67,197 | | soil_ecology | 18,651 |

Ecológicos clave: climate_ecology 14,066 · phylogeography 12,133 ·
evolutionary 10,732 · conservation 9,893 · population_ecology 9,885 ·
community_ecology 8,273 · marine 7,512 · macroecology 2,905 · sdm 1,838 ·
ecoevo 432.

---

## 3. Arquitectura dLLM-MoE (validation status)

### 3.1 El problema que se resolvió (100% diagnosticado)

**MoE + DDP + batch pequeño = deadlock.** Con batch 1/GPU, cada rank rutea sus
pocos tokens a un subconjunto DISTINTO de expertos → expertos inactivos no reciben
grad → DDP exige `find_unused_parameters=True` (OOM) o rechaza con:
`Expected to have finished reduction... Parameter indices which did not receive grad`
(índices 90-397 = los expertos no activados).

### 3.2 El fix (probe + aux real)

- El `balance_loss` original con `P.detach()` mataba el grad a TODO (incluso el
  router) — era puro ruido
- **Fix**: (1) `_gate_probs` diferenciable → aux de balance pasa grad real al
  router; (2) **probe**: un token por iteración pasa por CADA experto → todos los
  expertos reciben grad siempre → `find_unused_parameters=False` sin deadlock
- Smoke test CPU verifica: TODOS los expertos con grad ≠ None, gate con grad,
  sin NaN

### 3.3 Validación en hardware (medida)

**Smoke Blackwell** (job 27477584, 2× RTX PRO 6000, sm_120, torch 2.7.1+cu128):
- DDP world=2, 30 steps, batch 4/rank, sin OOM ni deadlock
- Loss: step0 ~11.9 → step30 **9.37**, checkpoint g30 OK, rc=0

**Pitfall de despliegue descubierto**: `srun -n N apptainer exec --overlay <file>`
→ los ranks montan el MISMO overlay ext3 en paralelo → lock de escritura →
`exit 255` (síntoma: TCPStore timeout). **Fix**: `--ntasks=1` + `torchrun
--nproc_per_node=2` DENTRO de un único contenedor.

### 3.4 Entrenamiento real en curso (medida)

| | Ola 1 (27477651) | Ola 2 (27484386, RUNNING) |
|---|---|---|
| nodo | r30r24n01 | r30r24n01 |
| duración | 5:45:47 | 1:01:29 (y subiendo) |
| cierre | exit 42 (SIGUSR1 boundary, limpio) | — |
| estado | checkpoint g851 | step ~900, loss ~6.5-6.7 |

Config: corpus v4 (1.697B tok; el v5 quedará para la siguiente tanda), MoE 4
expertos top-1, hidden 1024, 16 layers, seq 768, batch 4/rank, grad_accum 4,
lr 2e-4, warmup 800, max 6000 steps, olas SIGUSR1@300 + AUTO_RESUBMIT.

Loss pretrain v4 (referencia): step 0 ~12.0 → step ~600 ~6.5 (la loss alta es
esperada: vocabulario 126,080, masked-diffusion D3PM absorb-mask).

---

## 4. Infraestructura operativa (datos para la sección de métodos)

- **Clúster**: KU HPC, partición sixhour (walltime 5:50, 333 nodos)
- **Blackwell**: solo 3 nodos pro6000 — r30r08n01 (2 GPU, usado por el teacher
  ollama-v4serve), r30r24n01 (2 GPU, training), r23r09n01 (1 GPU)
- **Teacher destilación**: DeepSeek-V4-Flash 284B (MXFP4) en 2×pro6000 vía
  Ollama, puerto efímero por job, túnel SSH local :20006 (solo en reumanlab;
  desde HPC se apunta directo al nodo:puerto interno)
- **Stack**: apptainer SIF (pytorch-cuda.sif) + overlay ext3 20GB con venv
  limpio (torch 2.7.1+cu128 para sm_120; SIF base 2.4.1 NO soporta Blackwell)
- **Datos**: `/beegfs/a474r867/ecoreasoner/data/` · PubMed parquet 30.8M filas /
  9.9GB · PMC-ids.csv.gz 252MB
- **Regla operativa**: ningún job autónomo sin autorización; todo build/fine-label
  solo vía Slurm (guardas SLURM_JOB_ID), nunca login node

---

## 5. Eficiencia del pipeline (tiempos reales medidos)

| Paso | Job | Tiempo | Nota |
|---|---|---|---|
| build v5 (dedup+balance+12 dom) | 27478899 | 34 min | CPU |
| pre-tokenize v5 | 27479048 | 21.7 min | 200G, 16 workers |
| fine-label v5 (MeSH+texto) | 27485400 | 46.4 min | CPU, duckdb+parquet |
| smoke Blackwell | 27477584 | ~5 min | 30 steps |
| join duckdb IN naive | (cancelado) | >27 min | 10 escaneos parquet |
| join duckdb único | 27485400 (pase 2) | ~7 min | 1 escaneo + filtro Python |

**Pitfall duckdb**: `WHERE pmid IN (200K params)` escanea los 1456 parquet 10×.
Fix: un solo `SELECT pmid, mesh` (27M filas con mesh) + filtro en set (~7 min).

---

## 6. Pendiente / próximos pasos (para el writeup)

1. **Entrenar sobre v5** (con etiqueta fina como señal auxiliar o análisis de
   especialización por dominio en el MoE — "usar la arquitectura a nuestro favor")
2. **EcoBench-EVAL** (Fase 4): sin él no hay métrica piso publicable
3. **B1 destilación**: reintentada con fixes (endpoint dinámico + clave OpenRouter
   multi-ruta); 442/442 trazas code_valid previas, tanda nueva en curso
4. Evaluar si los 12 dominios gruesos (build) vs ~40 finos (MeSH) sirven como
   señal de curriculum o routing condicional