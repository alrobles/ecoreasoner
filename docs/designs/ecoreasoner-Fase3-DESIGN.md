# EcoReasoner Fase 3 — Diseño: harness + test-and-drop + escala heterogénea

Fecha: 2026-09-03
Autor: A.L. Robles Fernández (Hermes)
Estado: BORRADOR v0.2 — pendiente validación de Angel

---

## 0. Resumen ejecutivo

Construir un dLLM ecológico desde-cero competitivo requiere 3 correcciones simultáneas:
objetivo (el masking tiene un atajo local), datos (estructura del input) y escala
(0.5B tok/día actual vs ~15-20B tok necesarios). Este diseño separa las 3: un harness
hace que cada hipótesis cueste HORAS (no días), un micro-sweep decide el objetivo/datos
ganadores con criterios pre-registrados, y la escala heterogénea (pool NVIDIA completo,
hito b) produce el modelo Y el paper de systems con el mismo cómputo.

Regla de oro: **si una hipótesis no se puede validar en <4h de GPU, no se escala.**
Test-and-drop: el perdedor se descarta sin escalar.

---

## 1. Lecciones de bw0→bw4_span (lo que NO se repite)

| Experimento | Resultado | Lección |
|---|---|---|
| bw0/bw1v5 (base, random 15%) | loss 11.9→5.2, sin texto coherente | 37M tok vistos ≈ 3% del corpus: subentrenamiento severo |
| bw3 (random 15%, v7, 93K steps) | loss 2.0-4.2 oscilante; smokes: word 0.20-0.33, rep4 0.20-0.41 (colapso copia-vecino) | mask aleatoria dispersa premia copiar el adyacente; loss NO es proxy de lenguaje |
| bw4_span (span 64, v7, 1 epoch) | loss 6.98→5.58 luego PLANO (5.3-6.6, 28K pasos); smokes: rep4 0.003 (repetición erradicada) pero word 0.19-0.34, uniq 0.19-0.39 (sopa de subwords); 0/180 JSON | span rompe el atajo local pero no construye estructura; el objetivo solo no basta |
| SFT v1/v2/v3 (LLaDA-MoE-7B LoRA) | tool-calls 2%/0%/7%; EcoBench pass@1 1/14 (v2) | el modelo SÍ genera texto; el problema es data/formato tool-use |

Diagnóstico consolidado: (a) objetivos de masking al 15% tienen un atajo local que baja
la loss sin aprender estructura — confirmado con A/B limpio; (b) la escala actual es
inviable para lenguaje desde-cero; (c) los datos (prosa cruda de journals) no aportan
estructura explícita que el objetivo pueda capturar.

## 2. Realidad de cómputo (2026-09-03)

| Recurso | Cantidad | Notas |
|---|---|---|
| pro6000 (Blackwell) | 3 (r30r08, r30r24 = 2 GPU; r23r09 = 1) | únicos con torch 2.7.1 cu128; 14 steps/min × 24576 tok/opt-step |
| Q6000 | ~28 (7 nodos × 3, 2 nodos × 4) | SIF base 2.4.1 cu121; NCCL ~50ms/10MB; sin sm_120 |
| L40 | ~4 | — |
| A100 | ~4 | — |
| MI210 | 4 nodos (hpc_wang_5) | NO confiable (químicos/otros); no contar para Fase 1 |
| **Pool NVIDIA total** | **~35-40 GPUs** | ~10-14× el cómputo actual de 2 pro6000 |

Tasas:
- Actual (2 pro6000): 24,576 tok/optimizer-step × ~14 steps/min ≈ **0.5B tok/día**
- Pool completo (estimado): **5-7B tok/día** (asumiendo hito b = grad_accum adaptativo + all-reduce eficiente)
- Chinchilla para 863M: ~15-20B tok → **3-4 días de pool completo por entrenamiento base**
- 1 epoch de v7 (31,865 opt-steps) ≈ 1.58 días en 2 pro6000 → **~3h en pool completo**

## 3. Principio rector: coste por hipótesis < 4h GPU

El edge real no es "cientos de iteraciones GRANDES" (400 GPU-días y 8 meses), sino hacer
cada hipótesis barata:

| Escala | Coste | Iteraciones posibles |
|---|---|---|
| Micro-run (1 pro6000, modelo 150-400M, seq 128-256, 1-2B tok) | 1-4h, ~0.1-0.2 GPU-días | **100+ por mes con 1 GPU** |
| Run medio (2 pro6000, 863M, v7) | 1.58 días | ~20/mes |
| Run grande (pool completo) | 3-4 días | solo el GANADOR |

Regla: micro-run → go/no-go objetivo → escalar solo al ganador. Nunca escalar sin
criterio pre-registrado cumplido.

## 4. Harness de entrenamiento (Fase 0, ~48h) — el "CI del training"

Componentes (nuevos, reutilizando lo que ya funciona):

1. `harness/configs/run.yaml` — spec declarativa:
   ```yaml
   model: {hidden: 768, layers: 12, n_experts: 8, expert_k: 1}
   data: {cache: /beegfs/.../train_ids_v7_clean.npy, tokenizer: LLaDA-8B}
   training: {lr: 1.5e-4, warmup: 200, grad_accum: 4, batch: 4, seq: 768}
   masking: {type: span, span_len: 64, mask_p: 0.5}   # o random/none (causal)
   steps: {max: 3000, target_kTok: 2000}
   output: /beegfs/.../runs/RUN-<ts>/
   seed: 42, tag: "m2-span50"
   ```
2. `harness/run_micro.py` — valida config, arma slurm (1 GPU), ejecuta, recolecta.
3. `harness/report.py` — JSON estándar: suite smoke completa + loss curve summary +
   throughput + sha256(config) + seed. Escrito en `runs/<ts>/report.json`
   y append a `runs/index.jsonl` (append-only, comparable entre runs).
4. `harness/compare.py` — tabla A/B/N contra baselines del texto real
   (word=0.435, rep4=0.153, uniq=0.432) con deltas y veredicto go/no-go.
5. Suite smoke estandarizada (heredada de smoke_cont_bw3.py, que ya produce JSON):
   contexto real del corpus v7 (0/64/200/400 tok) × 6 docs × temps 0.7/0.9 × steps 64/128,
   métricas word_ratio/rep4/uniq_ratio + n_json_found. Siempre contra baselines reales.
6. `runs/` en beegfs = fuente de verdad; `index.jsonl` evita re-correr.

Reutilizar sin tocar:
- smoke_cont_bw3.py / smoke_cont_span.py (JSON con cfgs/samples/baselines) ✓
- train_mdlm_moe.py `--mask_type {random,span}` + `--span_len` ✓ (ya implementado)
- curate_embed vectors_full.npy (BGE-small) ✓
- train_ids_v7_clean.npy (corpus base 1.4M docs) ✓

Nuevo: registro de seeds + config hash + logs de loss en intervalos fijos (para que
compare.py pueda alinear curvas).

## 5. Micro-sweep (Fase 0b, 6 runs, ~12-24h en 1 GPU)

Todos los runs con el mismo harness, mismo data_cache, misma suite smoke.

| ID | Hipótesis | Variable | Criterio GO (ctx200, t0.7 s128) |
|---|---|---|---|
| M1 | control random 15% (replica bw3 a escala micro) | — (baseline) | calibrar harness, no go/no-go |
| M2 | span ratio ALTO 50-80% (T5/unified-io) | mask_p=0.5/0.8 | word≥0.35 && rep4≤0.25 |
| M3 | corpus SKETCHES (mapas conceptuales serializados) | data=sketch | word≥0.35 && rep4≤0.25 && uniq≥0.30 |
| M4 | mix 70% texto + 30% sketches | data=mix | word≥0.38 |
| M5 | apriorismo sintáctico: 25% FineWeb-Edu (filtrado) + 75% v7 | data=mix-edu | word≥0.40 && rep4≤0.20 |
| M6 | control causal next-token (techo de referencia) | trainer=causal | word≥0.435 |

Decisión (test-and-drop): el ganador (criterio GO + estabilidad en 2 seeds) define la
receta de Fase 1. Los perdedores se descartan y se documenta el porqué en runs/.
Si NINGUNO gana → iterar el objetivo con coste < 4h por intento.

## 6. Corpus de SKETCHES (la idea de Angel, formalizada)

Entrada: los 1.4M papers curados (v7_clean + phys) en texto plano. Nada de n-grams
(refuerzan el atajo local). Nada de RedPajama crudo (basura para 863M).

Extracción (fase A — CPU/mesh, 0 GPU nueva):
1. Heurística spaCy (pattern-based) sobre el texto: relaciones x → y con verbos de
   señal ("X is associated with Y", "X drives Y", "X inhibits Y", "X predicts Z") →
   serialización fija: `X --regulate--> Y ; Z --inhibit--> W` (texto plano).
   Escalable: 8 shards × 16 cpus en el clúster CPU (patrón ya usado).
2. (Opcional, fase B) Extracción LLM con glm-4.7-flash vía ollama HPC o deepseek-flash
   vía OpenRouter para relaciones de alta señal que la heurística pierde.

Serialización final: ~1.4M sketches → train_ids_sketch.npy (estimado 0.5-1B tok).

Uso (2 fases): fase 1 = denoising de sketches (aprende estructura del conocimiento);
fase 2 = texto completo (aprende a rellenar la estructura con lenguaje). El input
cambia de estructura → el objetivo ya no tiene atajo local.

Justificación (lit.): sketch-to-text / planning-then-denoising; los "mapas
conceptuales" serializados de forma plana son representación estructurada del
conocimiento.

## 7. Escala heterogénea (Fase 1) — el hito b productivo

- train_hybrid.py ya existe (grad_accum adaptativo por VRAM, all-reduce manual de
  gradientes; validado single-GPU; pendiente el device-binding LoRA multi-nodo).
  Extensión: aplicar el patrón al trainer full del dLLM (no solo LoRA).
- Pool: 28 Q6000 + 4 L40 + ~4 A100 + 3 pro6000 = ~35-40 GPUs.
  NCCL: 0.1-0.2 GB/s en Q6000 → all-reduce 3.4GB = 17s/paso VANILLA → exige
  sync_every (no_sync DDP) o all-reduce manual de gradientes para llegar a 5-7B tok/día.
- Micro-batch por rank según VRAM; grad_accum adaptativo = gradiente global equivalente
  (batch global idéntico con micro-batch distinto — ya diseñado en train_hybrid.py).
- Slurm: patrones ya validados (overlay + torchrun intra-contenedor, lock del overlay,
  SIGUSR1 waves, checkpoints atómicos + resume tolerante — ver pitfalls del skill).

## 8. Publicación (Fase 2) — 2 papers con el mismo cómputo

- Paper 1 (systems): entrenamiento heterogéneo multi-GPU con grad_accum adaptativo por
  VRAM + harness reproductible (el hito publicable de Angel; no depende del dominio).
- Paper 2 (dominio): primer dLLM ecológico desde-cero (no existe equivalente público —
  TinyLlama/Pythia son causales). "Competitivo" = superar baselines scrub en
  eco/tool-use + ablaciones de objetivo, NUNCA contra DeepSeek general.

## 9. Mesh + agentes + openrouter (integración de la org)

Asignación de roles (patrón mini-factory, GitHub Issues como bus si hace falta):

| Nodo | Rol | Tarea Fase 0/1 |
|---|---|---|
| alpha (62GB, R, apptainer) | Data eng | implementar extractor sketch (spaCy) + shards; integración beegfs |
| beta (14GB, ligero) | QA / validación | dedup sketches, validación formato, checks de cobertura por dominio |
| terminal (30GB) | Dev harness | desarrollo local del harness (run_micro/report/compare) + tests |
| gamma (P620) | Revisión | review de resultados A/B (hermes chat -q, barato) |
| reumanlab (hub) | Orquestador | cron watchdogs + comparativas + decisión go/no-go |
| HPC (beegfs+slurm) | Cómputo | micro-runs, extracción sketch, entrenamiento grande |

Storage: beegfs = fuente de verdad (runs/, corpus/); nodos mesh trabajan contra
/beegfs y suben código al repo alrobles/ecoreasoner; GitHub Issues (etiqueta
hermes-task) como bus asíncrono para tareas durables entre nodos.

OpenRouter (con el saldo nuevo que Angel va a agregar):
- Agentes de curaduría/QA: ling-3.0-flash (0.021/0.063 $/M) — el más barato, suficiente.
- Extracción LLM de sketch (opcional): deepseek-v4-flash (0.06/0.12) o glm-4.7-flash
  vía ollama (gratis local). Estimar: ~$1-3/M papers (in≈500 tok, out≈150).
- Presupuesto Fase 0 total: $5-20 (agentes + micro-QA). NO depender de :free en producción.

## 10. Roadmap con hitos medibles

| Hito | Qué | Duración | Criterio de salida |
|---|---|---|---|
| H0 | harness + 6 micro-runs + decisión | 48-72h | ganador del sweep con GO cumplido; index.jsonl con 6 entradas |
| H1 | corpus sketch (extractor + serialización) | 1-2 sem (CPU/mesh) | train_ids_sketch.npy + cobertura ≥70% de los papers |
| H2 | entrenamiento pool completo (hito b extendido) | 2-4 sem | 15-20B tok vistos; smoke word≥0.40, rep4≤0.20 |
| H3 | modelo final + eval EcoBench + paper 1 draft | mes 2-3 | pass@1 EcoBench > baselines scrub; draft paper 1 |

## 11. Riesgos y mitos

- Escalar antes de Fase 0 = repetir bw3 con más GPUs (coste 10×). NO.
- Loss como proxy: NUNCA solo. Siempre + smoke de continuación con contexto.
- N-grams: NO (atajo local). RedPajama crudo: NO (basura para 863M). FineWeb-Edu filtrado: SÍ como apriorismo 25%.
- "Cientos de iteraciones": solo útiles si cada una cuesta <4h y el resultado es comparable.
- MI210: no contar (no confiable). Blackwell: solo 3 tarjetas — usarlas para los micro-runs (son las rápidas) y asumir que el pool Q6000 hace el grueso.
- Resume/olas: usar el patrón validado (SIGUSR1, checkpoints atómicos, TARGET_STEPS absoluto) — no reinventar.

---

## Apéndice A — Archivos que se tocan / crean

| Archivo | Acción |
|---|---|
| `scripts/train_mdlm_moe.py` | NO tocar (lo usa el run vivo) — el harness llama con flags |
| `scripts/smoke_cont_bw3.py` | heredar a `harness/suite_smoke.py` (misma salida JSON) |
| `harness/` (nuevo) | run_micro.py, report.py, compare.py, suite_smoke.py, configs/ |
| `scripts/build_sketch.py` (nuevo) | extractor spaCy + serialización |
| `data/sketch/` (nuevo) | sketch jsonl + train_ids_sketch.npy |
| `runs/` (nuevo, en beegfs) | report.json + index.jsonl + checkpoints |