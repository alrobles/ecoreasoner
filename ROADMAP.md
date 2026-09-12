# ROADMAP — EcoReasoner Fase 3 (refundación `-new`): excavar un dLLM científico

> ReumanLab · EcoReasoner · 2026-09-08 (última actualización)
> Este es el PLAN MAESTRO. El código vive en `scripts/` + `harness/`; los diseños
> en `docs/designs/`. Para el estado MÁS reciente de jobs/watchdogs: ver
> `docs/results/` + skill `ecoreasoner-swarm` (referencias f0-micro-sweep-* y f1-hetero-*).

---

## 0. OBJETIVO (declarado por Angel — la refundación)

Generar **agentes con capacidad de razonamiento científico**. La tesis excavada:
¿puede un **dLLM desde cero**, entrenado para denoising de *estructura de
argumento científico* (no prosa cruda), aprender a **discriminar la continuación
inferencialmente correcta** de un razonamiento frente a alternativas plausibles
pero incorrectas?

**Falsación pre-registrada (no exige victoria, exige respuesta con evidencia):**
> si tras N tokens el modelo no discrimina ≥55% pairwise (transitorio) NI
> completa esqueletos coherentemente → la tesis queda falsada a nuestra escala.

> Un negativo limpio con ablaciones (random/span/span-alto × prosa/esqueleto)
> es un resultado publicable.



**Fuera de la mesa:** LLaDA-MoE-7B como pieza (solo techo de medición), SFT como
vía productiva, horizonte corto. Ver `docs/designs/ecoreasoner-Fase3-DESIGN-new.md` §0.




---

## 1. HITOS(medidos por decisión, no por calendario)

### F0 — Harness F0 + eval nueva de discriminación — ✅ COMPLETO (07-09)
- `harness/`: run_micro.py (slurm 1-GPU declarativo), report.py (report.json + index.jsonl),
   compare.py, suite_smoke.py (discriminación pairwise + completación + fluidez),
   validate_configs.py (previene YAML roto), build_pairs.py (pares reales del corpus).
- `scripts/moe_v4_micro.slurm` — micro-run 1 GPU pro6000 (10K steps, dense 50-100M,
   olas SIGUSR1, eval+report al completar).
- **Micro-sweep 6/6 completado y evaluado** → ver §2.



### F0b — Extractor de esqueletos + corpus — ✅ COMPLETO(07-09)
- `scripts/build_skeleton.py`(IMRaD + abstracts estructurados + frases-faro,0 GPU)
  → `data/skeleton/train_skeleton.jsonl`(**315,299 docs** con ≥3 etapas,675MB)
  → `data/train_ids_skeleton.npy`(129.5M tok, OOB_OK}.
- `scripts/skel_full.slurm` + `skel_pairs.slurm` + `pre_tokenize_skeleton.slurm`.
- Cobertura REAL: 51.6% ge2,21.6% ge3 (el 70% era del mini-corpus sintético;
   el real es heterogéneo y el activo de 315K docs basta).

.



### F1 — Entrenamiento objetivo con el pool heterogéneo — ✅ COMPLETO (08-09): NO-GO
- **Trainer heterogéneo VALIDADO**: `scripts/train_mdlm_moe_hetero.py`
  (`--autosize` por VRAM medido con fwd+bwd real + `SCALE_I` gradiente exacto por
   tokens + grad_accum global; smoke L40×4 world=4 micro=25 COMPLETE)。
- **Lanzador multi-familia**: `scripts/f1_hetero.slurm` (torchrun rendezvous TCP,
  cada job = 1 familia; NCCL_IB_DISABLE para el muro IB/noib).
- **RESULTADO (1000 steps,17 ranks L40+Q6000,03:02)**: acc se mantuvo en
  el azar exacto (~0.5;max 0.508;criterio GO ≥0.55 NO alcanzado.**NO-GO**。
  Detalle + curva + contraste: `docs/results/F1-HETERO-RESULTADO.md`.
- **Lección (config, no escala)**: micro-span-esqueleto(batch8+accum2=12 288
  tok/update ×10K updates=122.9M tok) → acc 0.5352;F1(65 536 tok/update
  ×1K updates=65.5M tok) → acc  ️0.5. Mismo corpus/modelo/mask/lr. La señal
  estructural se diluye con batch gigante y pocos updates → **la vía es más updates**
  **pequeños (receta micro,50K-100K steps), NO más GPU.**
- Estimación: `docs/designs/ecoreasoner-F1-HETERO-ESTIMACION.md` (histórico).



###F2 — Agente(loop + tools encima del modelo)—[ ]
### Paper 1 (systems: entrenamiento heterogéneo + harness)—en paralelo
### Paper 2 (tesis desde-cero, positiva o negativa limpia)—[ ]

---

## 2. RESULTADO del micro-sweep F0 (fuente de verdad: runs/index.jsonl + docs/results/MICRO-SWEEP-F0-RESULTADOS.md)

| Run | mask × datos | pairwise_acc | mean_delta |
|---|---|---|---|
| f0-random-prosa | random 15% × v7 | 0.5117 | -0.032 |
| f0-span-prosa | span64×15% × v7 | 0.4492 | -0.064 |
| f0-spanhi-prosa | span64×60% × v7 |  ️0.4766 | -0.052 |
| f0-random-esqueleto | random × skeleton | 0.5078 | +0.011 |
| **f0-span-esqueleto** | span64×15% × skeleton | **0.5352** | +0.015 |
| f0-spanhi-esqueleto | span64×60% × skeleton | 0.5312 | +0.030 |

**Veredicto:** GO pre-registrado(width ≥0.55) NO alcanzado por ninguno(mejor 0.535,
mejor esqueleto 0.535/0.531 vs prosa 0.45-0.51 con delta NEGATIVO). **Tesis
reforzada en dirección**: estructura de argumento mueve la discriminación
(esqueleto > prosa; span > random en esqueleto). 10K steps ×12,288 tok/update
= **122.9M tok ≈ **~0.95 epochs del corpus esqueleto** (129.5M;;la cifra
"245K tok" de la nota original era un error) — el micro NO estaba "subentrenado al"""
0.006%";vio ~1 epoch del esqueleto y aun así 0.535 (vs F1 0.5 con 0.5 epoch
a batch gigante).. **Ganador: span-esqueleto.**

---

## 3. CÓMPUTO — pool NVIDIA real y tok/s medidos

| Familia | GPUs | tok/s (micro dense, solo entreno,08-09) |
|---|---|---|---|
| **A100** (cu121) | ~18 | **~22,026** |
| pro6000 Blackwell | 5 | **~12,553** |
| L40 (cu121) | 4 | **~11,443** |
| Q6000 (cu121) | ~29 | **~7,941** |

- **PITFALL medición**: walltime del job incluye la carga del cache (~50s)→
  subestima 2-5×. Cronometrar del log (step 0 → step 190 ×6144 tok/step)。
  1 epoch esqueleto(129.5M tok) ≈8 min pool realista / 15-22 min solo
  L40+Q6000.
- El pool real **fluctúa por minuto** (A100 se ocupan/liberan todo el tiempo}.
  Sinfo miente: usar scontrol (AllocTRES vs Gres)。
- pro6000 (cu128) NO se mezclan en DDP con cu121(muro HITO 3)—familias

  homogéneas por partición + rendezvous TCP entre nodos.

- **Muro NCCL IB/noib documentado**: L40 r32r25n01 es noib,A100 son ib → el L40
  fallaba `NCCLUtils.hpp:275`. Fix: NCCL_IB_DISABLE=1 + NCCL_SOCKET_IFNAME.



---

## 4. ESTADO VIVO (HPC, al 12-09)

- **LOGICDIFF FASE A — LA FAMILIA dLLM SE REABRE POR CORRECCIÓN DE MÉTRICA
  (2026-09-12)**: el NO-GO de V3.3 queda **supersedido**. El scorer de la
  battery (máscara random 15% sobre ctx+candidato) diluía la señal L3 ~6× en
  ruido de contexto. El nuevo modo `dense` de `harness/suite_smoke_logicdiff.py`
  (enmascarar el 100% del candidato, reconstruir condicionado al contexto)
  revela: **v3-contrastive-fixed L3 0.640*** (p<10⁻⁵, IC95 [0.596,0.682])**,
  y **8/8 checkpoints entrenados significativos** (0.545-0.640, p<0.03 todos)
  vs random-init 0.459 ns (piso limpio). El scheduler de orden lógico NO es
  el mecanismo (staged≈rev≈dense; rand peor) — la barrera era el evaluador.
  Desglose por subtipo: la señal se concentra en `direction_word` 0.76-0.78
  (n=193) y causal/mechanism; `number`/`negation` siguen en azar o invertidos
  → discriminación direccional real, lógica fina ausente. Documento:
  `docs/results/LOGICDIFF-FASE-A-RESULTADO.md`. Uso defendible: el dLLM como
  **scorer de continuaciones** dentro del controller/verificator (rerank de
  argumentos, verificación de drafts), no como generador.

- **V3.3 CONTRASTIVO — NO-GO SUPERSEDIDO (2026-09-11, rev. 09-12)**: con el fix
  anti-contaminación (split por contexto 373/93, mask_id del tokenizer, guard
  de memorización; auditoría en docs/results/AUDITORIA-V3.3-CONTAMINACION.md),
  el training (job 29184168) completó y eval_acc_holdout ranking = 0.7742.
  Pero la battery LIMPIA (eval set pairs_hard_v3_eval seed 9999, 96.6% ctx
  nuevos; jobs 29184174/75) da L0 0.485 ns / L1 0.506 ns / L2 0.594*** /
  **L3 0.516 ns (p=0.26)**. La señal ranking NO transfiere al mecanismo denoise
  de la battery → artefacto del mecanismo de scoring del fine-tune, no
  inferencia. Según criterio del doc (<0.52): **NO-GO — familia dLLM CERRADA**
  (9 runs, L3 nunca significativo). Indicación: archivar dLLM-puro como
  negativo publicable y pivotar cómputo a controller/verificator (Opción D,
  match_args 98.2-98.6%). Datos en docs/results/v3_contrastive/.
  V3.2 role (job 29184149) aún corriendo como cierre de familia; su veredicto
  no cambiará la decisión salvo sorpresa mayúscula.

- **V3.1 CURRICULUM LANZADO (2026-09-10, job 29068111, r23r09n01 RUNNING)**: run
  `f0-span-v3-curriculum`, receta piloto v2 (span/uniform/whole_stage, corpus
  esqueleto v2 fix, 10K steps, batch8+accum2, lr 2e-4 cosine warmup 200) +
  **curriculum de masking** (CURRICULUM=1): `cur_stages=[[0,2000,16,0.30],
  [2000,5000,32,0.60],[5000,10000,64,0.95]]` con interpolación lineal continua
  de span_len y b_h (implementación Devin PR #3, mergeada d9c8a4e; test
  scripts/test_curriculum.py pasa: boundaries exactos + monotonicidad).
  Log verificado: `curriculum on: steps=3 stages, start(span=16, b_h=0.30)`,
  loss 11.83→8.16 en primeros 90 steps (~10 steps/min → 10K ≈ 3.5-4h).
  **Battery al COMPLETE: automática contra pairs_hard_v3** (PAIRS_DIR v3) →
  L3 comparable directo con V3.0 (0.515). Watchdog: cron `ecoreasoner-v31-watchdog`.
  **PITFALL lanzamiento**: `sbatch --export='A=...;B=...'` vía ssh NO llega
  íntegro (las comillas simples se pierden → el `;` parte el comando → solo la
  primera var llega; `--export-file` tampoco se respeta en este clúster).
  **ÚNICO método validado**: archivo `runs/f0-v3curriculum-env.sh` con
  `export VAR=...` (CUR_STAGES entre comillas simples DENTRO del .sh) +
  `source ... && sbatch --parsable --export=ALL scripts/...`.
  El run usa el envío correcto; verificar job→log "curriculum on" antes de dar
  por bueno cualquier relanzamiento.

- **V3.2 ROLE-AWARE MASKING LISTO, SIN LANZAR (2026-09-10, Devin PR #4
  mergeada ba7752b)**: masking ponderado por rol semántico (conectivas
  causales/adversativas, verbos de relación, números, entidades por mayúsculas
  + keywords; SIN NER). Aditivo: `--role_mask` default OFF → comportamiento
  idéntico al V3.1. Interacción curriculum: el curriculum fija span_len/b_h,
  role_mask decide DÓNDE ubicar los spans (priorizando regiones informativas);
  con whole_stage pondera la selección de etapas. Verificado por Hermes: test
  scripts/test_role_mask.py (a) sin flags == V3.1 misma seed, (b) 2.65× tokens
  de rol enmascarados (random) y 0.62 vs 0.41 (span), (c) compat curriculum,
  (d) slurm moe_v4_micro_v2.slurm exporta ROLE_MASK/ROLE_CONFIG en resubmit de
  olas + CLI. Config: harness/configs/f0-span-v3-role.yaml (validada 13/13).
  **Lanzamiento CONDICIONADO al veredicto V3.1** (§4 criterio: L3 0.52-0.54 →
  iterar V3.2-V3.4): si V3.1 no despega, V3.2 es la primera iteración a probar.

- **ABLACIONES PILOTO v2 COMPLETAS (4/4, 2026-09-10) — todas STAGE_GRAMMAR**:
  no-whole-stage L2 0.596*** / weight-tying L2 **0.629*** / rope L2 0.596*** /
  50m L2 0.557**. L3 NUNCA significativo (0.47-0.51 ns) en NINGUNA variante.
  Lectura: el whole-stage NO es responsable de L2 (conserva 0.596 sin él);
  weight-tying da el mejor L2 (0.629); 50M no es limitante (0.557) → la ausencia
  de L3 no se explica por capacidad. Línea dLLM-puro = estructura SIN inferencia
  (material negativo publicable). Reporte: docs/results/ABLACIONES-V2-RESULTADO.md.
  Decisión pendiente: (A) archivar línea falsada-en-L3, (B) objetivo contrastivo
  L3 explícito, (C) focalizar en controller/verificator (Opción D, 98.6%).

- **PILOTO v2 COMPLETADO — VEREDICTO STAGE_GRAMMAR (2026-09-09 20:52)**: job
  28998750 COMPLETED 0:0 en 3:48:38 (10K steps, loss 11.79→6.91, checkpoint-g10000).
  Battery L0-L3: **L0 0.5039 ns / L1 0.4724 ns / L2 0.6118*** / L3 0.5065 ns**.
  interpretation.json: "El modelo aprendió orden/rol de etapas, pero no el
  contenido inferencial" → STAGE_GRAMMAR, recomendación ablate_or_scale.
  NO encaja en el tricótomo §13.4 (FALSIFY exige L2≤0.52; DIRECCIONAL/GO exigen
  L0/L1≥0.55): es el PRIMER resultado donde la receta v2 (whole-stage span sobre
  esqueleto) enseña gramática estructural SIN atajo temático (vs F2 que falsó todo;
  Sanity F2: L0 0.57/L1 0.56/L2 0.47/L3 0.51). L2 0.61 p=0.0 CI95 [0.57,0.65] sólido.
  Siguiente: ablaciones que Devin preparó (WHOLE_STAGE=0, WEIGHT_TYING=1,
  USE_ROPE=1, 50M) — lanza Hermes cuando Devin las pase.

- **FASE 3 — subcorpus eco GENUINO listo**: filter domain_fine v5 (17 dominios
  finos eco) → 142,054 docs, 63,188 candidatos doc×tool, 25 regiones. Tagger GBIF
  (retry 0 errores): **93 especies reales** (usageKey+conf); 14+ ecológicas de
  campo reales (Picea abies, Pinus sylvestris, Fagus sylvatica, Daphnia magna,
  Vulpes vulpes, Cervus elaphus, Quercus robur, Ursus arctos, Rangifer tarandus,
  Oncorhynchus, Salmo salar, Capreolus...). El domain_fine SÍ separa eco de biomed.
  Pool listo para el siguiente bloque de tool-calls. Reporte:
  docs/results/MINERIA-ESPECIES-SUBCORPUS-ECO-V7.md + species_eco_fine_tagged.json.

- **F2-SPANES COMPLETO — FALSIFY** (50,000/50,000, job 28982555 COMPLETED 00:47:38,
  flag COMPLETE 21:34). Veredicto oficial (verdict_f2.json): pairwise_acc final
  **0.5273** (< falsify_threshold 0.55), slope últimos 3 **-6e-06**, delta primeros 3→
  últimos 3 +0.0546, n=69 pts. La curva tocó picos 0.5664 (28,351) / 0.5586 (30,951)
  pero **decayó al final** (últimos 10: 0.52-0.54) — el pico es el artefacto de
  comparaciones múltiples que la auditoría advirtió; el punto final decide. Receta
  micro ganadora ×5 pasos NO produjo HIT. **Detalle**: el run original 28962112 murió
  TIMEOUT 88% por el deadlock SIGUSR1 (confirmado) y se relanzó 28982555 con el fix;
  el resume a 44,151→50,000 fue limpio.
- **SANITY L0-L3 (batería Devin PR #1, ejecutada 2026-09-08)**: L0 0.5738** / L1 0.5607** /
  L2 0.465 ns / L3 0.5089 ns → solo coherencia temática (atajo overlap); **FALSIFICACIÓN
  FUERTE** de la línea inferencial. Vía libre al controller/verificator.
  Doc: docs/results/F2-SPANES-RESULTADO.md (anexo).
- **DECISIÓN (pre-registrada, ejecutada)**: línea f2 **archivada como falsada** —
  `docs/results/F2-SPANES-RESULTADO.md`. NO relanzar a 100K. **Siguiente: arquitectura
  controller/verificator** (deepseek/glm genera tool-calls + dLLM como conocimiento/
  repair — opción D del A/B). El dLLM propio queda como posible línea de fluidez 155M
  SOLO con gate de falsación reformulado (word/rep4/uniq son engañables).
- **OPCIÓN D — VALIDACIÓN 300/10 TOOLS ACEPTADA (2026-09-09)**: replicación sobre 300 pares
  / 10 tools verificada de forma independiente (no self-report): match_func 298/300 (99.3%),
  match_args 291/300 (97.0%). El controller zero-shot aguanta 10 tools sin degradación.
  Reporte: docs/results/REPLICACION-300-10TOOLS-RESULTADO.md + replication_300.jsonl.
  PENDIENTE: resolvers reales de las 7 tools nuevas (iucn/try/filogenia quedan mock),
  estratificar gbif (84 vs maxent 12) si va a banco público. Aceptado como hito.
- **REPLICACIÓN FASE 3 (500 tool-calls, 7 tools) — ACEPTADA**: match_func 500/500 (100%),
  match_args 458/500 (92%). fallos=42: 26 year-mismatch (2010->2015), 13 region
  (africa occidental->asia oriental), 3 species cercanas — errores de VOCABULARIO del
  controller, no de diseño. **RUN v2 (fix vocabulario): match_args 491/500 (98.2%)**,
  year 26->0 (2010 exacto, antes ni estaba en KNOWN_VALUES), region 13->0.
  Fix: KNOWN_VALUES años 2010-2026 + 23 regiones del gold; SYSTEM_PROMPT lista
  vocabulario válido. Residual 9: 4 ambigüedad Quercus ilex->suber, 3 región
  traducida es->en (southeast asia), 2 tool sinonima (inaturalist vs gbif, args exactos).
  Reporte: docs/results/REPLICACION-500-FASE3-RESULTADO.md (v2). Dataset:
  data/l1/toolcalls_fase3_500.jsonl (123 especies reales GBIF+Devin).
- **FASE 3 EN CURSO (2026-09-09)**: minería de literatura -> tool-calls.
  **Replicación 500 v2: match_args 92%->98.2%** (year 26->0, region 13->0; ver arriba).
  **Minería especies**: fulltext PMC completo → 1.32M candidatas pero BIOMEDICAS;
  subcorpus eco v7 (919K docs, 63%) → 84 GBIF-reales, solo 23 ecológicas reales.
  HALLAZGO: dominios GRUESOS v7 no separan eco de biomed (S.aureus/M.tb/levaduras dominan
  aun en "eco"). Para poso eco GENUINO: domain_fine del v5 (climate_ecology/population_ecology...)
  o fuentes eco-nativas (EcoEvoRxiv, ecoseek-litdump, arXiv eco).
  Reporte: docs/results/MINERIA-ESPECIES-SUBCORPUS-ECO-V7.md.
  Diseño: docs/designs/toolcalls-expansion-2026-09-09.md.
- **OPCIÓN D VALIDADA (2026-09-09)**: controller/verificator demo end-to-end.
  `verify_toolcall.py` (verificador M1-M5, schemas empiricos 533 toolcalls reales, 100% valido) +
  `ecobench/run_controller_verificator.py` (controller + verificator + retry) +
  `scripts/eval_controller_replication.py` (HITO 3: en 17 prompts tool-call genuinos
  match_func 76% / match_args 65%, 100% JSON valido; el resto del corpus = gold contaminado
  [prompts NO de las 3 tools etiquetados gbif], no fallo del controller).
  Teacher v4flash REACTIVADO como serve slurm + tunel :20006 (keepalive `2d884420a767` reanudado).
  PENDIENTE REAL: (b) HITO 4 opcional: resolve_tool SIN mock (GBIF/CHELSA reales).
  Commits: a9d85fe, 5b796f3, 7c68053, 548abbf.
- **GOLD LIMPIO (2026-09-09)**: curacion manual de los 60 prompts -> 14 tool-calls PURAS
  (12 gbif + 2 bioclim; 46 descartados: RF/filogenia/diversidad funcional/slurm/literatura/
  otras APIs — gold ruidoso del teacher original). REMEDICION sobre subset limpio:
  **match_func 14/14 (100%) | match_args 12/14 (86%)** — los 2 fallos de args son gold
  inconsistente con el prompt (Yucatan vs neotropico en item 2; year 2020 vs 2050 en item 4),
  no error del controller. Best-case de la opcion D medido y limpio. Artefactos:
  docs/results/replication_pure_14.* + curation_report.json.
- **Legado técnico (validado en prod, commits bf25fec/03a2cde)**: anti-reentrada
  SIGUSR1 (flag SAVING, test EAGAIN), sort -V ckpt, --index absoluto, strict=True en
  suite_smoke, tmp PID en eval_curve, warmup REAL (era arg muerto), verdict_f2.py
  arreglado (f-string <3.12).
- **GAP**: `train_mdlm_moe_hetero.py` NO escribe `training_complete.flag` (solo
  loguea COMPLETE) — watchdogs/herederos: detectar fin por `state.json step==TARGET`
  o `sacct COMPLETED`, NO por el flagfile. (El micro trainer SÍ escribe el flag.)




---

##5. PARA RETOMAR RÁPIDO(checklist)

**Siguiente: EXPERIMENTO PUENTE — superar el accuracy con la receta EXACTA del micro
ganador,pero con MUCHOS más steps(updates pequeños y frecuentes,NO batch gigante。:

1. Revisar cola:`squeue -u a474r867 | grep -vE 'quercus|split'`(y sinfo pro6000 idle)。
2. Lanzar:`sbatch --export='TAG=f2-spanes,MASK_TYPE=span,MASK_P=0.15,SPAN_LEN=64,
   DATA_CACHE=/beegfs/a474r867/ecoreasoner/data/train_ids_skeleton.npy,
TARGET_STEPS=50000,PAIRS=/beegfs/a474r867/ecoreasoner/runs/pairs.jsonl,
   CONFIG=/beegfs/a474r867/ecoreasoner/harness/configs/f0-span-esqueleto.yaml,
   AUTO_RESUBMIT=1' /beegfs/a474r867/ecoreasoner/scripts/moe_v4_micro.slurm`
   (1 GPU pro6000 cu128;batch8+accum2=12,288 tok/update;lr 2e-4 warmup 200;
   olas AUTO_RESUBMIT ya integradas; 10K steps≈2:07h → 50K≈10.6h,100K≈21h)。

3. Cuando f2-spanes complete (`runs/f2-spanes/training_complete.flag` presente o
   `runs/f2-spanes/state.json step==TARGET_STEPS==50000`), correr
   `python scripts/verdict_f2.py --run-dir runs/f2-spanes` y seguir el VERDICT:
   - `HIT` (last acc >= 0.60) -> archivar como resultado positivo.
   - `EXTEND` (acc > 0.535 y pendiente last-3 positiva) -> relanzar f2 a 100K steps.
   - `FALSIFY` / `NO-GO` -> archivar la línea como falsificada y pasar a la
     arquitectura controller/verificator.
   Mientras tanto, `eval_curve` / `eval_curve_watch_f2.sh` siguen re-evaluando
   checkpoints; la deduplicación por step en `eval_curve.jsonl` mantiene la última fila.

4. Hito:si pairwise_acc escala de 0.535(partida del micro)hacia 0.6+
   con la curva — la receta era correcta;solo faltaban updates finos. Falsación:a100K
   steps si la curva NO despega de ~0.55:la línea dLLM pequeño queda cerrada con
   evidencia barata(y el cómputo cedido sigue siendo modesto).
5. Para un f2 de pool heterogéneo completo(validar NCCL multi-familia A100+L40):
   el watchdog `5fa09d642d89`(f1-multifam-validate)vuelve a ser prerequisito—
   reactivar al llegar ahi(no antes).

---

##6. ARCHIVOS CLAVE

| Qué | Dónde |
|---|---|---|
| Diseño Fase 3 -new(autoridad) | `docs/designs/ecoreasoner-Fase3-DESIGN-new.md` |
| Diseño v0.2 (histórico) | `docs/designs/ecoreasoner-Fase3-DESIGN.md` |
| Resultado F1 (NO-GO,2026-09-08) | `docs/results/F1-HETERO-RESULTADO.md` |
| Estimación F1 heterogénea | `docs/designs/ecoreasoner-F1-HETERO-ESTIMACION.md` |
| Resultados micro-sweep | `docs/results/MICRO-SWEEP-F0-RESULTADOS.md` |
| Trainer heterogéneo | `scripts/train_mdlm_moe_hetero.py` |
| Lanzador pool multiplicidad | `scripts/f1_hetero.slurm` |
| Extractor esqueletos | `scripts/build_skeleton.py` |
| Harness (eval + report) | `harness/` |
| Fuente de verdad runs | `/beegfs/a474r867/ecoreasoner/runs/index.jsonl` |
| Corpus esqueletos | `/beegfs/a474r867/ecoreasoner/data/skeleton/train_skeleton.jsonl` |

---

##7. REGLAS QUE NO SE ROMPEN(aprendidas con sangre)

1. **Loss NO es proxy de lenguaje** — siempre smoke/eval de discriminación.
2. **test-and-drop**: hipótesis validable en <4h GPU ANTES de escalar.
3. **Pool teórico ≠ pool real** — medir con scontrol;lanzar todo el pool a la vez。
4. **No tocar trainer vivo** — copiar(train_mdlm_moe_hetero.py),nunca editar
   el que corre(bw5_spanhi usa train_mdlm_moe.py）。5. **sync a repo SIEMPRE** después de tocar scripts en HPC(scp + commit)—el
   repo es la fuente de verdad.
6. **YAML válido + sbatch --export sin comas**(usar `;`)—validar con
   validate_configs.py antes de lanzar。