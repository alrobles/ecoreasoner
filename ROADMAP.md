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

##4. ESTADO VIVO(HPC,al 08-09 09:50+ CDT)

- **F1-hetero COMPLETO — NO-GO** (1000/1000 steps,jobs COMPLETED 03:02;curva
  completa en `runs/f1-hetero/eval_curve.jsonl`;acc max 0.508 — veredicto + lección
  en docs/results/F1-HETERO-RESULTADO.md。


- **GAP**: `train_mdlm_moe_hetero.py` NO escribe `training_complete.flag`(solo
  loguea COMPLETE) — watchdogs/herederos: detectar fin por `state.json step==TARGET`
  o `sacct COMPLETED`,NO por el flagfile。


- **bw5_spanhi** COMPLETE(38,000 steps,flag COMPLETE;curva de ablación cerrada:
random/span/spanhi — veredicto: la ratio de mascara no era el ingrediente;losDatos
  estructurados sí. F0+SWEEP cerrado.Watchdog `ecoreasoner-f0-sweep-watchdog`
  (`d7d8067a09d5`) ya en no-op(F0 cerró;considerar pausar.


- **Watchdogs**: `5fa09d642d89` f1-multifam-validate — ya NO es prerequisito( el
  F1 completo corrió por la vía L40+Q6000 sin A100;verificar su estado;si sigue esperando
  A100, pausarlo(ya no hará falta. Otros watchdogs del swarm(bw0/bw1,
  ollama-governors) activos.



- **Pool**:17 ranks del F1 liberados;L40/Q6000 vuelven al pool; pro6000 idle(
  r23r09n01/r30r08n01/r30r24n01,5 GPUs cu128 libres);A100 status variable。




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

3. Al llegar cada ~5-10K steps(ckpt-gN):correr `eval_curve`(pro6000,--no-gen,min
   --min-step 500;~1 min/ckpt)o adaptar el watch `scripts/eval_curve_watch.sh`(detectar
   fin por state.json step==TARGET,NO por el flagfile)。

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