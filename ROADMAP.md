# ROADMAP — EcoReasoner Fase 3 (refundación `-new`): excavar un dLLM científico

> ReumanLab · EcoReasoner · 2026-09-07 (última actualización)
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

## 1. HITOS (medidos por decisión, no por calendario)

### F0 — Harness F0 + eval nueva de discriminación — ✅ COMPLETO (07-09)
- `harness/`: run_micro.py (slurm 1-GPU declarativo), report.py (report.json + index.jsonl),
  compare.py, suite_smoke.py (discriminación pairwise + completación + fluidez),
  validate_configs.py (previene YAML roto), build_pairs.py (pares reales del corpus).
- `scripts/moe_v4_micro.slurm` — micro-run 1 GPU pro6000 (10K steps, dense 50-100M,
  olas SIGUSR1, eval+report al completar).
- **Micro-sweep 6/6 completado y evaluado** → ver §2.

### F0b — Extractor de esqueletos + corpus — ✅ COMPLETO (07-09)
- `scripts/build_skeleton.py` (IMRaD + abstracts estructurados + frases-faro, 0 GPU)
  → `data/skeleton/train_skeleton.jsonl` (**315,299 docs** con ≥3 etapas, 675MB)
  → `data/train_ids_skeleton.npy` (129.5M tok, OOB_OK).
- `scripts/skel_full.slurm` + `skel_pairs.slurm` + `pre_tokenize_skeleton.slurm`.
- Cobertura REAL: 51.6% ge2, 21.6% ge3 (el 70% era del mini-corpus sintético;
  el real es heterogéneo y el activo de 315K docs basta).

### F1 — Entrenamiento objetivo con el pool heterogéneo — EN PROGRESO 🔶
- **Trainer heterogéneo VALIDADO**: `scripts/train_mdlm_moe_hetero.py`
  (`--autosize` por VRAM medido con fwd+bwd real + `SCALE_I` gradiente exacto por
  tokens + grad_accum global; smoke L40×4 world=4 micro=25 COMPLETE).
- **Lanzador multi-familia**: `scripts/f1_hetero.slurm` (torchrun rendezvous TCP,
  cada job = 1 familia; NCCL_IB_DISABLE para el muro IB/noib).
- **Pendiente**: validación multi-familia A100+L40 (watchdog `f1-multifam-validate`
  lanza cuando se liberen los A100) → luego F1 completo: receta ganador
  (span-esqueleto), TARGET_STEPS ~1 epoch, evaluar pairwise_acc cada checkpoint.
- Estimación: `docs/designs/ecoreasoner-F1-HETERO-ESTIMACION.md`.

### F2 — Agente (loop + tools encima del modelo) — [ ]
### Paper 1 (systems: entrenamiento heterogéneo + harness) — en paralelo
### Paper 2 (tesis desde-cero, positiva o negativa limpia) — [ ]

---

## 2. RESULTADO del micro-sweep F0 (fuente de verdad: runs/index.jsonl + docs/results/MICRO-SWEEP-F0-RESULTADOS.md)

| Run | mask × datos | pairwise_acc | mean_delta |
|---|---|---|---|
| f0-random-prosa | random 15% × v7 | 0.5117 | -0.032 |
| f0-span-prosa | span64×15% × v7 | 0.4492 | -0.064 |
| f0-spanhi-prosa | span64×60% × v7 | 0.4766 | -0.052 |
| f0-random-esqueleto | random × skeleton | 0.5078 | +0.011 |
| **f0-span-esqueleto** | span64×15% × skeleton | **0.5352** | +0.015 |
| f0-spanhi-esqueleto | span64×60% × skeleton | 0.5312 | +0.030 |

**Veredicto:** GO pre-registrado (≥0.55) NO alcanzado por ninguno (mejor 0.535,
mejor esqueleto 0.535/0.531 vs prosa 0.45-0.51 con delta NEGATIVO). **Tesis
reforzada en dirección**: estructura de argumento mueve la discriminación
(esqueleto > prosa; span > random en esqueleto); 10K steps = 245K tok = 0.006%
del corpus → subentrenado, el criterio era "transitorio". **Ganador: span-esqueleto.**

---

## 3. CÓMPUTO — pool NVIDIA real y tok/s medidos

| Familia | GPUs | tok/s (micro dense, solo entreno, 08-09) |
|---|---|---|
| **A100** (cu121) | ~18 | **~22,026** |
| pro6000 Blackwell | 5 | **~12,553** |
| L40 (cu121) | 4 | **~11,443** |
| Q6000 (cu121) | ~29 | **~7,941** |

- **PITFALL medición**: walltime del job incluye la carga del cache (~50s) →
  subestima 2-5×. Cronometrar del log (step 0 → step 190 × 6144 tok/step).
  1 epoch esqueleto (129.5M tok) ≈ 8 min pool realista / 15-22 min solo
  L40+Q6000 → **el F1 completo ya es viable con el pool de HOY** (ver §5).

- El pool real **fluctúa por minuto** (A100 se ocupan/liberan todo el tiempo).
  Sinfo miente: usar scontrol (AllocTRES vs Gres).
- pro6000 (cu128) NO se mezclan en DDP con cu121 (muro HITO 3) — familias
  homogéneas por partición + rendezvous TCP entre nodos.
- **Muro NCCL IB/noib documentado**: L40 r32r25n01 es noib, A100 son ib → el L40
  fallaba `NCCLUtils.hpp:275`. Fix: NCCL_IB_DISABLE=1 + NCCL_SOCKET_IFNAME.

---

## 4. ESTADO VIVO (HPC, al 07-09 22:5x CDT)

- **bw5_spanhi** (job 28826427→28858877, olas): step ~27,300/38,000, loss ~5.5,
  sin flag COMPLETE aún. Tercer punto de la curva de ablación (span hi 60%).
  NO bloquea F1 (usa pro6000 cu128; F1 usa cu121).
- **Watchdogs activos** (crons):
  - `d7d8067a09d5` ecoreasoner-f0-sweep-watchdog (cada 10m, agent) — vigila bw5 + f0-*/skel-*;
    conoce el veredicto del sweep, no lo repite.
  - `5fa09d642d89` f1-multifam-validate (cada 15m, no_agent bash) — espera A100
    libres y lanza la validación NCCL multi-familia solo; reporta VALIDADO/FAILED una vez.
  - Oleada de watchdogs viejos del swarm (bw0/bw1, ollama-governors) activos.
- **Pool ahora**: L40 4/4 libres, A100 ocupados (r31* 0/3), Q6000 1-2 sueltos,
  pro6000 1-2 libres (cu128).

---

## 5. PARA RETOMAR RÁPIDO (checklist)

1. `hermes cron history 5fa09d642d89` → ¿la validación multi-familia se lanzó/salió?
   - Si VALIDADO OK → lanzo F1 completo (span-esqueleto, pool cu121 completo).
   - Si FAILED → leer el tail y arreglar el NCCL antes de F1.
2. `cat outputs/bw5_spanhi/training_complete.flag` → ¿bw5 completó? (smoke de continuación
   pendiente: smoke_cont sobre bw5_spanhi, criterio rep4→0.15 Y word→0.435).
3. Comprobar cola: `squeue -u a474r867 | grep -vE 'quercus|split'`.
4. Si quiero F1 YA (sin esperar A100): con L40+Q6000 libres es ~15-22 min por
   epoch esqueleto. Lanzar `f1_hetero` multi-job (0=A100/L40, 1=Q6000...) con
   TARGET_STEPS = 1 epoch (129.5M tok / micro tokens), receta span-esqueleto.
   Medir la curva con `eval_curve` (eval_curve.py + eval_curve.slurm, hecho
   08-09: recorre checkpoints y evalúa pairwise_acc). A100 es opcional.
5. ~~Medir Q6000~~ ✅ HECHO (08-09): ~7,941 tok/s (smoke 28879250). Milestone
   del pool: A100 22K, pro6000 12.6K, L40 11.4K, Q6000 7.9K.

---

## 6. ARCHIVOS CLAVE

| Qué | Dónde |
|---|---|
| Diseño Fase 3 -new (autoridad) | `docs/designs/ecoreasoner-Fase3-DESIGN-new.md` |
| Diseño v0.2 (histórico) | `docs/designs/ecoreasoner-Fase3-DESIGN.md` |
| Estimación F1 heterogénea | `docs/designs/ecoreasoner-F1-HETERO-ESTIMACION.md` |
| Resultados micro-sweep | `docs/results/MICRO-SWEEP-F0-RESULTADOS.md` |
| Trainer heterogéneo | `scripts/train_mdlm_moe_hetero.py` |
| Lanzador pool multiplicidad | `scripts/f1_hetero.slurm` |
| Extractor esqueletos | `scripts/build_skeleton.py` |
| Harness (eval + report) | `harness/` |
| Fuente de verdad runs | `/beegfs/a474r867/ecoreasoner/runs/index.jsonl` |
| Corpus esqueletos | `/beegfs/a474r867/ecoreasoner/data/skeleton/train_skeleton.jsonl` |

---

## 7. REGLAS QUE NO SE ROMPEN (aprendidas con sangre)

1. **Loss NO es proxy de lenguaje** — siempre smoke/eval de discriminación.
2. **test-and-drop**: hipótesis validable en <4h GPU ANTES de escalar.
3. **Pool teórico ≠ pool real** — medir con scontrol; lanzar todo el pool a la vez.
4. **No tocar trainer vivo** — copiar (train_mdlm_moe_hetero.py), nunca editar
   el que corre (bw5_spanhi usa train_mdlm_moe.py).
5. **sync a repo SIEMPRE** después de tocar scripts en HPC (scp + commit) — el
   repo es la fuente de verdad.
6. **YAML válido + sbatch --export sin comas** (usar `;`) — validar con
   validate_configs.py antes de lanzar.
7. **Checkpoints atómicos + resume tolerante** — ya integrados, no revertir.