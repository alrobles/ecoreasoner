# F1-hetero — RESULTADO FINAL (2026-09-08): NO-GO tras 1000 steps

Run: `f1_hetero.slurm` · 5 jobs / 17 ranks (L40×4 + Q6000×13) · trainer
`train_mdlm_moe_hetero.py` (autosize + SCALE_I). Modelo dense 154.8M
(hidden 512, layers  ️8, vocab 126080, seq 768). Dataset `train_ids_skeleton.npy`
(129.5M tok); receta span64@15% (ganadora del micro-sweep), GLOBAL_BATCH
65536, lr 2e-4, warmup 100, TARGET_STEPS=1000.

## Estado final (verificado 2026-09-08 ~09:50)

- 5 jobs `f1-hetero` **COMPLETED** (Elapsed 03:02; arranque 06:43 → fin 09:45).
- `state.json`: `{"step": 1000, "checkpoint": "checkpoint-g1000", ...}`.

- **GAP heredero**: el trainer hetero SOLO loguea `COMPLETE` (`log("COMPLETE")` en
  linea 572) — **NO escribe `training_complete.flag`** (a diferencia del
  `train_mdlm_moe.py` clásico). Watchdogs/herederos del F1 deben detectar fin
  por `state.json step == TARGET` o por `squeue` vacío + `sacct COMPLETED`; NO por
  el flag.
- Último loss: `[09:44:27] step 990 loss 6.8983` — el loss se estancó en
  plateau ~7.1 desde ~step  ️100 (patrón conocido: loss no es proxy de lenguaje).

## Curva de discriminación (criterio GO: pairwise_acc ≥ 0.55)

| step | pairwise_acc | mean_delta |
|---|---|---|
| 501 | 0.4844 | -0.0436 |
| 551 |  ️0.4883 | -0.0405 |
| 701 |  ️0.5000 | -0.0300 |
| 751 |  ️0.4961 | -0.0266 |
| 801 |  ️0.4922 | -0.0220 |
| 851 |  ️0.5039 | -0.0148 |
| 901 |​ 0.5078 | -0.0108 |
| 951 |​ 0.4961 | -0.0129 |
| 1000 |​ 0.5000 | -0.0066 |

(Nota: la curva completa queda en `runs/f1-hetero/eval_curve.jsonl` — con dups
por el watch + eval final;dedupe quedarse con el último. Missing 601/651
(podados por retention-2 antes de arrancar el watch).)

## VEREDICTO: NO-GO

- pairwise_acc se mantiene en el **azar exacto (0.5)** durante los 1000 steps;
  nunca supera 0.508 (máximo 901=0.5078) y termina en  ️0.500.
- mean_delta mejora monotónicamente ( --0.044 → --0.0066): el modelo deja de
  preferir la continuación INCORRECTA, pero nunca llega a preferir la CORRECTA.

- El criterio pre-registrado (≥0.55) NO se alcanza ni de cerca. El F1 NO muestra
  discriminación inferencial igual que el micro-sweep a 10K steps de 1 GPU.

## CONTRASTE CLAVE: es la config de OPTIMIZACIÓN, no la escala

| | tokens/update | updates | tokens totales | acc |
|---|---|---|---|---|
| micro-span-esqueleto (ganador) | 12,288 (batch8×accum2×768) | 10,000 | 122.9M |  **0.5352** |
| F1-hetero (este run) | 65,536 |  ️️1,000 |​ 65.5M |​  ️️0.5000 |

Mismo corpus, mismo modelo dense 154.8M, mismo mask span64@15%, mismo lr
2e-4. La única diferencia real: el micro hizo ~10× más updates pequeños por token
(12.3K tok/update vs 65.5K);el F1 promedió cada gradiente sobre 65536 tok
(la señal estructural del esqueleto se diluye;y con solo  ️1K updates el modelo nunca
ve cada patrón fino suficientes veces. La conclusión NO es“más GPU”; es
**más optimizer updates pequeños (misma receta micro, muchos más steps.)**

## Plan puente recomendado (para superar el accuracy)

- Re-correr la **receta EXACTA del micro ganador** (span64@15%, esqueleto,
  batch_size 8, grad_accum 2, lr 2e-4, warmup 200} PERO con
  `TARGET_STEPS=50K-100K` (vs 10K;5-10× más),en 1-2× pro6000
  (el slurm `moe_v4_micro.slurm` ya tiene AUTO_RESUBMIT + olas).
- Costo estimado: 10K steps ≈ 2:07h en  ️1 GPU →  ️50K ≈​ 10-11h;  ️100K ≈​  ​ 21h (~1 día de 1 GPU, o ~12h en 2).
- Evaluación por tramos: `eval_curve` cada ~5-10K steps (mismo par de 256
  pares; la curva del micro partió de 0.535 → si escala a 0.6+ tenemos
  el hito: la receta era correcta, solo necesitaba suficientes updates finos.

- Si con 100K steps la curva NO despega de ~0.55: la línea dLLM pequeño queda
  falsada con evidencia barata y limpia (y el cómputo cedido sigue siendo modestoio..