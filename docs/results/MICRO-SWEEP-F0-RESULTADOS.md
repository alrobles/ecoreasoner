# Micro-sweep F0 — RESULTADOS (2026-09-07)

6/6 runs completados (10K steps, micro dense 50-100M, 1 GPU pro6000, seed 7331)
y evaluados con suite_smoke (discriminación inferencial pairwise, 256 pares
del corpus de esqueletos, runs/pairs.jsonl).

## Resultados

| Run | mask | datos | loss@10K | pairwise_acc | mean_delta | generation uniq |
|---|---|---|---|---|---|---|
| f0-random-prosa | random 15% | v7_clean | 3.28 | 0.5117 | -0.032 | 0.664 |
| f0-span-prosa | span 64 @15% | v7_clean | 6.37 | 0.4492 | -0.064 | 0.578 |
| f0-spanhi-prosa | span 64 @60% | v7_clean | 5.93 | 0.4766 | -0.052 | 0.570 |
| f0-random-esqueleto | random 15% | skeleton | — | 0.5078 | +0.011 | 0.688 |
| f0-span-esqueleto | span 64 @15% | skeleton | — | **0.5352** | +0.015 | 0.609 |
| f0-spanhi-esqueleto | span 64 @60% | skeleton | — | 0.5312 | +0.030 | 0.734 |

## Veredicto vs criterio pre-registrado

**Criterio GO: pairwise_acc ≥ 0.55 (un 55% discriminación inferencial).**
**NINGUNO lo alcanza.** El mejor (f0-span-esqueleto, 0.535) queda a 1.5 p.p.

Pero el patrón es el esperado por la tesis:

1. **Datos**: esqueleto >> prosa. Prosa: acc 0.45-0.51 con mean_delta **negativo**
   (el modelo sistemáticamente prefiere la continuación INCORRECTA — la
   "sopa de subwords" de bw4_span explica: no hay rastro de estructura que
   guíe). Esqueleto: acc 0.51-0.54 con mean_delta **positivo** (dirección
   correcta). La estructura de argumento mueve la aguja de discriminación.
2. **Objetivo de masking**: en esqueleto, span > random (0.535, 0.531 vs
   0.508). En prosa, random > span (0.51 vs 0.45-0.48) — el span descubre
   mejor estructura cuando HAY estructura que reconstruir.
3. **Token budget**: 10K steps × (2 grad_accum × 8 batch × 768 seq) ≈ **245K
   tok vistos ≈ 0.006% del corpus**. Es el equivalente de bw0/bw1v5 (37M tok,
   3%) — severamente subentrenado. La tesis no se falsa con esto: se apoya
   (la dirección es la correcta, el criterio era "transitorio" y con 3-4
   órdenes de magnitud más de tokens el span-esqueleto tiene el mejor camino).

## Conclusión para F1

**Ganador: span-esqueleto** (span 64 @15%, corpus de esqueletos). El camino
de F1 = escalar "entrenar el dLLM desde cero sobre esqueletos con span
masking" a muchísimos más pasos (criterio: discriminar ≥55% + completar
esqueletos), en lugar de intentar prosa cruda. La estimación heterogénea
(ecoreasoner-F1-HETERO-ESTIMACION.md) aplica: con el pool real (4× L40
ahora mismo) son ~4.9 días por epoch; con pool completo ~4-10h.

## Archivos

- `runs/index.jsonl` — índice de los 6 micro-runs (fuente de verdad).
- `runs/<tag>/report.json` — reporte completo (config sha, seed, suite).
- `harness/configs/f0-*.yaml` — configs (el ganador = span-esqueleto).
- `runs/pairs.jsonl` — 256 pares de discriminación (reutilizables para F1).

## Pitfalls golpeados en este camino (2026-09-07)

- **YAML flow-mapping inválido** (`train: {mask_type: random ...}` sin
  comas) mata la eval de suite_smoke/report SILENCIOSAMENTE (el entrenamiento
  no lo lee — solo la eval). validate_configs.py lo previene.
- **sbatch --export=TAGS=a,b,c**: slurm parte por comas → solo llega la
  primera variable. Usar punto y coma (`;`) y separar en el script con
  `IFS=';'`.
- **suite_smoke.generate()** creaba `torch.tensor(...)` en CPU → crash
  cuda:0/cpu al llamar al modelo en GPU. Fix: `next(model.parameters()).device`.
- **VENV duplicado** en slurm (`/bb/bwvenv/bin/python/bin/python`): definir
  `PY=$VENV/bin/python` UNA vez.
- **build_pairs.py** asumía `etapas` como dict, pero el FULL serializa el
  texto con etiquetas y `etapas` es int (nº de etapas). Parsear el texto
  con regex de etiquetas.
- report.py con `--index runs/index.jsonl` relativo al cwd del job ≠ base
  del proyecto: usar ruta ABSOLUTA siempre.