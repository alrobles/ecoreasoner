# Resultados V3.3 — Fine-tuning contrastivo L3

> Job final: 29184167 (relanzado tras fix de contaminación train/eval).
> Checkpoint: `runs/f0-span-v3-contrastive/checkpoint-g-final/model.pt`.
> Pares de evaluación hold-out: `runs/pairs_hard_v3_eval` (seed 1337, generados
> desde `data/skeleton/train_skeleton_val.jsonl`).

## 1. Entrenamiento

- Steps: 3000.
- Pares: 466 (train 373, eval 93, split por contexto).
- `train_acc`: 1.000.
- `eval_acc` (holdout interno del trainer): **0.7742**.
- `mlm` final: ~7.3.
- El entrenamiento convergió rápido y el modelo separa correctamente los
  pares en el ranking loss del trainer.

## 2. Batería final L0-L3 (pares hold-out, `suite_smoke_v2`)

| Nivel | n_pairs | pairwise_acc | IC95 | p | sig |
|---|---|---:|---|---:|---|
| L0 | 507 | 0.4931 | [0.450, 0.536] | 0.63880 | ns |
| L1 | 508 | 0.5335 | [0.490, 0.576] | 0.07154 | ns |
| L2 | 507 | **0.6114** | [0.568, 0.653] | 0.00000 | *** |
| L3 | 475 | **0.4863** | [0.442, 0.531] | 0.73966 | ns |
| **Promedio** | 1997 | 0.5311 | - | - | - |

## 3. Veredicto

> **ESTRUCTURA SIN INFERENCIA**: el modelo distingue el orden de etapas
> (L2 = 0.61, significativo) pero no el contenido inferencial (L3 = 0.49,
> no significativo).

Esto es el mismo patrón `STAGE_GRAMMAR` observado en los 7/7 runs MLM/masking
anteriores, ahora reproducido bajo un objetivo contrastivo/ranking.

## 4. Implicaciones

- El fine-tuning contrastivo sobre pares hard-negative **no produjo inferential
  discrimination** cuando se evalúa con una métrica de denoising (no ranking)
  sobre pares hold-out.
- El `eval_acc = 0.774` del trainer es una señal engañosa: la métrica de
  ranking con el mismo sufijo enmascarado aprende a separar el par, pero no
  generaliza a discriminación por denoising sobre secuencias completas.
- El problema no es sólo contaminación (ahora controlada) ni falta de convergencia.
  La señal de entrenamiento sigue siendo insuficiente para razonamiento real.

## 5. Decisión según el árbol de diseño

Según `docs/designs/dLLM-reasoning-from-scratch.md`:

- L3 = 0.486 < 0.52 → **NO-GO**.
- L2 = 0.611 ≥ 0.55 (preservado).
- `rank_acc` del trainer no es evidencia de L3.

Conclusión recomendada:

1. **Archivar la línea pura dLLM** como resultado negativo publicable.
2. **Pivotar al controlador/verificador** como MVP práctico.
3. Opcionalmente, documentar que el ranking L3 no es equivalente a
   discriminación por denoising.

## 6. Nota sobre el cierre del job

El job 29184167 falló al finalizar `suite_smoke_v2` porque el config no tenía
`eval.mask_p` en el momento del lanzamiento. Se corrigió el config y se lanzó la
batería manualmente (job 29184264). Los resultados son los definitivos.
