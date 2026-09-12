# LOGICDIFF FASE B1 — candidate_focus: RESULTADO NO-GO (el objetivo no era el cuello)

> Run: `f0-span-v3-candfocus` (checkpoint-g10000, 10K steps, 3.5h) · battery automática 11:43
> Criterio pre-registrado (doc Fase B): **GO si L3-dense ≥ 0.66** o Δ≥+0.02 vs el
> mejor base (contrastive L3-dense = 0.640). Evaluación: `pairs_hard_v3_eval` (n=475),
> mecanismo canónico **dense** (candidato 100% enmascarado) + base (random 15%).

## Verdicts (battery_verdict_base.json / battery_verdict_dense.json, seed 7331)

| Nivel | base acc (p) | dense acc (p) |
|---|---|---|
| L0 | 0.5207 ns (0.19) | 0.5444 * (0.025) |
| L1 | 0.5118 ns (0.31) | 0.5295 ns (0.10) |
| L2 | 0.5819 *** (1.3e-4) | **0.6627 *** (0.0)** |
| **L3** | 0.5179 ns (0.23) | **0.5832 *** (1.7e-4)** |

Reading base: "ESTRUCTURA SIN INFERENCIA". Reading dense: "SEÑAL INFERENCIAL:
discrimina el payload mutado → la línea no está falsada; la métrica L0 era
demasiado fácil."

## Comparación con los controles de la Fase A

| Run | L3 dense | lectura |
|---|---|---|
| random-init | 0.459 ns | piso limpio |
| v3-contrastive (V3.3-fixed) | **0.640 *** | mejor base de la familia |
| v3-role | 0.592 *** | control |
| **B1 candfocus** | **0.5832 *** | por DEBAJO de ambos controles |

## Desglose por subtipo (L3, dense) — el patrón NO cambió

| subtipo | n | acc | vs Fase A (direction 0.76-0.78; number/negation azar o invertidos) |
|---|---|---|
| direction_word | 193 | 0.741 | sigue fuerte (el grueso de la señal) |
| causal_phrase | 44 | 0.705 | |
| mechanism_word | 9 | 0.667 | |
| environment_word | 11 | 0.636 | |
| temporal_phrase | 33 | 0.606 | |
| causal_word | 66 | 0.515 | azar |
| temporal_word | 23 | 0.478 | azar |
| **number** | 52 | **0.308** | PEOR que azar (Fase A ya daba azar/invertido) |
| **negation** | 43 | **0.186** | INVERTIDO (significativamente peor que azar) |

El candidate-focus **no repara** los subtipos donde la lógica fina importa; si
algo, los DEGRADA (number 0.31, negation 0.19 vs el azar ≈ 0.5). El acierto se
concentra en direction_word / causal_phrase — la misma firma direccional de
siempre, sin profundizar.

## Lectura crítica

1. **B1 = NO-GO según criterio**: 0.5832 < 0.66 y < 0.640 del mejor base. El
   objetivo candidate-focused (practicar exactamente la tarea del scorer denso
   en training) NO supera al contraste incidental de V3.3 ni al role-mask.
2. **El objetivo no era el cuello** → confirma la hipótesis Fase B: el techo
   está en el INPUT (sin etapa inferencial intermedia ni contrastes válidos).
   Siguiente en la escalera: **B2** (rellenar HIPOTESIS+PREDICCION con teacher
   DeepSeek-V4-Flash — en vuelo) y **B3** (sintético FLD, números/negación
   controlados).
3. La señal L3-dense se mantiene significativa en TODOS los runs entrenados
   (8/8 en Fase A + B1) → el dLLM como **scorer direccional** sigue siendo la
   salida defensible si B2/B3 fallan (criterio de cierre: nada supera 0.66 ni
   repara number/negation → dLLM = scorer de rerank en controller/verificator).

## Artefactos

- `/beegfs/a474r867/ecoreasoner/runs/f0-span-v3-candfocus/battery_logicdiff/`
  (logicdiff_summary.json con subtipos; base/ y dense/)
- `battery_verdict_base.json` / `battery_verdict_dense.json` en el run dir.