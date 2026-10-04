# EcoReasoner — Reporte de avance dLLM (2026-10-04)

Estado: **activo**. Mainline `v5-1b` en **~g750K/1.3M (~57.7%)**,
watcher y cadena AUTO_RESUBMIT sanos. Brazo block-diffusion
**completado (30K/30K) y evaluado: FALSADO** (§2). Continuación del
PROGRESS-REPORT-2026-10-03.

---

## 1. Curva mainline — dos puntos nuevos (EMA)

| ckpt | denseL3 | cons_inf | gx_ok inf | ctxnec | gx_ok ctx | C4 drop ctx | remask gx ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| g656001 | 0.518 | 0.655 | 0.329 | 0.739 | 0.289 | −8.3 | 0.335 |
| g683001 | 0.512 | 0.659 | 0.329 | 0.737 | 0.277 | −8.4 | 0.328 |
| g710001 | 0.517 | 0.668 | 0.337 | 0.743 | 0.290 | −7.8 | 0.335 |
| g738065 | 0.517 | 0.665 | 0.334 | 0.738 | **0.293** | −8.7 | 0.335 |

**Lecturas:**

1. **`ctxnec`: meseta confirmada** — 5 puntos consecutivos en/bajo el
   máximo (0.743): banda 0.728-0.743 durante los últimos ~170K steps.
2. **`cons_inf` se reafirma en ~0.665** (0.668/0.665 tras el vaivén
   0.655/0.659) — el nivel nuevo ~0.66-0.67 vs meseta vieja ~0.655.
3. **`gen_exact` ctxnec es la única pendiente viva**: 0.290 → 0.293 =
   nuevo máximo de serie. Crece lento pero sigue.
4. Subtipos sin sorpresas: causal ~0.59-0.61 cola persistente;
   `role_swap` 0.891 (máximo reciente), `entity_rebind` 0.82,
   `direction` 0.68, `negation` 0.68.

## 2. Decisión blkdiff (E-arq) — FALSADO

Eval de decisión a dosis igualada (init EMA g683001 + 30K; referencia
mainline g710K ≈ g683K + 30K):

| métrica | init g683K | mainline g710K | blkdiff g30K | Δ vs mainline |
|---|---:|---:|---:|---:|
| cons_inf pairwise | 0.659 | 0.668 | 0.643 | **−2.5pt (~2σ)** |
| gx_ok inf | 0.329 | 0.337 | 0.286 | −5.0pt |
| ctxnec pairwise | 0.737 | 0.743 | 0.696 | **−4.7pt (~3.5σ)** |
| gx_ok ctxnec | 0.277 | 0.290 | 0.242 | −4.8pt |
| C4 drop | −8.4 | −7.8 | **−10.1** | +2.3pt más profundo |
| remask gx ctx | 0.328 | 0.335 | 0.270 | −6.5pt |
| dense L3 | 0.512 | 0.517 | 0.530 | ≈0 |

**Veredicto: falsado, con un matiz documentado.**

- Por la letra del gate pre-registrado, el criterio "C4 drop ≥2pt más
  profundo" se cumple (−10.1 = récord de la serie). **En sustancia es
  negativo**: las tres métricas primarias no solo no superan al
  mainline — **retroceden vs el propio init** del brazo (ctxnec
  0.737→0.696; remask 0.328→0.270; cons_inf 0.659→0.643).
- Lectura honesta del C4 profundo: no es más binding, es **fragilidad**
  — el modelo quedó más dependiente del contexto limpio y peor en
  usarlo; la corrupción pega más sobre un baseline degradado.
- Subtipos del brazo (ctxnec): role_swap 0.839, entity_rebind 0.772,
  value_rebind 0.618 — todos bajo el mainline (0.89/0.82/0.66).
- La hipótesis "la asimetría contexto→target cableada en la máscara
  de atención mejora el binding" queda **falsada a esta escala/dosis**
  (30K, warm-start maduro, 3×A100).

## 3. Estado del programa tras cerrar el eje arquitectura

Los dos brazos estructurales del programa salieron negativos a dosis
igualada: **fgmode** (fine-grained MoE, FLAT) y **blkdiff** (BD3-LM
semi-AR, regresión). Conclusión consolidada: el cuello de binding del
dLLM no está en la estructura de atención ni en la especialización de
expertos — es consistente con la lectura de capacidad/datos (la única
palanca que sigue moviendo algo es la exposición, vía `gen_exact`, y
el prior de plausibilidad ya en meseta).

Negativos controlados de ambos brazos = material directo para el
paper 1 (P8): protocolo de comparación a dosis igualada + batería de
instrumentos corregidos + falsación pre-registrada.

## 4. Qué queda del plan

| # | Punto | Estado | Próxima acción |
|---|---|---|---|
| P1 | Decisión fgmode | CERRADO: falsado | — |
| P2 | Exposición | en curso, ~58% | la curva sigue; ctxnec en meseta, gx la única pendiente viva |
| P3 | sciev backbone→decisión | en ventana | re-eval sobre ckpt reciente (~g740-780K) para 3er punto de trayectoria |
| P5 | Chat-SFT v2 | listo, no lanzado | decisión pendiente con Angel — backbone ya maduro, el plan decía ≥g700K (ya superado) |
| P6 | Corpus UNAM-EN | parcial | decisión de datos pendiente |
| P7 | block-diffusion | **CERRADO: falsado** | eje arquitectura terminado (ambos brazos negativos) |
| P8 | Paper 1 (systems+harness) | backlog | material listo: crisis de instrumento + 2 falsaciones pre-registradas |
| P9 | Paper 2 (tesis) | backlog | depende del desenlace exposición→1.3M |

## 5. Riesgos / vigilancia

- **Sixhour churn**: sano; mainline job 30928463 en r32r25n01, ETA al
  target ~5-6 días al ritmo actual (~70-90K/día).
- **Watcher**: vivo desde el relanzamiento (setsid); generó g656K,
  g683K, g710K, g738K sin huecos. Mitigación pendiente: watchdog/cron.
- **`entity` n=1**: auditoría barata sigue pendiente.

## 6. Fuente de datos

- Brazo: `/beegfs/a474r867/ecoreasoner/runs/v5-1b-blkdiff/`
  (battery_curve/g30000 completo: dense+cons_inf+cons_ctxnec+remask,
  EMA; train.log, ckpts).
- Curva mainline: `runs/v5-1b/battery_curve/g710001`, `g738065` (EMA).
- Jobs eval blkdiff: 30928706-308.
