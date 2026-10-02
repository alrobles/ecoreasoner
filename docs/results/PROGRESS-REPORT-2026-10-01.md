# EcoReasoner — Reporte de avance dLLM (2026-10-01)

Estado: **activo**. Pretrain `v5-1b` en **g585K/1.3M** (~45%), ~85K
steps en las últimas ~27h (sixhour, isla actual r32r05n01). Watcher
vivo, baterías cada ~25K steps aterrizando al día. Continuación del
PROGRESS-REPORT-2026-09-30 — update de 1 día: **tres puntos nuevos de
curva (g520K/g545K/g573K) + control remask g341K aterrizado**.

---

## 1. Curva de exposición v5-1b — tres puntos nuevos

| ckpt | denseL3 | cons_inf | gx_ok inf | ctxnec | gx_ok ctx | C4 drop ctx | remask gx ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| g470001 | 0.517 | 0.658 | 0.309 | 0.728 | 0.285 | −9.5 | — |
| g495501 | 0.528 | 0.671 | 0.313 | 0.724 | 0.268 | −8.7 | — |
| g520501* | 0.520 | 0.662 | 0.318 | 0.731 | 0.269 | −8.5 | 0.316 |
| g545501 | 0.527 | 0.663 | 0.332 | 0.736 | 0.283 | −8.8 | 0.326 |
| **g573501** | 0.521 | 0.662 | 0.328 | **0.743** | 0.281 | **−10.2** | **0.328** |

\* g520501 evaluado sobre `model.pt` crudo — el snap se tomó antes de
que `ema_model.pt` estuviera (único punto non-EMA de la serie reciente);
sus números caen en la misma banda, mantener el caveat.

**Lecturas:**

1. **`cons_inf`: el watch item se resolvió a favor.** La regla era "si
   g520K repite ≥0.66 la meseta empieza a moverse" — repitió **tres
   puntos seguidos** (0.662/0.663/0.662) tras el 0.671 de g495K. Cuatro
   puntos consecutivos en/sobre el máximo histórico de la meseta
   (0.660, banda 0.652-0.658 durante ~290K steps). La meseta de
   plausibilidad subió ~+0.7pt y sostiene — movimiento pequeño pero
   consistente, no un outlier.
2. **`ctxnec` sigue con pendiente**: 0.743 en g573K = **nuevo máximo
   de la serie** (anterior 0.732 en g393K). Ventana g207K→g573K:
   0.690→0.743 (+5.3pt acumulado). El C4 drop se profundizó a
   **−10.2pt**, el más fuerte medido — el binding a contexto crece sin
   señal de saturación.
3. **`gen_exact` ctxnec 0.283/0.281** — consolida el rango nuevo
   (~0.28 vs ~0.24 al inicio de la ventana). `remask` gen_exact sigue
   subiendo: 0.316→0.328.
4. `dense` L3 plano (0.52) — solo continuidad, no decide.

## 2. Subtipos corregidos (cons_inf)

| ckpt | direction_w | negation | causal_w | causal_p | temporal_p | number | env_w | mechanism_w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| g495501 | 0.695 | 0.658 | 0.583 | 0.602 | 0.717 | 0.833 | 0.674 | 0.656 |
| g520501 | 0.688 | 0.684 | 0.579 | 0.590 | 0.667 | 0.852 | 0.633 | 0.625 |
| g545501 | 0.688 | 0.658 | 0.579 | 0.596 | 0.681 | 0.852 | 0.633 | 0.625 |
| g573501 | 0.686 | 0.665 | 0.574 | 0.596 | 0.681 | 0.852 | 0.633 | 0.625 |

Sin cambios estructurales: `number` resuelto (0.85), la cola sigue siendo
causal (~0.58-0.60) y la frontera fina direction/negation/temporal en
0.66-0.69. `entity` = 0.000 con n=1 — mismo caveat.

ctxnec subtipos:

| ckpt | role_swap | entity_rebind | value_rebind | direction_rebind |
|---|---:|---:|---:|---:|
| g520501 | 0.863 | 0.820 | 0.651 | 0.532 |
| g545501 | 0.863 | 0.811 | 0.661 | 0.596 |
| g573501 | **0.915** | **0.823** | 0.656 | 0.511 |

`role_swap` saltó a 0.915 (serie previa 0.84-0.87) y `entity_rebind` a
0.823 — los subtipos confirman que la subida de ctxnec es de binding,
no de una sola clase. `direction_rebind` n=47, ruidoso.

## 3. fgmode — caso cerrado

El control remask de mainline a dosis igualada aterrizó
(`g341001/remask_ctxnec`, job 30717045, EMA):

| métrica | mainline g341K | fgmode@g30000 | Δ |
|---|---:|---:|---:|
| remask gen_exact ctxnec | 0.303 | 0.311 | +0.8pt (<1σ) |

El único canal que quedaba pendiente confirma **FLAT**: fgmode no
levanta nada a dosis igualada en ninguno de los cuatro canales
(pairwise/gen_exact/C4/remask). Brazo cerrado, falsación limpia.

## 4. Gate block-diffusion — sigue en espera (correctamente)

La condición pre-registrada para activar el brazo BD3-LM era:
fgmode falsado (✓) **Y** exposición aplanada en ctxnec/gen_exact.

**La segunda condición NO se cumple**: ctxnec pairwise puso máximo
nuevo (0.743) y C4 drop el más profundo (−10.2pt) en el punto más
reciente — la pendiente está viva. Por la regla, el brazo queda en
espera (implementado + smoke OK en commit 411a5b1, sin lanzar;
verificado: no hay job blkdiff en cola). El escenario "meseta que
justifica cambio estructural" se está debilitando justo en la métrica
que decide.

## 5. Qué queda del plan (delta sobre §3 del 09-30)

| # | Punto | Estado | Próxima acción |
|---|---|---|---|
| P1 | Decisión fgmode | **CERRADO: falsado** | control remask confirmó flat — archivar brazo |
| P2 | Exposición | en curso, pendiente viva | vigilancia: cons_inf sostiene ≥0.66 (4 pts), ctxnec en máximo — próxima batería ~g598K |
| P3 | sciev backbone→decisión | 2 puntos | próximo re-eval ~g650-700K |
| P4 | remask en curva | cableado | ya aterrizando en todos los puntos (g520K+) |
| P5 | Chat-SFT v2 | listo, no lanzado | `sft_mdlm_v5.slurm` + `watch_sft_v2.sh` + smoke 30738663 OK; decisión de cuándo (¿esperar a ≥g700K como dice el plan, o lanzar sobre g495K/g573K ya?) pendiente con Angel |
| P6 | Corpus UNAM-EN | parcial | sin cambios — decisión de datos pendiente |
| P7 | block-diffusion | en espera activa | gate re-eval en cada punto de curva; si ctxnec se aplana 2 puntos seguidos → lanzar |
| P8/P9 | Papers | backlog | sin cambios |

## 6. Riesgos / vigilancia

- **Sixhour churn**: cadena AUTO_RESUBMIT sana; único job vivo es
  `v5-1b` (30790656). Vigilancia diaria de `state.json` como siempre.
- **g520K non-EMA**: si se quiere serie EMA pura, re-evaluar el punto
  cuando el EMA esté disponible en el snap — costo ~1h de batería.
- **`entity` n=1**: auditoría barata sigue pendiente.
- El ritmo actual (~85K/día) pone el target 1.3M en ~8-9 días si las
  islas aguantan.

## 7. Fuente de datos

- Curva: `/beegfs/a474r867/ecoreasoner/runs/v5-1b/battery_curve/g<N>/`
  (nuevos: g520501, g545501, g573501 — dense/cons_inf/cons_ctxnec
  +remask completos; g341001/remask_ctxnec para el control fgmode).
- Acceso: ssh `kuhpc` (esta máquina no monta /beegfs).
- Watcher: `data/watch_v5_battery.log` (sincronizado, última entrada
  10-01 14:31, latest=g581001).
