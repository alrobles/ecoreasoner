# EcoReasoner — Reporte de avance dLLM (2026-10-03)

Estado: **activo, dos brazos corriendo**. Mainline `v5-1b` en
**~g715K/1.3M (~55%)** (job 30858002, r32r05n01) y **brazo
block-diffusion LANZADO** tras cumplirse el gate (§2). Continuación
del PROGRESS-REPORT-2026-10-02.

---

## 1. Curva — puntos nuevos g656001 y g683001 (EMA)

| ckpt | denseL3 | cons_inf | gx_ok inf | ctxnec | gx_ok ctx | C4 drop ctx | remask gx ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| g573501 | 0.521 | 0.662 | 0.328 | **0.743** | 0.281 | **−10.2** | 0.328 |
| g630501 | 0.521 | 0.667 | 0.330 | 0.728 | 0.283 | −7.4 | **0.337** |
| g656001 | 0.518 | 0.655 | 0.329 | 0.739 | **0.289** | −8.3 | 0.335 |
| g683001 | 0.512 | 0.659 | 0.329 | 0.737 | 0.277 | −8.4 | 0.328 |

**Lecturas:**

1. **`ctxnec`: aplanamiento confirmado.** Tres puntos consecutivos sin
   máximo nuevo (0.728/0.739/0.737 tras el 0.743 de g573K). Pendiente
   decayó: +4.2pt en los primeros ~186K de la ventana vs +1.3pt en los
   últimos ~188K → banda ~0.735, meseta estructural.
2. **`cons_inf` se enfría**: 0.655/0.659 tras los 5 seguidos ≥0.66. El
   movimiento real sobre la meseta vieja es ~+0.5pt con distribuciones
   traslapadas — ganancia marginal, no escalón.
3. `gen_exact` ctxnec aún pone máximos (0.289 en g656K) — la única
   señal con vida, y débil. `remask` también aplanándose (0.335/0.328
   tras el 0.337).
4. Subtipos estables: causal ~0.58-0.59 sigue de cola; `entity_rebind`
   0.80-0.83, `role_swap` 0.86-0.88, `value_rebind` 0.65-0.66.

## 2. Brazo block-diffusion — LANZADO (gate cumplido)

- **Job 30882855** (`v5-1bblk`), corriendo desde 10-03 ~22:41 CDT en
  **3×A100 80GB** (r31r35n01) — aterrizó en isla rápida de una.
- **Warm-start**: EMA **g683001** (snap pineado) — `882/882 tensores,
  0 reinit`. Se repuntó del diseño original (g495501) para medir el
  delta estructural contra el backbone actual; la comparación a dosis
  igualada es contra mainline ~g713K (= g683K + 30K).
- **`--block_size 32` verificado** en el cmdline del proceso (3 ranks).
- Loss: 2.52 → 2.10 en los primeros 30 steps — arranque elevado
  esperado (el modelo nunca vio atención block-causal) ≪ init random,
  decayendo. Smoke previo (30739949) ya validaba la cadena.
- Dosis: **30K steps** (idéntica a fgmode). ETA ~1.5-2 días en A100.
- AUTO_RESUBMIT + watchdog activos; propagan `INIT_FROM` y
  `BLKDIFF_ARGS` entre olas.

## 3. Protocolo de decisión (pre-registrado, §2 del diseño)

Al completar g30000 → `battery_point.sh` con
`RUN_OUT=runs/v5-1b-blkdiff EVAL_CFG=eval-moe-v5-blkdiff.yaml`
(yaml ya synced al cluster). Comparación a dosis igualada vs mainline
~g713K (batería g710001 ya corriendo — servirá de referencia cercana).
**GO**: ctxnec gen_exact_ok o pairwise > mainline por ≥2σ
(≈+2.5pt pairwise / ~+2pt gx), o C4 drop ≥2pt más profundo. Plano →
cerrar eje arquitectura y consolidar el negativo.

## 4. Qué queda del plan (delta sobre §4 del 10-02)

| # | Punto | Estado | Próxima acción |
|---|---|---|---|
| P1 | Decisión fgmode | CERRADO | archivado |
| P2 | Exposición | meseta en ambas métricas | la curva automática sigue (próx. ~g737K); vigilancia ya solo de pendientes residuales en gx/remask |
| P3 | sciev backbone→decisión | en ventana | lanzar re-eval sobre ~g710-740K (tercer punto de trayectoria) |
| P5 | Chat-SFT v2 | listo, no lanzado | decisión pendiente con Angel |
| P7 | block-diffusion | **CORRIENDO** | eval de decisión al completar 30K (~10-05) |

## 5. Fuente de datos

- Curva: `/beegfs/a474r867/ecoreasoner/runs/v5-1b/battery_curve/g<N>/`
  (nuevos: g656001, g683001 EMA; g710001 en cola al cierre de este
  reporte).
- Brazo: `/beegfs/a474r867/ecoreasoner/runs/v5-1b-blkdiff/` (train.log,
  state.json, ckpts cada 500).
- Watcher: `data/watch_v5_battery.log` — vivo (generó g656K/g683K/g710K
  sin huecos).
