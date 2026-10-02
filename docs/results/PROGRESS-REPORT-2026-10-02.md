# EcoReasoner — Reporte de avance dLLM (2026-10-02)

Estado: **activo**. Pretrain `v5-1b` en **g632K/1.3M** (~48.6%), job
30797247 en r32r05n01 — la cadena AUTO_RESUBMIT siguió sana durante
un incidente del watcher (§3). Continuación del
PROGRESS-REPORT-2026-10-01: un punto nuevo de curva (**g630501**,
EMA ✓) + resolución del watch item de `cons_inf`.

---

## 1. Curva de exposición — punto nuevo g630501

| ckpt | denseL3 | cons_inf | gx_ok inf | ctxnec | gx_ok ctx | C4 drop ctx | remask gx ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| g520501* | 0.520 | 0.662 | 0.318 | 0.731 | 0.269 | −8.5 | 0.316 |
| g545501 | 0.527 | 0.663 | 0.332 | 0.736 | 0.283 | −8.8 | 0.326 |
| g573501 | 0.521 | 0.662 | 0.328 | **0.743** | 0.281 | **−10.2** | 0.328 |
| **g630501** | 0.521 | 0.667 | 0.330 | 0.728 | 0.283 | −7.4 | **0.337** |

\* non-EMA (ver reporte 10-01). Hueco g573K→g630K por el incidente
del watcher — ver §3.

**Lecturas:**

1. **`cons_inf`: meseta nueva confirmada.** 0.667 en g630K = **quinto
   punto consecutivo ≥0.66** (0.671/0.662/0.663/0.662/0.667) contra la
   banda histórica 0.652-0.660 que duró ~290K steps. Deja de ser watch
   item: el nivel nuevo es ~0.665, +~1pt sobre la meseta vieja.
2. **`ctxnec` se enfría tras el pico**: 0.728, ~1σ bajo el máximo de
   g573K (0.743) pero sobre la banda 0.69-0.73. C4 drop vuelve a −7.4
   (vs −10.2). La lectura honesta: **banda alta con ruido**, no
   pendiente limpia — el punto ~g655K decide si g573K fue spike o
   escalón.
3. **`gen_exact` ctxnec 0.283** consolida el rango ~0.28; **remask
   gx 0.337 = nuevo máximo de serie** (0.316→0.326→0.328→0.337) — la
   capacidad latente de binding que remask extrae sigue creciendo aun
   cuando el forward único se estabiliza.
4. Subtipos cons_inf: `number` 0.87 (resuelto), `env` subió a 0.694,
   `temporal_p` 0.688, `direction` 0.689, `negation` 0.671; causal sigue
   de cola (0.574/0.602). ctxnec: `role_swap` de vuelta a 0.863 (el
   0.915 de g573K fue spike), `entity_rebind` 0.799, `value_rebind`
   0.650, `direction_rebind` 0.617 (n=47, ruidoso).

## 2. Gate block-diffusion — aún en espera, borde acercándose

La condición sigue sin cumplirse del todo: un punto plano tras un
máximo no es meseta. Pero la evidencia se movió hacia aplanamiento —
ctxnec pairwise dejó de poner máximos (0.728 < 0.743). Regla operativa:
**si ~g655K repite ~0.72-0.73** (segundo punto sin máximo nuevo),
la condición "exposición aplanada" se considera cumplida y el brazo
BD3-LM (implementado, smoke OK, commit 411a5b1) se lanza con
warm-start desde el EMA pineado más reciente.

## 3. Incidente: watcher de baterías muerto ~26h

- `watch_v5_battery.sh` (proceso local que SSH-polea kuhpc cada hora)
  **murió el 10-01 ~14:31** sin error en el log — último estado
  `latest=g581001`. Sin baterías para ~g598K y ~g623K; el trainer solo
  retiene ~2 ckpts → **puntos irrecuperables** (hueco de 57K en la
  curva).
- Relanzado 10-02 16:10 con `setsid nohup` (sobrevive cierre de
  sesión). Al arrancar detectó el gap y disparó batería en g630501 de
  inmediato (jobs 30797608-610, snap con EMA ✓).
- Mitigación pendiente: considerar un watchdog/cron que re-lance el
  watcher si muere — el diseño actual depende de que un proceso nohup
  viva semanas.

## 4. Qué queda del plan (delta sobre §5 del 10-01)

| # | Punto | Estado | Próxima acción |
|---|---|---|---|
| P1 | Decisión fgmode | CERRADO: falsado | archivado |
| P2 | Exposición | en curso | cons_inf ≥0.66 ya es nivel establecido; vigilancia pasa a ctxnec: ¿banda 0.72-0.74 = meseta? → decide gate blkdiff |
| P3 | sciev backbone→decisión | 2 puntos | ventana abriéndose: próximo re-eval ~g650-700K (ya estamos en rango) |
| P5 | Chat-SFT v2 | listo, no lanzado | decisión de cuándo pendiente con Angel (≥g700K según plan; backbone ya maduro) |
| P7 | block-diffusion | en espera activa | gatillo reformulado: 2º punto ctxnec sin máximo nuevo → lanzar |

## 5. Fuente de datos

- Curva: `/beegfs/a474r867/ecoreasoner/runs/v5-1b/battery_curve/g630501/`
  (dense/cons_inf/cons_ctxnec+remask completo, EMA).
- Watcher relanzado: `data/watch_v5_battery.log`, entries cada ~1h.
- Acceso cluster: ssh `kuhpc`.
