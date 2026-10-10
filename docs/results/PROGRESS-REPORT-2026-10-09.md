# EcoReasoner — Reporte de avance dLLM (2026-10-09)

Estado: **activo**. Mainline `v5-1b` en **g992.5K/1.3M (~76.3%)**,
cadena AUTO_RESUBMIT sana (job 31252735 corriendo; `state.json` fresco
23:04). Hubo un **apagón de beegfs ~10-07→10-08 (~24h de cómputo
perdido)** ya resuelto. Tres puntos nuevos de curva consolidados
(meseta total). **Chat-SFT v2 COMPLETADO y evaluado: chat=no
(veredicto §3)**. UNAM-EN al **~89%**. Watcher de baterías revivido
hoy (§6). Continuación del PROGRESS-REPORT-2026-10-04.

---

## 1. Curva mainline — tres puntos nuevos (EMA, consolidados)

| ckpt | denseL3 | cons_inf | gx_ok inf | ctxnec | gx_ok ctx | C4 drop ctx | remask gx ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| g710001 | 0.517 | 0.668 | 0.337 | 0.743 | 0.290 | −7.8 | 0.335 |
| g738065 | 0.517 | 0.665 | 0.334 | 0.738 | **0.293** | −8.7 | 0.335 |
| g763501 | 0.516 | 0.669 | 0.332 | 0.742 | 0.286 | −9.0 | 0.333 |
| g791001 | 0.506 | 0.652 | 0.326 | 0.743 | 0.279 | −9.6 | 0.333 |
| g816501 | 0.523 | 0.662 | 0.323 | **0.744** | 0.287 | −8.3 | **0.339** |

**Lecturas:**

1. **Meseta total**: `ctxnec` pairwise plano en ~0.743-0.744 (8 puntos
   en banda 0.728-0.744, ~250K steps); `cons_inf` ~0.66; C4 drop
   ~−8.5±1pt.
2. **`gen_exact` ctxnec también se aplanó** — la única pendiente viva
   del reporte 10-04 marcó 0.286/0.279/0.287, por debajo del pico 0.293
   de g738K. Ni discriminación ni generación exacta mejoran ya con
   exposición dentro de esta receta.
3. Subtipos estables: role_swap ~0.90 (máximo estructural), entity_rebind
   ~0.81-0.83, value_rebind ~0.66, direction ~0.55-0.60 (cola débil
   persistente).
4. `denseL3` ruidoso 0.506-0.523 — sin tendencia.

## 2. Ops: apagón beegfs 10-07→10-08 (ya resuelto)

- Última ola pre-incidente: job 31052532 CANCELLED 10-07 08:00.
  Siguiente ola viva: 31052583 desde **10-08 08:00** — **~24h sin
  entrenar** (ventana coincide con los probes de storage del 10-08:
  `beegfs_probe`×3, `du_probe`, `resfs_probe` FAILED, `sciev_probe`).
- Desde la recuperación la cadena corrió limpio: 31112029, 31157528,
  31161776, 31193657 COMPLETED → 31252735 RUNNING.
- Ritmo post-incidente ~55-75K/día (vs ~70-90K antes): la partición
  sixhour está saturada además por la flota `v6_*` de `xsdm_pipeline`
  (otro proyecto de la cuenta) — menos islas libres para v5-1b.
- **ETA a g1.3M**: ~310K steps restantes → **~4-5 días (~10-13/14)**.

## 3. Chat-SFT v2 — veredicto: chat=no (empírico)

Entrenamiento COMPLETO (g35500, 2 épocas, init EMA g738065; DONE
10-05 23:52). Evals automáticas corridas:

**(a) Generativa `eval_devin_hard` (3,000 held-out, gold_recall):**

| tipo | sft-v1 (init g94K) | sft-v2 (init g738K) |
|---|---:|---:|
| multihop (n=800) | 0.031 | 0.084 |
| numerical (n=1055) | 0.052 | 0.153 |
| negation (n=324) | 0.036 | 0.125 |
| causal (n=820) | 0.032 | 0.086 |

Recall **×3** vs sft-v1 (el backbone maduro sí carga más conocimiento),
pero absoluto bajo; `num_ok` 1-4%.

**(b) Regresión ctxnec vs init (EMA g738065):**

| métrica | init g738K | sft-v2 g35500 | Δ |
|---|---:|---:|---:|
| pairwise ctxnec | 0.738 | 0.733 | −0.5pt |
| gen_exact ctxnec | 0.293 | 0.263 | −3.0pt |
| remask gx ctx | 0.335 | 0.297 | −3.8pt |

**(c) Probe cualitativo `probe_sft_chat` (10-06, job 31043199):**
salida **degenerada** — loops de repetición ("the species' fundamental
fundamental niche is a of the...", "a star, or a star, or a star")
en los 3 prompts (EN/ES). El formato [USER]/[ASSISTANT] se respeta y
hay EOS@112, pero el contenido es inutilizable.

**Veredicto**: SFT-v2 deja señal medible (recall paramétrico ×3) pero
**chat productivo NO es viable a 1.13B con este nivel de exposición**
— confirma y extiende el "chat=no" del probe sobre backbone (09-30).
El tax sobre la batería de binding es pequeño (−0.5 a −3.8pt) pero real.
SFT queda como herramienta de sonda (QA→SFT para leer contenido
paramétrico), no como vía de producto. Reintentar post-1.3M con
SFT-v3 (UNAM-QA) dará el punto de comparación sobre el ckpt terminal.

## 4. UNAM-EN — traducción ~89% (verificado en cluster)

- **515 pendientes de 4,873** (conteo directo en beegfs hoy).
- Watcher mtolmo vivo (relanzó array 31253305 a las 19:22; 1 task viva).
- Ritmo ~100-150 docs/día → **ETA ~3-5 días**.
- `watch_unam_qa.sh` armado: al caer el flag MT dispara QA→SFT-v3
  automático sobre el ckpt terminal.

## 5. sciev — harness dev activo, 4º punto pendiente

- Desarrollo activo en `~/sciev_submit`: `antileak_v2.slurm` (10-09),
  `sciev_antileak` COMPLETED 10-08 (3h), **`sciev_serve` corriendo ahora**
  (job 31281652, LLaDA-8B-Instruct vía uvicorn en q6000 — infra del
  techo de referencia).
- El 4º punto de la trayectoria backbone→decisión sigue planeado para
  el ckpt terminal g1.3M.

## 6. Watcher de baterías — relanzado hoy

- Confirmado muerto desde 10-06 14:35 (sin entradas nuevas; mismo
  apagón). Puntos g841K-g975K saltados — **~7 checkpoints sin batería**
  (no recuperables salvo backfill manual sobre snaps pineados; solo
  existen snaps de los puntos ya evaluados).
- **Relanzado 10-09 23:41** (`watch_v5_battery.sh`): detectó gap
  816501→992501 y disparó batería en **g992501** al instante
  (jobs 31286081-83).
- Mitigación pendiente sin cambios: watchdog/cron para que no vuelva a
  morir 3 días sin notarse.

## 7. Estado del programa

| # | Punto | Estado | Próxima acción |
|---|---|---|---|
| P1 | Decisión fgmode | CERRADO: falsado (FLAT) | — |
| P2 | Exposición | **76.3%** (g992.5K/1.3M), meseta total | ETA ~10-13/14; curva automática restaurada |
| P3 | sciev backbone→decisión | 3 puntos; harness dev activo | 4º punto sobre ckpt terminal |
| P5 | Chat-SFT v2 | **CERRADO: chat=no** (evals + probe, §3) | SFT-v3 (UNAM-QA) repetirá la sonda sobre ckpt terminal |
| P6 | Corpus UNAM-EN | **~89% traducido** | MT ~3-5 días → QA→SFT-v3 automático |
| P7 | block-diffusion | CERRADO: falsado | eje arquitectura terminado |
| P8 | Paper 1 (systems+harness) | backlog | + material: apagón/ops + meseta total documentada |
| P9 | Paper 2 (tesis) | backlog | desenlace exposición→1.3M inminente (~4-5 días) |

## 8. Plan de aterrizaje (v5-1b → 1.3M)

- [x] Verificar/relanzar watcher battery — **hecho 10-09 23:41**
  (batería g992501 en vuelo).
- [x] Consolidar métricas g763K-g816K — **hecho** (tabla §1).
- [ ] Batería final + sciev final sobre ckpt terminal g1.3M (EMA+raw).
- [ ] UNAM-EN: MT termina ~3-5 días → cadena QA→SFT-v3 sola.
- [ ] Consolidar curva g207K→g1.3M + 2 falsaciones + trayectoria sciev
  + veredicto SFT → insumos P8/P9.
- [ ] Auditoría pendiente barata: `entity` n=1 en subtipos.
- [ ] Considerar: pausar competencia sixhour o aceptar ETA ~10-13/14
  (la flota v6_* xsdm está comiendo islas).

## 9. Fuente de datos

- Cluster (verificado directo hoy ~23:00): `runs/v5-1b/state.json`
  (g991001@23:04), `battery_curve/g{763501,791001,816501}/
  logicdiff_summary.json`, `runs/sft-v2/` (eval_devin_hard summary +
  gens), `runs/contrast/pairs_l3_ctxnec_sft-v2/logicdiff_summary.json`,
  `logs/probe_sft_chat_31043199.out`, squeue/sacct.
- Logs locales: `data/watch_v5_battery.log` (revivido 23:41),
  `data/watch_mtolmo.log` (vivo), `data/watch_sft_v2.log` (DONE),
  `data/watch_unam_qa.log` (armado).
