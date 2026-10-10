# EcoReasoner — Reporte de avance dLLM (2026-10-09)

Estado: **activo**. Mainline `v5-1b` último punto confirmado por watcher
**g831001/1.3M (~64%)** al 10-06 14:35; proyección al ritmo observado
(~70-85K/día): **~g1.06-1.11M (~82-85%)** hoy, ETA al target ~10-12/13.
**Chat-SFT v2 COMPLETADO 10-05** (g35500, 2 épocas) con evals automáticas
corridas. UNAM-EN al **~89%** de traducción. Continuación del
PROGRESS-REPORT-2026-10-04.

---

## 1. Curva mainline — tres puntos evaled post-10-04

El watcher encadenó baterías automáticas sobre:

| ckpt | lanzada | estado |
|---|---|---|
| g763501 | 10-05 ~23:11 | evaled (métricas en cluster) |
| g791001 | 10-06 01:11 (jobs 31030897/904/905) | evaled |
| g816501 | 10-06 09:11 (jobs 31036197-99) | evaled |

Último checkpoint observado por el watcher: **g831001** (10-06 14:35).
Las métricas de estos tres puntos viven en
`runs/v5-1b/battery_curve/` (beegfs) — pendiente consolidarlas al
próximo acceso al cluster.

**Vigilancia**: el log local del watcher battery no tiene entradas
post-10-06 14:35 (último `watcher v5-battery start` 12:35). La cadena
AUTO_RESUBMIT de entrenamiento es independiente del watcher — el pretrain
sigue cubierto — pero las baterías automáticas requieren verificar que el
watcher esté vivo en el próximo login (mismo patrón de hueco 10-06→10-09
que tuvo mtolmo y resultó ser watcher caído, no jobs muertos).

## 2. Chat-SFT v2 — COMPLETADO (10-05)

- Olas: 30996592 → 31001423 → 31010235; **target g35500 alcanzado
  10-05 ~05:51** (init EMA g738065 pineado; `sft_ids_v1.npz`,
  36,677 pares sciq+smoltalk, 2 épocas).
- Evals automáticas al cierre, sin intervención:
  - **Generativa**: `eval_devin_hard` sobre 3,000 held-out (job 31013127).
  - **Regresión ctxnec**: consistency + remask contra la curva del init.
- `SFT_V2_DONE.flag` caído 10-05 23:52. **Decisión pendiente**: leer las
  métricas (¿el SFT recupera capacidad chat sin degradar ctxnec?).
- Herramienta nueva: `probe_sft_chat.py/.slurm` (commit 031c17a) — probe
  de chat paramétrico sobre el ckpt SFT-v2 para la lectura cualitativa.

## 3. UNAM-EN — traducción ~89%, cadena QA→SFT armada

| fecha | pendientes | nota |
|---|---:|---|
| 10-06 12:35 | 1,382 | watcher mtolmo start (curada_v3/science: 4,873 docs) |
| 10-06 14:57 | 1,254 | hueco de log hasta 10-09 (watcher caído, flota siguió) |
| 10-09 13:17 | 613 | watcher relanzado → array 31196967 |
| 10-09 19:22 | 515 | flota muerta → re-lanzó array 31253305 |
| **10-09 22:14** | **515** | 1 task viva; ~4,358/4,873 ya traducidos |

Ritmo observado hoy: ~100-150 docs/día efectivos → **ETA traducción
~3-5 días**. Al caer pendientes=0: `watch_unam_qa.sh` (armado, esperando
flag MT) dispara la cadena **QA→SFT** automática post-traducción
(`unam_qa_finalize.slurm` → SFT-v3 sobre ckpt terminal, commit 8ae8010).

## 4. Estrategia de escala (doc 10-06)

`docs/estrategia-escala-2026-10-06.md` — conclusiones:

- **Cuello real = interconexión** (~200 MB/s efectivos inter-nodo), no
  GPUs. Multi-nodo vanilla colapsa (22 s comm vs ~1 s compute/paso);
  solo rinde con batch efectivo ×30+ (`sync_every`, ya en v4-lsgd,
  falta portar a v5) — decisión científica, no de infra.
- **Pool masivo sin usar: AMD MI210** (~81 GPUs, 64GB, gfx90a).
  `venv_rocm` ya existe; `smoke_mi210.slurm` listo para medir fwd/bwd
  de la arch v5-1b — si ≥~2K tok/s/GPU, una isla 3-ranks equivale a
  una isla NVIDIA rápida adicional (el pool es ~2.5× toda la NVIDIA
  rápida junta).
- Fleets ya explotan el paralelismo correcto (mtolmo/qagen 16-way);
  pools lentos ociosos (q6000×26, v100×30) son candidatos a un segundo
  backbone big-batch si Angel decide duplicar throughput agregado.

## 5. Estado del programa

| # | Punto | Estado | Próxima acción |
|---|---|---|---|
| P1 | Decisión fgmode | CERRADO: falsado (FLAT) | — |
| P2 | Exposición | ~64% confirmado, ~82-85% proyectado | curva sola hasta g1.3M; verificar watcher battery |
| P3 | sciev backbone→decisión | 3 puntos (último g738K: choice 0.540, GPQA≈LLaDA-8B) | 4º punto sobre ckpt terminal |
| P5 | Chat-SFT v2 | **COMPLETADO 10-05** (entrenamiento + evals) | leer métricas + probe_sft_chat → decisión |
| P6 | Corpus UNAM-EN | **~89% traducido** | terminar MT → QA→SFT-v3 automático sobre ckpt terminal |
| P7 | block-diffusion | CERRADO: falsado | eje arquitectura terminado |
| P8 | Paper 1 (systems+harness) | backlog | material listo (2 falsaciones pre-registradas) |
| P9 | Paper 2 (tesis) | backlog | depende del desenlace exposición→1.3M |

## 6. Plan de aterrizaje (v5-1b → 1.3M)

- [ ] **Verificar watcher battery** al próximo login (log local mudo
  desde 10-06 14:35); relanzar `watch_v5_battery.sh` si murió — las
  baterías g763K/g791K/g816K ya quedaron, faltarían las del tramo final.
- [ ] **Consolidar métricas g763K-g831K** (en beegfs, no en repo).
- [ ] **Decisión Chat-SFT v2**: leer eval_devin_hard (3000 gens) +
  regresión ctxnec ya corridas; probe_sft_chat para lectura cualitativa.
- [ ] **Batería final + sciev final** sobre ckpt terminal g1.3M
  (EMA + raw) — cierra curva de exposición y trayectoria
  backbone→decisión (4º punto).
- [ ] **UNAM-EN**: al terminar MT (~3-5 días) la cadena QA→SFT-v3
  dispara sola sobre el ckpt terminal.
- [ ] Auditoría pendiente barata: `entity` n=1 en subtipos.

## 7. Fuente de datos

- Logs locales espejo: `data/watch_v5_battery.log` (hasta 10-06 14:35),
  `data/watch_sft_v2.log` (completo, DONE 10-05 23:52),
  `data/watch_mtolmo.log` (vivo hoy 22:14), `data/watch_unam_qa.log`,
  `data/SFT_V2_DONE.flag`, `data/sft_v2_init.txt`.
- Reporte previo: `docs/results/PROGRESS-REPORT-2026-10-04.md`.
- Estrategia: `docs/estrategia-escala-2026-10-06.md`.
- Métricas de baterías recientes: `runs/v5-1b/battery_curve/g{763501,
  791001,816501}` en beegfs (no sincronizadas a este repo).
