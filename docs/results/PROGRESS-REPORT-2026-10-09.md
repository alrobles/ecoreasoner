# EcoReasoner — Reporte de avance dLLM (2026-10-09)

Estado: **activo**. Mainline `v5-1b` en **g992.5K/1.3M (~76.3%)**,
cadena AUTO_RESUBMIT sana (job 31252735 corriendo; `state.json` fresco
23:04). Hubo un **apagón de beegfs ~10-07→10-08 (~24h de cómputo
perdido)** ya resuelto. Tres puntos nuevos de curva consolidados
(meseta total). **Chat-SFT v2 COMPLETADO y evaluado: chat=no
(veredicto §3)**. UNAM-EN al **~89%**. Watcher de baterías revivido
hoy (§6). Continuación del PROGRESS-REPORT-2026-10-04.

---

## 1. Curva mainline — seis puntos nuevos (EMA, consolidados)

| ckpt | denseL3 | cons_inf | gx_ok inf | ctxnec | gx_ok ctx | C4 drop ctx | remask gx ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| g710001 | 0.517 | 0.668 | 0.337 | 0.743 | 0.290 | −7.8 | 0.335 |
| g738065 | 0.517 | 0.665 | 0.334 | 0.738 | **0.293** | −8.7 | 0.335 |
| g763501 | 0.516 | 0.669 | 0.332 | 0.742 | 0.286 | −9.0 | 0.333 |
| g791001 | 0.506 | 0.652 | 0.326 | 0.743 | 0.279 | −9.6 | 0.333 |
| g816501 | 0.523 | 0.662 | 0.323 | 0.744 | 0.287 | −8.3 | 0.339 |
| g992501 | — | 0.665 | 0.331 | **0.748** | 0.282 | −8.5 | 0.322 |
| g1017501 | — | 0.660 | 0.332 | 0.748 | 0.292 | −8.3 | 0.329 |
| g1043501 | — | **0.676** | **0.338** | 0.743 | **0.293** | **−8.0** | **0.339** |

(g992501+ evaluados por el watcher revivido — la automatización
post-apagón funciona: 3 puntos de curva producidos sin intervención.)

**Lecturas:**

1. **Meseta total**: `ctxnec` pairwise en banda 0.743-0.748 (~330K
   steps); `cons_inf` ~0.66-0.68; C4 drop ~−8.5±1pt.
2. **`gen_exact` ctxnec también se aplanó** — la única pendiente viva
   del reporte 10-04 oscila 0.279-0.293 alrededor del pico 0.293 de
   g738K. Ni discriminación ni generación exacta mejoran ya con
   exposición dentro de esta receta.
3. g1043501 marca el punto más fuerte del tramo (cons_inf y gx ctx en
   máximo de serie, C4 mínimo) — desviación dentro de la banda, no
   ruptura de meseta.
4. Subtipos estables: role_swap ~0.90 (máximo estructural), entity_rebind
   ~0.81-0.83, value_rebind ~0.66, direction ~0.55-0.60 (cola débil
   persistente).
5. `denseL2`/`denseL3` no consolidados aún en los puntos nuevos (batería
   en vuelo al cierre).

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

## 4. UNAM-EN — traducción ~89% + fix de stall (10-11 01:30)

- **515 pendientes de 4,873** — pero el conteo estaba **congelado**:
  las tasks morían en `llm.chat` con `ValueError: prompt 7198 tok >
  max_model_len 4096`. `para_chunks` corta por **caracteres** (4000);
  contenido denso en tokens (tablas/secuencias/fórmulas) supera 4096
  igualmente. Un solo chunk-ballena en la ola de 64 docs mata toda la
  ola → crash-loop: los ~32 pendientes por shard caben en una ola con
  al menos una ballena → progreso cero.
- **Fix desplegado en `traducir_vllm.py`** (repo + beegfs, md5 igual):
  1. `fit_tok()` — sub-división recursiva hasta que el prompt quepa en
     presupuesto de tokens `(maxlen−160)/2.2` (~1789 tok a maxlen 4096).
  2. `max_tokens` clamp: `min(n*1.15+128, maxlen−n−48, 6000)` — prompt+
     salida nunca exceden la ventana (además corrige la fórmula previa
     `n*1.7/3.5`, que sobre tokens —no chars— daba ~la mitad del
     presupuesto pensado → sub-generación latente).
  3. `llm.chat` en try/except: una ola fallida cuenta `nfail` y sigue,
     no mata el task.
- El task vivo con código viejo (31447540_0) morirá en su ballena; el
  watcher relanzará el array 0-15 con el fix → las ballenas se traducen
  por fin. Pendiente confirmar descenso del conteo en el próximo ciclo.
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
- [x] Auditoría `definitional` n=1 — **CERRADA** (§9: sesgo de split,
  integridad OK, fix propuesto para SFT-v3).
- [ ] Considerar: pausar competencia sixhour o aceptar ETA ~10-13/14
  (la flota v6_* xsdm está comiendo islas).

## 9. Auditoría `definitional` n=1 — CERRADA (causa raíz encontrada)

El eval `eval_devin_hard` (3,000 items) tiene solo **1 `definitional`**.
Causa: el split `qa_devin_elite` (3,339) → {supp 339, eval 3,000} envió
**129/130 definitional al supp de train** (verificado por match de
preguntas: 129 en supp, 1 en eval, 0 huérfanos). No es azar (esperanza
~13): el supp se seleccionó con sesgo hacia items chat-style.

- **Integridad OK**: los sets son disjuntos — no hay fuga train→eval;
  las métricas reportadas no están contaminadas.
- **Efecto**: eval_devin_hard es ciego a QA definicional (masa ~4%);
  el agregado es válido, la cobertura por tipo es asimétrica.
- **Fix recomendado (SFT-v3)**: construir una tajada eval de
  definitional con items UNAM-QA nuevos (nunca en train) — barato y
  sin contaminar la comparación sft-v1/v2 (los 129 ya están en
  sft_ids_v1; moverlos ahora sí crearía fuga).

## 10. Spike MI210/ROCm — VALIDADO (gate superado ~12×)

Medido hoy en r06r18n01 (`venv_rocm` torch 2.5.1+rocm6.2, sin apptainer):

| config | tok/s | peak mem | nota |
|---|---:|---:|---|
| 1×MI210 bs4 | **13,838** | 16.4 GiB | single-GPU, DDP-free |
| 1×MI210 bs12 | **24,065** | 24.4 GiB | ~64GB deja bs≥12 |
| 3×MI210 DDP bs4/rank | **13,761 agg** | 26.0 GiB | = receta viva (9,216 tok/step) |
| 3×MI210 DDP bs12/rank | **30,921 agg** | 32.7 GiB | batch ×3 = receta distinta |

Hallazgos del spike:
1. **RCCL requiere `NCCL_P2P_DISABLE=1` + `NCCL_IB_DISABLE=1`** — sin
   ellos init_process_group falla ("invalid argument"); con ellos el
   transporte cae a SHM y conecta (jobs 31442554/61/81).
2. **DDP + MoE top-1 exige `balance_loss` en la loss** — el trainer ya
   la lleva (por eso `find_unused_parameters=False` funciona en
   producción); el smoke la replica.
3. **Una isla 3×MI210 a la receta viva (bs4/rank = 9,216 tok/step) da
   13.8K tok/s agg ≈ 1.5-2.5× una isla NVIDIA actual**; a bs12/rank
   (batch efectivo ×3, cambio de receta) 30.9K tok/s ≈ 3-6×.
4. `v5_1b_mi210.slurm` **validado en producción** (job 31442672, acotado
   a 50 steps, OUT de test): pipeline completo real — corpus 11.96B tok
   memmap, modelo 1.13B, receta idéntica, `checkpoint-g50` +
   `training_complete.flag` escritos, exit 0 limpio. Mismo formato de
   ckpt que las olas NVIDIA → resume cross-stack sin conversión.
   Dependencias resueltas: `pydeps_tf451` en PYTHONPATH (transformers
   sin torch, no sombrea ROCm).
   Pendiente solo la decisión: ¿integrar MI210 a la flota del mainline
   o reservarlo para un segundo backbone?

**Mainline: g1,061,501/1.3M (81.7%)** a 10-11 01:14 — ritmo ~64K/día,
ETA ~10-14. Job 31446847 vivo; watcher battery produciendo puntos.

## 11. Fuente de datos

- Cluster (verificado directo hoy ~23:00): `runs/v5-1b/state.json`
  (g991001@23:04), `battery_curve/g{763501,791001,816501}/
  logicdiff_summary.json`, `runs/sft-v2/` (eval_devin_hard summary +
  gens), `runs/contrast/pairs_l3_ctxnec_sft-v2/logicdiff_summary.json`,
  `logs/probe_sft_chat_31043199.out`, squeue/sacct.
- Logs locales: `data/watch_v5_battery.log` (revivido 23:41),
  `data/watch_mtolmo.log` (vivo), `data/watch_sft_v2.log` (DONE),
  `data/watch_unam_qa.log` (armado).
