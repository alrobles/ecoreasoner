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

## 3b. sciev backbone→decisión — 3er punto (g738K EMA, job 30929279)

| eval | g316K | g495K | g738K | Δ último | techo LLaDA-8B |
|---|---:|---:|---:|---:|---:|
| choice elite (n=448) | 0.473 | 0.520 | **0.540** | +2.0 | 0.870 |
| noul elite (n=1200) | 0.667 | 0.667 | 0.667 | 0 | 0.783 |
| score elite (n=1218) | 0.329 | 0.334 | 0.336 | +0.2 | 0.608 |
| GPQA main (n=441) | 0.256 | 0.247 | **0.293** | **+4.6** | ~0.315 |
| GPQA diamond (n=193) | 0.275 | 0.269 | **0.326** | **+5.7** | ~0.33 |
| SciFact noul (n=332) | 0.593 | 0.596 | 0.569 | −2.7 | 0.853 |
| SciFact score (n=304) | 0.286 | 0.398 | 0.368 | −3.0 | 0.624 |

**Lecturas:**

1. **`choice` sigue creciendo**: 0.473→0.520→0.540 — desacelerando
   pero sin meseta; ~57% del headroom hacia LLaDA-8B extraído.
2. **GPQA se movió por primera vez**: main +4.6, diamond +5.7 — el
   backbone **empata prácticamente a LLaDA-8B** en GPQA (0.293/0.326
   vs ~0.315/0.33). Primera señal de conocimiento paramétrico
   emergiendo a ~5.6B tokens (caveat: GPQA está cerca del piso para
   todos a esta escala).
3. **SciFact retrocedió** (−2.7/−3.0 tras el salto +11.2) — un punto;
   ruido o regresión leve, el siguiente punto lo decide.
4. Net: la eval externa sigue convirtiendo exposición en capacidad
   decisoria **aunque la métrica interna pairwise ya está en meseta**
   — la trayectoria backbone→decisión no ha cerrado.

## 4. Qué queda del plan

| # | Punto | Estado | Próxima acción |
|---|---|---|---|
| P1 | Decisión fgmode | CERRADO: falsado | — |
| P2 | Exposición | en curso, ~58% | la curva sigue; ctxnec en meseta, gx la única pendiente viva |
| P3 | sciev backbone→decisión | **3er punto hecho (g738K)** | próximo: ckpt terminal 1.3M |
| P5 | Chat-SFT v2 | **LANZADO 10-04** | watcher encadena olas hasta g35500 (2 épocas); evals automáticas al final (eval_devin_hard + regresión ctxnec) |
| P6 | Corpus UNAM-EN | **EN CURSO 10-04** | flota traduciendo (array 30996717); al terminar: merge a `unam_en.jsonl` → decisión de integración (dosis post-1.3M o QA→SFT) |
| P7 | block-diffusion | **CERRADO: falsado** | eje arquitectura terminado (ambos brazos negativos) |
| P8 | Paper 1 (systems+harness) | backlog | material listo: crisis de instrumento + 2 falsaciones pre-registradas |
| P9 | Paper 2 (tesis) | backlog | depende del desenlace exposición→1.3M |

## 4b. Lanzamientos 10-04

**Chat-SFT v2** (`sft-v2`, job 30996592, watcher `watch_sft_v2.sh` vivo):
- init = EMA g738065 (snap pineado); datos `sft_ids_v1.npz` (36,677 pares
  sciq+smoltalk); 2 épocas ≈ g35500. Loss ~4-5 nominal. Al completar,
  el watcher dispara eval generativa (eval_devin_hard, 3000 held-out) +
  regresión ctxnec (consistency+remask) contra la curva del init.

**UNAM-EN — pipeline reactivado (variante compliant)**:
- Flota beta(w0-2)+gamma(w3-5) rsync → `tesis_unam/fleet_md/`:
  **5,131 tesis nuevas** (slices 0-5, scrape completo).
- Curado (`curate_tesis_unam.py`) → `curada_v3/`: **4,873 science**
  (258 dropped short; 0 latex — las slices ya venían filtradas).
  **Cero traslape** con los 2,031 docs ya traducidos.
- HY-MT purgado (veto 09-20) → traductor nuevo: **OLMo-2-1124-13B-
  Instruct** (compliant) vía `mt_olmo.slurm` + `traducir_vllm.py`
  (--dtype añadido; stack pylibs_vllm + VLLM_USE_V1=0, igual que
  qa_gen). Smoke 3 docs OK; flota array **30996717_[0-15]** corriendo,
  resume por `.en.md` → se relanza el array tras cada ventana sixhour.

## 5. Plan de aterrizaje (v5-1b → 1.3M)

Queda ~42% de corrida (~550K steps, ~5-6 días al ritmo actual). Para
cerrar la fase con el material consolidado:

- [ ] **Dejar correr** — watcher + AUTO_RESUBMIT cubren la curva solos
  hasta el final; baterías automáticas cada ~25K hasta g1.3M.
- [ ] **Batería final en ckpt terminal** (g1.3M, EMA + raw) — cierra la
  curva de exposición.
- [ ] **sciev final** sobre el ckpt terminal — 4º y último punto de la
  trayectoria backbone→decisión.
- [ ] **Decisión Chat-SFT v2** — el plan decía backbone ≥g700K (ya
  superado); lanzar cuando Angel confirme.
- [ ] **Consolidar**: curva completa g207K→g1.3M + las dos falsaciones
  pre-registradas (fgmode, blkdiff) + trayectoria sciev = insumo de
  papers P8/P9.
- [ ] Auditoría pendiente barata: `entity` n=1 en subtipos.

## 5b. Riesgos / vigilancia

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
