# EcoReasoner — Reporte de avance dLLM (2026-09-30)

Estado: **activo**. Pretrain `v5-1b` en **g501K/1.3M** (~38.5%), ritmo
reciente ~70-95K steps/día (isla 4×A40 actual; histórico ~46K/día) → ETA
~8-17 días al target. Curva automática de instrumentos corregidos cada
~25K steps (`battery_curve/`). Brazo `v5-1b-fgmode` (DeepSeekMoE)
completado a g30000 — **eval de decisión corrida hoy: FLAT**
(ver §2.3). Continuación del PROGRESS-REPORT-2026-09-23.

---

## 1. Objetivo vigente (tesis, sin cambios)

dLLM desde cero (<1B-activos) que (a) haga **inferencia científica ligada
a contexto** (verificar un slot contra evidencia visible, derivar
conclusiones forzadas por premisas) y (b) hable con utilidad. Corpus
(~12B tok esqueleto científico) e infraestructura resueltos; la variable
abierta es la capacidad.

Falsación pre-registrada original (≥55% pairwise transitorio /
esqueletos coherentes) ya quedó **superada por la corrección de
instrumento** (§3 del reporte 09-23): la métrica vieja `dense` medía
reconstrucción holística; `consistency`-sum (ctx visible, solo slot
enmascarado) es el estándar desde entonces.

## 2. Avance medido desde el último reporte (09-23 → 09-30)

### 2.1 Curva de exposición v5-1b (EMA, n=1628 inf / n=1200 ctxnec)

| ckpt | denseL3 | cons_inf | gx_ok inf | gx_bad | ctxnec | gx_ok ctx | C4 drop ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| g207501 | 0.546 | 0.657 | 0.307 | 0.085 | 0.690 | 0.240 | −6.9 |
| g234001 | 0.533 | 0.657 | 0.302 | 0.082 | 0.701 | 0.244 | −7.8 |
| g261001 | 0.539 | 0.652 | 0.305 | 0.082 | 0.702 | 0.248 | −7.0 |
| g288001 | 0.527 | 0.654 | 0.320 | 0.075 | 0.712 | 0.255 | −8.4 |
| g316001 | 0.522 | 0.657 | 0.318 | 0.081 | 0.709 | 0.243 | −7.3 |
| g341001 | 0.523 | 0.652 | 0.319 | 0.083 | 0.711 | 0.250 | −8.8 |
| g368001 | 0.528 | 0.652 | 0.303 | 0.089 | 0.718 | 0.243 | −9.5 |
| g393501 | 0.514 | 0.653 | 0.319 | 0.087 | 0.732 | 0.273 | −9.6 |
| g419260 | 0.515 | 0.657 | 0.313 | 0.085 | 0.730 | 0.271 | −8.8 |
| g444501 | 0.527 | 0.660 | 0.315 | 0.088 | 0.725 | 0.276 | −9.1 |
| g470001 | 0.517 | 0.658 | 0.309 | 0.084 | 0.728 | 0.285 | −9.5 |
| **g495501** | 0.528 | **0.671** | 0.313 | 0.079 | 0.724 | 0.268 | −8.7 |

**Lecturas (ventana g207K→g495K, ~2.7B tokens):**

1. **`cons_inf` sigue plano** (0.652-0.658) con un punto nuevo en
   **0.671** — el primero >0.66 de la serie. A n=1628, σ≈1.2pt → +1.3σ
   sobre la media histórica: *watch item*, no ruptura. Si g520K repite
   ≥0.66 la meseta de plausibilidad empieza a moverse.
2. **`ctxnec` sí tiene pendiente**: pairwise 0.690→0.732 (pico g393K,
   ~+3σ acumulado), C4 drop se profundiza (−6.9→−9.5pts). El componente
   de *binding a contexto* crece lento pero sostenido con exposición —
   la lectura "plausibility prior 0.62 + binding ~9pts" del 09-24 sigue
   vigente, con el binding engordando.
3. **`gen_exact_ok` ctxnec 0.240→0.285** (+4.5pts, >3σ acumulado en la
   ventana) — la única métrica con pendiente limpia; el modelo produce
   el slot forzado cada vez más, aunque el ranking siga saturado.
4. `dense` L3 (instrumento viejo, reportado solo por continuidad) decae
   0.546→0.528 — consistente con que dense mide reconstrucción
   holística, no la habilidad de interés.

### 2.2 Subtipos corregidos (cons_inf) — la "frontera number" era del instrumento viejo

| ckpt | direction_w (778) | negation (155) | causal_w (216) | causal_p (166) | temporal_p (138) | number (54) | env_w (49) | mechanism_w (32) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| g207501 | 0.677 | 0.619 | 0.593 | 0.590 | 0.688 | 0.870 | 0.653 | 0.594 |
| g316001 | 0.677 | 0.639 | 0.602 | 0.584 | 0.703 | 0.833 | 0.592 | 0.625 |
| g419260 | 0.673 | 0.703 | 0.574 | 0.602 | 0.667 | 0.833 | 0.633 | 0.594 |
| **g495501** | **0.695** | 0.658 | 0.583 | 0.602 | **0.717** | 0.833 | 0.674 | 0.656 |

- **`number` inferable = 0.83-0.87** (n=54 — el subset inferable, no el
  provenance). El "techo estructural ~0.40" declarado en G10 era
  correcto **solo bajo `dense`**: en consistencia corregida el número
  está resuelto a lo que la cobertura permite; `entity` = 0.000 con
  **n=1** (ruido confirmado, no colapso real — mismo caveat del
  reporte anterior).
- Frontera real ahora: **causal ~0.58-0.60, direction ~0.69,
  negation ~0.66** (subiendo) — la cola es causal/direccional fina, no
  binding de valor.
- ctxnec subtipos: role_swap 0.84-0.87, entity_rebind 0.75→0.82,
  value_rebind 0.61→0.66 — los tres con pendiente positiva lenta.

### 2.3 Estado de experimentos

- **B2-rbd (rebind-as-objective)**: FALSADO a la dosis probada
  (ctxnec 0.705, C4 −8.4 = control). El cuello de binding no está en el
  objetivo a p=0.7/5K steps.
- **EoS-latent pad**: FALSADO (nulo a k=32/128).
- **Remask (E-obj2)**: parcial positivo — +4.3pt gen_exact ctxnec,
  satura a la 1ª dosis → adoptado como estándar de reporte generativo.
- **v5-1b-fgmode** (32 finos@ff/2 top-2 + shared@ff/4, warm-start
  g316K): entrenamiento COMPLETO a g30000, **eval de decisión corrida
  hoy** (jobs 30705860-864; dense re-lanzado 30706429 tras TIMEOUT
  50min en Q6000 — fix aplicado: `g_eval_holdout_v5.slurm` ahora
  `--time=02:00:00`, ya sincronizado al cluster). Comparación a dosis
  igualada
  (mainline ≈g341-368K = init g316K + 30K):

  | métrica | mainline a dosis ≈igual | fgmode@g30000 | Δ |
  |---|---:|---:|---:|
  | cons_inf pairwise | 0.652-0.657 | 0.656 | ≈0 (flat) |
  | cons_inf gen_exact_ok | 0.303-0.319 | 0.296 | ≈0 |
  | ctxnec pairwise | 0.711-0.718 | 0.699 | −1.3pt (~1σ) |
  | ctxnec gen_exact_ok | 0.243-0.250 | 0.254 | ≈0 |
  | ctxnec C4 drop | −8.8/−9.5 | −7.2 | menos profundo |
  | dense L0/L1/L2/L3 | 0.770/0.762/0.539/0.523 | 0.762/0.761/0.543/0.527 | ≈0 |
  | remask gen_exact ctxnec | control mainline g341K lanzado (job
    30717045, aterriza en `v5-1b/battery_curve/g341001/remask_ctxnec/`;
    referencia a menor dosis: g185001 = 0.274) | **0.311** | pendiente |

  **Veredicto: FLAT → palanca falsificada a esta escala.** Fine-grained
  MoE no levanta la frontera a dosis igualada (cons_ctxnec 1σ abajo,
  todo lo demás dentro de ruido). Si el control remask aterriza ~0.27-0.30
  confirma flat; si cayera <0.27 sería el único punto a favor — no
  cambia la decisión primaria (el gate es cons_ctxnec/gen_exact). La
  palanca "más especialización MoE" queda falsificada → candidato
  estructural siguiente = **block-diffusion semi-AR**, diseño ya
  escrito en `docs/designs/BLOCKDIFF-EARQ-DESIGN-2026-09-30.md`
  (condicional a que la exposición se aplane).
- **sciev (backbone→decisión)**: trayectoria con 2 puntos (EMA,
  systemone-v2, canonical):

  | eval | g316K | g495K | Δ | techo LLaDA-8B |
  |---|---:|---:|---:|---:|
  | choice elite (n=448) | 0.473 | **0.520** | **+4.7** | 0.870 |
  | noul elite (n=1200) | 0.667 | 0.667 | 0 | 0.783 |
  | score elite (n=1218) | 0.329 | 0.334 | +0.5 | 0.608 |
  | GPQA main (n=441) | 0.256 | 0.247 | −0.9 | ~0.315 |
  | GPQA diamond (n=193) | 0.275 | 0.269 | −0.6 | ~0.33 |
  | SciFact noul (n=332) | 0.593 | 0.596 | +0.3 | 0.853 |
  | SciFact score (n=304) | 0.286 | **0.398** | **+11.2** | 0.624 |

  La decisión tipada **sí crece con exposición** (choice +4.7pt,
  SciFact-score +11.2pt) — consistente con la curva interna (ctxnec
  pairwise y gen_exact subiendo lento). GPQA plano = conocimiento
  paramétrico aún ausente a ~4B tokens vistos. Job: 30705884
  (`TAG=ev5g495k`, wrapper `ema_as_model.pt` en el snapshot pineado).

## 3. Qué queda del plan (puntos a seguir, priorizados)

| # | Punto | Estado | Próxima acción |
|---|---|---|---|
| P1 | **Decisión fgmode** (arquitectura) | eval en cola | leer g30000 vs mainline a dosis igualada (~g346K): si gx_ok/ctxnec > mainline → tronco nuevo; si plano → falsado, siguiente palanca estructural = block-diffusion |
| P2 | **Exposición** (¿más tokens rompen la meseta?) | en curso | la curva automática responde; vigilancia: cons_inf ≥0.66 sostenido y ctxnec/gy_ok pendientes |
| P3 | Trayectoria backbone→decisión (sciev) | 2º punto en cola | re-eval periódico cada ~150-200K steps (g495K hoy; próximo ~g650-700K) |
| P4 | remask como estándar generativo en la curva | **cableado hoy** | `battery_point.sh` ahora corre `remask` en ctxnec en todos los puntos futuros (commit + sync hechos) |
| P5 | Chat-SFT v2 sobre backbone maduro | esperando P1/P2 | re-SFT desde ckpt ≥g700K (sft-v1 demostró el pipeline; el límite era el backbone inmaduro) |
| P6 | Corpus UNAM-EN | parcial | slices 0-5 del scraper sin lanzar; solo entra traducido (lección G8: ES crudo degrada) — decisión de datos pendiente con Angel |
| P7 | E-arq: block-diffusion semi-AR | condicionado | solo si P1 falsado Y la exposición se aplana (evaluación LLaDA-2-style) |
| P8 | Paper 1 (systems: entrenamiento heterogéneo + harness + curva de instrumento) | backlog | el material de §7-8 (crisis de instrumento + contrast battery) es contribución central; redacción tras decisión fgmode |
| P9 | Paper 2 (tesis desde-cero) | backlog | depende del desenlace P1/P2 |

### Reglas vivas reafirmadas este ciclo

- Métrica primaria = `consistency` en `pairs_l3_inf` + `pairs_l3_ctxnec`
  con perfil C4; `gen_exact` (+remask) es la medida honesta de producción.
- `dense` se reporta solo por continuidad histórica — no decide nada.
- Selección por media entre seeds; holdout solo al campeón; toda ganancia
  pairwise debe sobrevivir al perfil C4 (flat = no context-bound).

## 4. Riesgos / vigilancia

- **sixhour churn**: el pretrain vive de resubmits en islas variables;
  la cadena AUTO_RESUBMIT ya sobrevivió el fix TERM/finalize y el
  cross-island opt-state sanitize. Vigilar diariamente `state.json`.
- **entity subtype = 0.000** en todos los puntos: verificar n y si es
  colapso real o residuo de construcción de pares (vale una auditoría
  barata de `l3_subtype_acc.entity.n`).
- **Truncación 768 vs LLaDA 4096** en sciev: asimetría de techo
  conocida; cuando seq_len suba (L5 escala) re-medir.

## 5. Fuente de datos

- Curva: `/beegfs/a474r867/ecoreasoner/runs/v5-1b/battery_curve/g<N>/`
  (dense/cons_inf/cons_ctxnec + snapshots pineados en `ckpts/`).
- Watcher local vivo: `scripts/watch_v5_battery.sh` (PID activo desde
  09-27), gap=25000, ahora con remask en ctxnec.
- Leaderboard GA (cerrado): `scripts/leaderboard.py --runs runs`.
