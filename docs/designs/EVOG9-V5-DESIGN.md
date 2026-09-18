# EVOG9 / v5 — diseño de la siguiente ruta del GA

> 2026-09-17 · post-V4S. Fuente de verdad de números: ROADMAP §4 +
> `runs/*/battery_logicdiff*/dense/battery_L3.json` + leaderboard.py.

## 1. Qué la campaña demostró (estudio de logros)

**Receta (espacio de genes de masking, backbone 90M denso):**
- GA convergió: `hinrcf10` = random + uniform{b_l:0.30, b_h:curric→0.95}
  + role_mask OFF + candidate_focus=1.0 + whole_stage. Óptimo local duro:
  ~90 corridas, toda mutación posterior empató o degradó.
- Los genes de literatura (cos, mrd, corr, numw, mras) son INERTES o dañinos
  a 90M. La frontera respondió a *volumen de práctica inferencial*
  (densidad + CF), no a arquitectura ni marcadores.
- Selección: dev n=475 sobrevende (mirages por seed); holdout n≈2000/nivel
  redefine podios. Protocolo seeds/réplicas vigente.

**Corpus:**
- ES crudo degrada la discriminación EN ~5σ (g8-c1 0.583 vs c1e 0.627).
  Lección permanente: UNAM solo entra traducido.
- Cap 40K/dominio correcto (medgen 33%→11.6% no fue el problema).
- Más corpus NO compró inferencia (v4 flat: L3 cayó a azar).

**Escala + receta (v4/v4s):**
- MoE 676M + receta plana: superficie sube (L0 0.696), inferencia colapsa
  (L3 0.496). INVERSIÓN documentada.
- MoE 676M + receta estructurada (v4s, +6K desde g11000): L3 0.600
  holdout / 0.627 dev (= hinrcf10), L2 0.694, L0/L1 quedan sobre el denso,
  ppl_proxy 926 (mejor medido — la capacidad absorbe el objetivo sin pagar
  LM). **La receta se transplanta; los dos regímenes coexisten.**

**Frontera dura (invariante a TODO):**
- `number`: 0.29–0.42 en las ~90 corridas (dev n=52, holdout n=193). Ni
  capacidad, ni steps, ni corrective p=0.10, ni numw, ni corpus logic lo
  mueven. Es EL cuello.
- `negation`: alta varianza por seed (0.14–0.65). v4s dev 0.651, holdout
  0.574. Mejorable pero ruidoso.
- `direction`: saturado 0.72–0.80. `temporal_phrase` colapsó en v4s
  (0.362 vs 0.573 flat) — único subtipo herido por el port.

## 2. Hipótesis sobre el piso `number`

El par number = discriminar "increased by 41%" vs "increased by 17%" dado
el contexto. Exige ligar magnitud→evidencia. Por qué podría estar
estructuralmente capado:

1. **Objetivo**: el denoising CE premia "un número plausible", no "EL
   número correcto". Mutar el candidato deja el ruido en posiciones
   informativas pero la presión sobre el token exacto es débil.
   → gen `contrastive-in-loss`: margen entre reconstrucción del candidato
   real y uno mutado-in-place (alineación directa con la métrica del eval;
   distinto del v3.3-fallido que era fase separada con scorer distinto).
2. **Dosis**: corrective p=0.10 fue bimodal/adverso ({0.279,0.581}); la
   dosis fina (0.02–0.05) nunca se probó — la señal existe pero saturada
   destruye. A 676M hay headroom de capacidad para absorberla.
3. **Datos**: knn_edges.tsv (16-NN emb_v1) mina docs que son near-miss
   reales entre sí — nunca entró como señal de entrenamiento.
4. **Varianza**: EMA de pesos (gen `ema_decay`, ya implementado en trainer,
   nunca evaluado — eval no carga ema_model.pt, hace falta flag) podría
   estabilizar la frontera ruidosa.

## 3. Diseño G9 — mutantes sobre backbone v4s (MoE 676M)

Convención heredada: `-sK` por seed, media entre seeds, fitness =
0.5·L3 + 0.5·min(num,neg) en dev v3_eval; holdout_clean solo al campeón.
Dos clases de brazo:

**Brazos continuación (desde retrain_v4s/g17000, +3K steps → 20000,
LR 1e-4 flat — baratos, ~40min c/u en GPU moderna):**

| tag | gen | env |
|---|---|---|
| g9-corrlo-s1/s2 | corrective fino | CORRECTIVE_P=0.02 |
| g9-corrmd-s1 | corrective medio | CORRECTIVE_P=0.05 |
| g9-numwl-s1 | numw suave | LOSS_NUM_W=1.5 LOSS_NEG_W=1.5 |
| g9-ep2-s1 | +práctica | (solo más steps del backbone) |
| g9-ema-s1 | EMA pesos | EMA_DECAY=0.999 + eval --ema (req. patch) |

**Brazos fresh (10K desde cero, ~2.2h en GPU moderna — linaje limpio):**

| tag | gen | nota |
|---|---|---|
| g9-scratch-s1 | MoE+hinrcf10 desde 0 | ¿el pre-historial flat de v4s
techa el techo? curriculum completo b_h 0.30→0.95 |
| g9-scratch-s2 | réplica | varianza propia |

**Fuera de G9 (necesitan código/datos nuevos):**
- `contrastive-in-loss` — **IMPLEMENTADO 17-09** (`--contr_w`,
  `--contr_margin`, `--contr_p` en `train_mdlm_moe_v2.py`; env
  `CONTR_W/MARGIN/P` en `g0_run.slurm`): hinge sobre
  `logp[real] − logp[mutado]` en posiciones enmascaradas mutables
  (is_num|is_flip de las tablas CORRUPT), sin forward extra. Alternativa
  = otro dígito del pool o antónimo de `flip_cand`. Default 0 → off.
  Limitación heredada: las tablas ven `num_ids=10`, `flip_ids=104`
  del vocab — la cobertura de mutables es estrecha.
- `hneg-data` (corpus += docs minados por knn_edges con número/negación
  load-bearing mutada — necesita build step de datos).
- `seq1024` (pos-emb 768→1024 rompe ckpt; solo en brazo fresh).

## 3b. G9 — RESULTADO FINAL (18-09, 8/8 evaluados dev v3_eval)

| brazo | L3 | num | neg | FIT |
|---|---|---|---|---|
| **ema** | **0.6421** | 0.3269 | 0.6977 | **0.4845** |
| corrmd | 0.6274 | 0.3077 | 0.6512 | 0.4675 |
| numwl | 0.6232 | 0.3077 | 0.4186 | 0.4654 |
| ep2 (control +steps) | 0.6189 | 0.3077 | 0.6047 | 0.4633 |
| corrlo ×2 seeds | 0.6106 | 0.3077 | 0.7093 | 0.4591 |
| scratch ×2 seeds | 0.5621 | 0.3077 | 0.3023 | 0.4219 |

**Campeón: `g9-ema-s1`. Holdout_clean (ema_model.pt@g20000, n=1767):**
L0 0.624 / L1 0.585 / L2 0.684 / **L3 0.619*** — vs backbone v4s 0.600
→ EMA suma +1.9pt en el árbitro. Subtipos: number 0.394 (n=193),
negation 0.613, direction 0.738, causal_word 0.597, temporal_phrase
0.464 (sigue herido vs flat 0.573).

Lecturas: (1) weight averaging generaliza mejor — el gen ganador queda
ON en G10; (2) fresh-from-scratch queda 6pt abajo → el pre-historial
flat no techaba, la continuidad suma; (3) number plano ~0.31 dev /
0.39 holdout en TODOS los brazos → confirma que no es de cobertura ni
de steps: es binding de valor; (4) numwl sacrifica negación (−0.23)
sin ganar number — mal trato, gen descartado; (5) corrective bimodal
otra vez (0.585/0.636) → dosis fina es ruido neutro.

## 4. Ruta más allá de G9 — revisada con lecciones Nemotron (17-09)

Estudio completo: `docs/NEMOTRON-LECCIONES.md`. Tres lecciones medidas
del Kaggle que reordenan G10:

1. **Cobertura antes que objetivo.** El ganador abandonó cryptarithm
   (~8% resoluble) y ganó barriendo el resto. **Audit 18-09
   (`scripts/audit_mutable_coverage.py`, alineación SequenceMatcher
   sobre pairs_L3 dev n=475 y holdout n=1632)** — corrige la estimación
   previa (~1%, que contaba ruido de desfase posicional):

   | subtipo | n_hold | mutable actual | expansión propuesta (medida) |
   |---|---|---|---|
   | number | 193 | 85.7% (dígitos sueltos) | **97.9%** (`num_piece`) |
   | negation | 155 | 64.0% (is_flip) | **100%** (+auxiliares/verbos reporte) |
   | direction_word | 778 | 10.1% | **56.5%** (+inflecciones+símbolos) |
   | causal_word | 216 | 0.0% | **58.4%** (+conectivas) |
   | causal_phrase | 166 | 2.6% | **32.0%** (phrases multi-token, más duro) |
   | temporal_word | 39 | 0.0% | **79.5%** |
   | temporal_phrase | 138 | 17.8% | **77.3%** |
   | environment/mechanism | 81 | 0.0% | 50–64% (+familias open-class frecuentes) |

   **Conclusión invertida**: `number` NO falla por cobertura (ya 85%) —
   falla porque el swap dígito↔dígito exige *binding* del valor correcto
   desde el contexto (razonamiento), no solo discriminación local. La
   clase mutable sí tiene huecos reales en los subtipos léxicos —
   `direction`/`causal`/`temporal` mutan palabras que la tabla no cubre
   (inflecciones `increased/reduced`, conectivas `however/therefore`,
   `because/despite`, temporales `acute/chronic`, `initial/final`…)
   y símbolos (`≥↔≤`, `→`, `±`, `=` en byte-BPE). Esos subtipos puntúan
   bien *a pesar* de la cobertura — la expansión es barata y ayuda al
   hinge, pero ya no es el cuello demostrado. El audit está en
   `scripts/audit_mutable_coverage.py` (standalone, fallback
   tokenizer.json sin transformers).
2. **Solo datos verificados** (v12: 12.2% respuestas erróneas → modelo
   confiadamente erróneo). `hneg-data` vía knn_edges debe filtrar a
   mutaciones cuyo target quede dentro de la clase mutable expandida;
   un hard-negative fuera de cobertura es label-noise.
3. **min-logprob ≈ hinge**: el ganador maximizaba el logprob mínimo de
   la traza, no la media CE — validación externa de `contr_w`.

### Plan G10 concreto (tras leer G9)

| paso | qué | criterio |
|---|---|---|
| 0 | expandir tabla mutable: `_NUMBER_RE`→`num_piece` + flip-table con las familias medidas (dirección inflecciones, conectivas causales, temporales, símbolos, auxiliares neg) — lista exacta del audit | re-audit: direction>50%, causal>40%, temporal>60%, number>95% |
| 1 | `g10-contr-s1/s2`: backbone = ganador G9, `CONTR_W={0.3,1.0}` × `CONTR_MARGIN={1.0,2.0}` | ΔL3 y Δnumber vs backbone |
| 2 | `g10-contrcov-s1`: mismo brazo SIN expansión de cobertura | aísla si el efecto es el gen o la tabla |
| 3 | `g10-hneg-s1`: corpus += pares knn_edges verificados (mutación dentro de clase mutable) | Δnum/neg |
| 4 | reportar **dos fitness**: `0.5·L3+0.5·min(num,neg)` y `L3 solo` | si number no se mueve pese a cobertura+hinge, se declara techo estructural y se deja de invertir (lección ganador) |

- **Post-training** (árbol de capas, ortogonal al GA): L2 chat-SFT con
  teacher LLaDA-8B servido en cluster; L3 trazas premisa→paso→conclusión;
  L4 tool-calls con scaffold externo. El ckpt v4s es el base para todo.
- **L5 escala** (1.5B-act/4B MoE, seq 1024): solo con cómputo de grant;
  G9-v5 decide si la receta aguanta otro salto de escala.
- Criterio de parada del GA (propuesta): cuando 2 generaciones
  consecutivas no muevan media+σ, cerrar y escribir el mapa completo como
  resultado negativo-controlado (publicable por sí mismo).

## 5. Riesgos / notas

- Brazos continuación comparten linaje → correlación entre arms; el
  par fresh es el control limpio.
- `ema`: resuelto 17-09 — eval acepta `ema_model.pt` vía unwrap
  `{"ema_model": sd}` en suite_smoke_logicdiff + eval_ppl_proxy;
  `EVAL_EMA=1` en g0_run selecciona el ckpt EMA al evaluar.
- number n=52 en dev es pequeño → cualquier gen que "lo mueva" a 0.5
  puede ser ruido; el holdout (n=193) arbitra al campeón.
- Q6000 (Turing, sin bf16) excluido por MIN_VRAM=40000; si el pool se
  seca a solo Q6000, bajar BATCH a 2 accum 8 y MIN_VRAM a 20000.
