# REPLANTEO 2026-09-22 — dLLM que razona y chatea, from scratch

> ReumanLab · EcoReasoner. Documento de replanificación tras la auditoría
> eval→datos→objetivo del 22-09. **El objetivo no cambia**: un dLLM entrenado
> desde cero que (a) discrimina/repara argumentos científicos, (b) genera
> continuaciones derivables, (c) chatea QA científico. Lo que cambia es el
> ORDEN: medir bien → enseñar la operación que falta → entonces escalar/chat.

---

## 1. Diagnóstico (con evidencia)

### 1.1 El eval `dense` mide menos de lo que creíamos

En `harness/suite_smoke_logicdiff.py` modo `dense`, el **100% del candidato**
va enmascarado → el input de `ok` y `bad` es *literalmente idéntico*
(`[ctx][MASK…MASK]`). El score colapsa a: ¿logp(token verdadero | ctx) >
logp(token mutado | ctx)? Dos consecuencias no contabilizadas:

- **Toda evidencia intra-candidato desaparece.** Ejemplo real del holdout:
  `"We included 3.25 patients"` (mutación de `13`) es absurdo — pero solo si
  se ve "patients", que está enmascarado. Ídem `"Despite X exhibit…"`
  (ungramatical): detectable solo con el resto de la frase visible →
  `causal_word` 0.44 en v5-1b no es fallo de inferencia, es que la pista
  está tapada.
- **`number` era estructuralmente inganable.** `"population of 100
  oscillators"→"200"`: el valor exacto NO está determinado por el ctx — es
  correcto solo *por procedencia*. El "techo binding ~0.40" declarado en
  G10/G11 es en parte una propiedad del eval, no (solo) del modelo. El GA
  usó `min(num,neg)` como fitness durante 11 generaciones contra un subtipo
  en gran parte imposible.
- **`entity` acc=0.0 en la curva v5** era n=1 en holdout_clean — ruido.
- **Lo que sí funciona es inferencia genuina**: `direction_word` 0.64–0.79 —
  el ctx (hipótesis/predicción) sí determina la dirección → el modelo hace
  binding ctx→slot cuando el slot está forzado. Señal real, más estrecha
  que "razonamiento".

**Conclusión**: "estructura sin inferencia" y "techo estructural" son
conclusiones contaminadas por el instrumento. La frontera real es menor
(slots determinables por ctx) y mayor (binding con evidencia interna del
candidato, nunca medido) a la vez.

### 1.2 Mismo mismatch en el objetivo de entrenamiento

Denoising sobre corpus real: los slots enmascarados son en su mayoría
**hechos arbitrarios** → el gradiente enseña "los slots son impredecibles;
adivina lo típico". El eval pide "este slot está forzado por el contexto".
El modelo nunca practica la operación que medimos. Los sintéticos
(B3/B4 FLD, logic-30K, hneg pair-docs) fueron <10% share, templated, y
medidos contra la métrica medio-rota → su "no movió" está confundido.

**El corpus tiene hechos; le faltan derivaciones.**

### 1.3 Chat: el decode no es el problema

`sft_mdlm.py` implementa bien LLaDA §2.3 (prompt visible, response
enmascarada t~U(0,1), CE solo en response+EOS). `generate_conf` es correcto;
salida idéntica a temp 0.05/0.7 = distribución picuda de backbone inmaduro
(~9% de tokens al momento del init), no bug. `eval_devin_hard` es QA
extractiva multihop que exige recall factual tipo memorización — barra
desproporcionada para 280M activos; gold_recall 0.039 mezcla inmadurez +
eval miscalibrada.

### 1.4 Lo que sí está sano

- Infra: olas SIGUSR1/SIGTERM, flota flexible multi-familia, watchers,
  EMA (+1.9pt holdout gratis), pretok streaming, zero1 cross-world-size.
- Datos: papers_db 3.14M docs dedup, skeleton_db 1.04M, corpus v4en,
  UNAM traducido, QA pipeline OLMo-teacher.
- v5-1b en curso = sustrato LM: L0/L1 récord (0.74/0.73), ppl_proxy 926.
- Receta hinrcf10 (random+hi+CF1.0+nr+ws0) probada transplanteable (v4s:
  L3 0.600 holdout tras 6K steps de anneal estructurado).
- Pipeline SFT verificado end-to-end (shakedown, no resultado).

---

## 2. Plan por fases

### FASE A — Instrumento correcto (prerequisito de toda decisión)

| # | Trabajo | Salida | Gate |
|---|---|---|---|
| A1 | Modo `consistency` en suite_smoke_logicdiff: candidato VISIBLE salvo el slot mutado (extender `payload`); + medida generativa argmax en el slot | eval que mide binding con evidencia interna | reproduce densos en casos conocidos |
| A2 | Split de pares L3: `inferable` vs `provenance-only` (heurística por subtipo + juez LLaDA-8B en muestra) | `pairs_L3_inferable.jsonl` | juez concuerda ≥80% con heurística en muestra |
| A3 | Re-baseline: hinrcf10, retrain_v4s, v5-1b@g actual, sft-v1 en modos dense+consistency | tabla de frontera real | — |

### FASE B — Datos con slots forzados (el eje no agotado)

| # | Trabajo | Salida | Gate |
|---|---|---|---|
| B1 | Spec + builder corpus `deriv`: (i) cloze aritmético verificable, (ii) QA extractiva (respuesta forzada por pasaje), (iii) contraste in-doc, (iv) silogismos sintéticos con diversidad de superficie | `corpus_deriv_v1.jsonl` ~15-25% share en mezcla | audit: ≥90% de items tienen respuesta forzada (verificación programática) |
| B2 | Pretok + mezcla `train_ids_deriv.npz`; brazo de continuación desde ckpt v5-1b maduro (o v4s si v5 sigue inmaduro), receta hinrcf10 | run `deriv-v1` | — |
| B3 | Eval comparativa deriv vs control a igual steps | veredicto | **Pre-registrado: si inferable-L3 no sube >2σ vs control → hipótesis de datos falsada → siguiente sospechoso: objetivo (head discriminativo explícito) o arch** |

### FASE C — Chat sobre backbone maduro

| # | Trabajo | Salida | Gate |
|---|---|---|---|
| C1 | Eval QA-cloze calibrada a escala (respuestas cortas derivables, no multihop memorización) | `eval_qa_cloze.jsonl` + scorer | — |
| C2 | Corpus trace-SFT (premisa→paso→conclusión; reusar build_logic_synth + teacher) + re-SFT desde ckpt post-deriv | `sft-v2` | gold_recall en QA-cloze > sft-v1 baseline |
| C3 | Decode sanity en ckpt maduro: sweep temp × conf-unmask, verificar EOS aprendido (terminación) | nota | sin loops a temp 0 en 20 prompts |

### OPS paralelas

| # | Trabajo | Nota |
|---|---|---|
| O1 | v5-1b sigue corriendo a TARGET 1.3M con battery curve cada 25K | **punto de decisión ~g300K**: si inferable-L3 plano ahí → escala no resuelve, acelera Fase B |
| O2 | push repo (main 95 commits ahead) | higiene |

---

## 3. Milestones y criterios de salida

- **M1 "medir bien"** (Fase A): frontera re-baselineada en métrica ganable;
  sabemos qué parte del L3 actual era artefacto. ~días, cómputo trivial.
- **M2 "enseñar derivación"** (Fase B): corpus deriv + run + veredicto
  pre-registrado. ~1-2 semanas según cola.
- **M3 "habla con sustancia"** (Fase C): sft-v2 con trazas + QA-cloze
  medible. Depende de M2 y de madurez v5-1b.

Falsación global vigente: si tras B el modelo no genera slots forzados
mejor que azar → la línea dLLM-desde-cero para razonamiento generativo
queda falsada a nuestra escala; el modelo queda como scorer/verificador
(fortaleza ya demostrada) y el paper reporta el negativo limpio.

---

## 4. Reglas heredadas que siguen

Loss ≠ proxy; test-and-drop <4h GPU antes de escalar; pool real ≠ sinfo;
no tocar trainer vivo; sync a repo siempre; YAML validado + sbatch
--export con `;`; nunca ES crudo al corpus; réplicas por seed, nunca
best-of-N; holdout solo al campeón validado.
