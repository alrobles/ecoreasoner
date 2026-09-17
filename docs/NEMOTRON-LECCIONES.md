# Lecciones del repo nemotron-eco-reasoner (Kaggle Nemotron Reasoning Challenge)

> Estudio 2026-09-17 del repo `alrobles/nemotron-eco-reasoner` (LoRA sobre
> Nemotron-3-Nano-30B-A3B para el challenge "Alice's Wonderland") y de los
> writeups públicos de los ganadores. Objetivo: extraer lecciones medidas
> que apliquen a la línea viva EcoReasoner-dLLM (GA sobre receta, piso
> `number`, gen contrastive G10).

## 1. El problema cryptarithm (la categoría imposible)

`cryptarithm_deduce`: ejemplos `AB$CD = XY` donde hay que deducir
**simultáneamente** (a) un cifrado inyectivo símbolo→dígito compartido
entre ejemplos y (b) la semántica del operador `$` (familia de ~150 ops
enteras/string: add±k, mulmod, div, concat, rev). Es un CSP con
backtracking — ningún LLM lo resuelve en un forward pass.

Medidas del repo:

| Quién | Cobertura cryptarithm |
|---|---|
| Solver propio (`solve_cryptarithm_ops.py`, forward-checking) | ~60-68% del subset resoluble; budget 15s |
| Solver del ganador (tonghuikang, 0.877) | ~8% — **la abandonó** |
| Nemotron fine-tuned (eval local) | 0/8 = 0% |

Fracción del test: 8.7%. Puzzles con operador del query ausente de los
ejemplos son genuinamente irresolubles (~20% del total).

## 2. Mapa de fallos por categoría (Nemotron base + LoRA v8)

| Categoría | Test % | Nemotron | Método del solver ganador |
|---|---|---|---|
| numeral | 16.6% | 8/8 | romanos greedy |
| unit_conversion | 16.8% | 7/8 | cadenas mult/div |
| cipher | 16.6% | 6/8 | mapeo char-a-char |
| gravity | 16.8% | 0/8 local* | long-div + truncate_3dp |
| bit_manipulation | 16.9% | **1/8** | enumeración exhaustiva ops×bit (1005 líneas) |
| equation_numeric | 7.7% | **2/8** | familia de operadores oculta por símbolo |
| cryptarithm | 8.7% | **0/8** | CSP cifrado+operador conjunto |

*gravity: n=8, ruido de medición; pasaba en Kaggle real.

Lección: **~83.5% del test eran categorías "fáciles"** que el solver
sacaba ~100% → el score se ganaba barriendo lo fácil, no crackeando lo
imposible. El cuello no se ataca de frente, se rodea.

## 3. Las tres lecciones duras (todas medidas en Kaggle)

1. **Volumen sin verificación = regresión.**
   v8 (CoT genérico): 0.67 → v11 (más del mismo): 0.54 → v9 (44.5%
   cryptarithm): 0.46. v12 tenía **12.2% de respuestas `\boxed{}`
   erróneas** (trazas `hypothesis_formed`/`rule_unknown` del solver del
   ganador, casi todo cryptarithm irresoluble) → modelo confiadamente
   equivocado. Regla: *sólo entrenar trazas verificadas contra gold*.

2. **Truncación silenciosa.** 50.2% de v12 >seq3072 → las trazas se
   cortaban antes de `\boxed{}` → el modelo aprendía a divagar sin
   comprometer respuesta. La categoría más dañada era bit_manipulation,
   la más valiosa. Regla: *el contrato de formato del eval es parte del
   objetivo de datos*.

3. **El objetivo importa más que el optimizador.** El ganador usó
   **min-logprob** (maximizar el logprob *mínimo* de la traza, no CE
   media) + masked-token training + 8.4K augmentos auxiliares. El 3er
   lugar (0.90, YS-L) usó **two-stage**: drill especializado de la
   categoría dura → mixto con recall-replay; lm_head en LoRA; LoRA en
   fp32; completion-only loss; **gradient-tying de expertos MoE**
   (sumar grads entre expertos, no compartir paso por top-1).

## 4. Qué es análogo en EcoReasoner-dLLM

| Nemotron | EcoReasoner | Lección transferible |
|---|---|---|
| cryptarithm 0/8 | subtipo `number` 0.38-0.42 (piso en ~90 runs) | El payload simbólico exacto no se aprende con CE media sobre prosa plausible; hace falta objetivo de margen o datos verificados |
| min-logprob del ganador | `contr_w` hinge (implementado 17-09) | Misma familia: peor-caso/margen vs media. **Validación externa de la dirección G10** |
| 12.2% respuestas erróneas | corrective p=0.10 bimodal | Corromper sin verificar introduce label-noise: la dosis fina (G9 corrlo/corrmd) es la respuesta correcta |
| "no sobre-representar lo irresoluble" (v9) | mutaciones sobre subtipo ruidoso | Si `number` tiene techo estructural, forzar volumen degrada — los brazos scratch responderán |
| two-stage drill + replay | curriculum + candidate_focus | Drill de lo difícil primero, luego mezcla con replay = el patrón ya en uso |
| stratified sampler por categoría | `mras_floor`/`contr_p` mezcla uniforme | Balancear la señal por tipo dentro del batch evita jitter de gradiente |
| solver verificado → CoT | knn_edges → hard negatives | Datos nuevos deben venir verificados, no generados a ciegas |
| truncación/`\boxed{}` | masking budget vs T=768 | El contrato del eval (parwise logprob) es parte del diseño de datos |

## 5. La lección más incómoda

El ganador **abandonó** cryptarithm (~8% del test, ~8% resoluble) y aun
así sacó 0.877 barriendo el resto. Traducción a EcoReasoner: si `number`
tiene un techo estructural (multi-token, cobertura de tablas
`num_ids=10/126k`), la estrategia óptima puede ser **asegurar el resto
de subtipos a ~0.8+ y aceptar number como pérdida acotada**, en vez de
quemar generaciones del GA en él. El fitness `min(num,neg)` actual
castiga exactamente esa lectura — considerar fitness alternativo
`L3_global` (number pesa poco en n) o `min` sobre un subset excluyendo
number.

Decisión pendiente del estudio: medir el **techo teórico de `number`**
bajo las tablas CORRUPT actuales (¿qué fracción de los pares number del
eval tienen el token mutado dentro del vocabulario mutable?) antes de
seguir invirtiendo en ese subtipo.

## 6. Fuentes

- Repo estudiado: `github.com/alrobles/nemotron-eco-reasoner`
  (`docs/POST_HACKATHON_ANALYSIS.md`, `docs/archive/DESIGN_V13.md`,
  `docs/RUNBOOK_V14_TIED.md`, `docs/archive/HANDOFF.md`,
  `scripts/solve_{cryptarithm,equation,bit}*.py`)
- Ganador: tonghuikang/nemotron (0.877) — solvers deterministas,
  min-logprob, masked-token training
- 3er puesto: YS-L (0.900 en rescoring) — two-stage SFT + MoE
  grad-tying + lm_head LoRA
