# LOGICDIFF FASE A — RESULTADO (2026-09-12) — REABRE LA LÍNEA dLLM

## Veredicto

- **VERDICT: GO condicional sobre la MÉTRICA; NO-GO sobre el scheduler de orden.**
- El scoring **denso de candidato** (enmascarar TODOS los tokens del candidato y
  reconstruirlos condicionado al contexto) revela señal inferencial L3 que el
  scorer anterior (máscara random 15% sobre ctx+candidato) diluía a azar.
- **El dLLM V3-contrastive-fixed alcanza L3 = 0.640*** (p<10⁻⁵, IC95 [0.596,0.682],
  n=475) — por encima del umbral GO 0.55 pre-registrado.
- El scheduler LogicDiff (orden PREMISE→CONNECTIVE→DERIVED→CONCLUSION→FILLER)
  **no añade sobre dense**: staged ≈ dense ≈ staged_rev; staged_rand es PEOR.
- Conclusión reescrita: el NO-GO anterior de la familia dLLM estaba **confundido
  por un evaluador insuficientemente focalizado**, no por ausencia de señal.

## El defecto del evaluador anterior

`suite_smoke_v2.py` puntúa L3 con una sola pasada de denoising: enmascara el 15%
de las posiciones de `ctx+candidato` (elegidas al azar) y compara la loss media
sobre las posiciones enmascaradas entre candidato correcto y mutado.

Con seqs de ~600-700 tokens, la máscara cae mayoritariamente sobre contexto y
tokens FILLER del candidato; las ~5-15 posiciones que realmente distinguen el
payload mutado se enmascaran con probabilidad ~15% cada una → la señal se diluye
~6× en ruido de reconstrucción de contexto. Resultado histórico: L3 0.51-0.52 ns
en TODOS los checkpoints, leído como "gramática sin inferencia".

El modo `dense` de `harness/suite_smoke_logicdiff.py` enmascara el 100% de los
tokens del candidato (contexto intacto) — cada token del candidato debe
reconstruirse exclusivamente desde el contexto. Es una prueba inferencial pura:
la loss ya no compara "¿es plausible este texto?" sino "¿qué continuación predice
el contexto?".

## Resultados principales (eval limpio `pairs_hard_v3_eval`, seed 9999, n_L3=475)

### v3-role (`f0-span-v3-role/checkpoint-g10000`)

| modo | L0 | L1 | L2 | L3 | p(L3) |
|---|---|---|---|---|---|
| base (scorer v2) | 0.519 | 0.502 | 0.663*** | 0.507 | 0.392 ns |
| **dense** | 0.572*** | 0.534 | 0.682*** | **0.592*** | 0.00004 |
| staged (PREM→CONN→DER→CONC→FILL) | 0.572*** | 0.535 | 0.684*** | 0.585*** | 0.00012 |
| staged_rev (orden inverso) | 0.568** | 0.532 | 0.669*** | 0.585*** | 0.00012 |
| staged_rand (orden aleatorio) | 0.570*** | 0.532 | 0.732*** | 0.545* | 0.027 |
| payload (solo payload mutado) | 0.495 | 0.510 | 0.578*** | 0.432 | 0.999 ns |
| staged_head (roles por cabeza) | 0.570*** | 0.537 | 0.675*** | 0.579*** | 0.00034 |

### Barrido de checkpoints (base → dense, L3, n=475)

| checkpoint | L3 base | L3 dense | p | Δ |
|---|---|---|---|---|
| **v3-contrastive-fixed** (g-final) | 0.516 ns | **0.640*** | <10⁻⁵ | +0.124 |
| v3-curriculum (g10000) | 0.520 ns | 0.594*** | 3e-5 | +0.074 |
| v3-role (g10000) | 0.507 ns | 0.592*** | 4e-5 | +0.084 |
| v2-weight-tying (g10000) | 0.516 ns | 0.568** | 0.002 | +0.053 |
| v2-50m (g10000) | 0.531 ns | 0.556** | 0.008 | +0.025 |
| v2-piloto (g10000) | 0.524 ns | 0.554* | 0.011 | +0.030 |
| v2-no-whole-stage (g10000) | 0.528 ns | 0.552* | 0.014 | +0.023 |
| v2-rope (g10000) | 0.501 ns | 0.545* | 0.027 | +0.044 |
| **random-init (piso)** | — | **0.459 ns** | 0.967 | — |

**8/8 checkpoints entrenados: L3 significativo con dense (p<0.03 todos).
1/1 random-init: azar.** La señal no es idiosincrásica de un run: toda la
familia dLLM entrenada sobre esqueletos codifica discriminación inferencial,
y el scorer anterior la ocultaba sistemáticamente (Δ medio +0.057).

### Control de piso (random-init, job 29210739)

Modelo `MdLMMoE` con pesos aleatorios, mismas dims que v3-role:

- dense: L0 0.537 ns, L1 0.526 ns, L2 0.503 ns, **L3 0.459 ns**
- staged: L2 0.511 ns, **L3 0.432 ns**

→ El scorer `dense` no fabrica señal ni sesgo: en un modelo sin aprender,
  devuelve azar en todos los niveles. La señal L3 observada proviene de los
  pesos entrenados.

## Lectura crítica

1. **El efecto es del SCORING, no del ORDEN.** staged (orden lógico correcto) =
   staged_rev (orden inverso) = dense (todo a la vez) dentro de ~0.6 p.p.;
   staged_rand cae a 0.545*. Si el orden de dependencia causal fuera el
   mecanismo, rev debería dañar y rand debería ser intermedio-peor que rev — no
   se observa. Lo que importa es (a) enmascarar el candidato completo y (b)
   preferiblemente hacerlo en una sola pasada o en pocas.

2. **V3-contrastive-fixed era el mejor modelo todo el tiempo.** El run que el
   documento V3.3 declaró NO-GO con `eval_acc_holdout=0.77` "sospechoso" muestra
   la señal más fuerte (0.640). El holdout interno no era artefacto: la batería
   era demasiado débil para detectarlo.

3. **La cabeza de roles es débil pero irrelevante.** val micro 0.384 vs ~0.38 del
   baseline mayoritario; CONNECTIVE 0.00 (0.5% del corpus); DERIVED 0.57 el
   único rol realmente aprendido. staged_head ≈ staged_rule (0.579 vs 0.585)
   porque el orden apenas importa — no porque la cabeza clasifique bien.

4. **`payload` <0.5 es diagnóstico.** Enmascarar solo el payload mutado rompe la
   comparación (el resto del candidato visible "absorbe" la contradicción);
   L3 0.432 indica que la coherencia local del candidato domina cuando el
   contexto no puede anclarlo. Confirma que la discriminación correcta requiere
   que TODO el candidato sea reconstruido desde el contexto.

5. **Precaución estadística.** Los 3 checkpoints entrenados muestran Δ positiva
   (0.030-0.124) y los IC95 de dense en contrastive no tocan 0.5. Pero n=475 y
   exploramos 7 modos — el p<10⁻⁵ de contrastive sobrevive corrección Bonferroni
   holgada; el 0.554* de v2-piloto es marginal. Replicación pendiente: otro seed
   de pares, otro dominio, subtipos de mutación.

## Qué se aprendió sobre el modelo

- El dLLM entrenado con span masking sobre esqueletos científicos **sí codifica
  restricciones contexto→continuación**: dadas premisas+conclusión visibles,
  reconstruye mejor el candidato inferencialmente correcto que el mutado.
- La señal es **probabilística y débil en margen** (mean_delta ≈ 0.008-0.037
  nats/token en v3-role vs 0.037 en contrastive) — consistente con que el
  contraste fino de la mutación (dirección/magnitud/entidad) es una fracción
  pequeña de la loss total.
- El orden de denoising interno NO es el cuello de botella a esta escala; el
  scorer sí lo era.

## Implicación para el roadmap

- La familia dLLM-puro **queda reabierta** como componente de scoring/drafting:
  puede servir como evaluador de continuaciones inferenciales (rerank,
  verificación de drafts del AR, scoring de rutas de razonamiento) aunque no
  como generador de tool calls.
- NO continuar LogicDiff orden-dependiente como línea principal: la hipótesis
  "el orden de desenmascaramiento es la barrera" está falsada por staged≈rev.
- Línea natural siguiente: **scorer denso como módulo de discriminación**
  dentro del controller/verificator — el dLLM puntúa continuaciones propuestas
  por el AR/controller sin necesidad de generar él mismo.
- Opcional (Fase B): entrenar/fine-tunear con objetivo candidate-focused
  (enmascarar siempre el segmento conclusión durante entrenamiento) para ver si
  la señal escala más allá de 0.64.

## Reproducibilidad

- Evaluador: `harness/suite_smoke_logicdiff.py` (modos base/dense/staged/
  staged_rev/staged_rand/payload/staged_head, `--ckpt RANDOM` para piso).
- Pares: `runs/pairs_hard_v3_eval/` (seed 9999, holdout limpio).
- Jobs: 29210730 (v3role batería completa), 29210737 (v2-piloto),
  29210738 (contrastive), 29210739 (random), 29210751-55 (barrido base),
  29210757-61 (barrido dense), 29210756 + 29210762 (subtipos).
- Dataset de roles: `data/logicdiff/role_train.pt` (299,534 docs, 3.7GB).
- Cabeza: `runs/logicdiff-head-v3role/` (head.pt + metrics.json).
- Scripts: `scripts/build_logicdiff_dataset.py`, `scripts/train_logicdiff_head.py`,
  `scripts/logicdiff_{data,head,eval}.slurm`.

## Subtipos de mutación L3 (modo dense)

| subtipo | n | contrastive | v3-role |
|---|---|---|---|
| direction_word | 193 | **0.777** | **0.762** |
| causal_word | 66 | 0.515 | 0.515 |
| number | 52 | 0.481 | 0.308 |
| causal_phrase | 44 | 0.773 | 0.659 |
| negation | 43 | 0.465 | **0.116** |
| temporal_phrase | 33 | 0.545 | 0.545 |
| temporal_word | 23 | 0.478 | 0.783 |
| environment_word | 11 | 0.273 | 0.636 |
| mechanism_word | 9 | 0.889 | 0.667 |
| entity | 1 | 1.000 | 1.000 |

**La señal es real pero concentrada:** ambos modelos detectan muy bien el
cambio de dirección semántica (`aumenta`↔`reduce`, 41% de los pares) y
razonablemente los cambios de frase causal/mecanismo. Son ~azar o peores en
`number`, `negation`, `causal_word`. v3-role está activamente invertido en
negation (0.116) y number (0.308): prefiere sistemáticamente la versión
mutada — consistente con un modelo que aprende "plausibilidad de esqueleto"
sin fijar el contenido fino del payload.

Lectura honesta: el 0.64 de contrastive NO implica discriminación inferencial
uniforme — es una mezcla de ~0.78 en dirección/mecanismo y ~azar en
numérico/negación. El dLLM capta semántica direccional del argumento, no la
lógica fina de cantidades y negaciones.
