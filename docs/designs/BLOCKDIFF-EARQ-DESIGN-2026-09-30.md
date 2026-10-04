# E-arq — Block-diffusion semi-AR (BD3-LM-style) para el dLLM desde cero

Fecha: 2026-09-30 · Estado: **FALSADO 2026-10-04** (ver
PROGRESS-REPORT-2026-10-04 §2: todas las métricas primarias
retrocedieron vs mainline a dosis igualada y vs su propio init; el
único positivo literal fue C4 −10.1pt, leído como fragilidad, no
binding). Lanzado 10-03 job 30882855, warm-start EMA g683001.
Condición de activación (de PROGRESS-REPORT-2026-09-30 §3-P7):
fgmode falsificado (hecho: FLAT a dosis igualada) **Y** exposición
aplanada en métricas corregidas (ctxnec pairwise + gen_exact — aún con
pendiente al 30-09; mientras tenga slope, esperar).

Predecesores: `LOGICDIFF-FASE-B-INPUT.md` (BD3-LM ya fichado),
`OBJETIVO-ARQ-2026-09-22.md` (H-arq diferida), lit-review §9 del
PROGRESS-REPORT-2026-09-23 (LLaDA2.0 2512.15745, Fast-dLLM v2,
BD3-LM 2503.09573, PARD 2607.24306 — matiz: en MDLM puro el orden
arbitrario iguala a L2R dentro del bloque; el bloque importa como
unidad de *condicionamiento*, no como orden de decodificación).

---

## 0. Por qué este eje (motivación actualizada por los datos)

La frontera medida no es plausibilidad ni conocimiento — es **binding**:
- Decomposición honesta del pairwise: ~0.62 prior de plausibilidad +
  ~9pts de binding a contexto (C4 drop).
- `remask` (decodificación iterativa) añade +4.3pt gen_exact ctxnec sin
  tocar pesos → hay capacidad de binding *latente* que el forward único
  no extrae.
- Todo el stack de objetivos/datos probado (corrective, hinge, rebind,
  hneg, numw, schedules, seeds) = plano a dosis probada → el cuello es
  estructural en cómo el modelo *usa* el contexto, no en cuánto práctica.

**Hipótesis E-arq**: la atención full-bidireccional sobre todo el canvas
difumina la distinción "evidencia dada" vs "contenido a producir". En un
modelo BD3-LM, cada bloque se denoisa condicionado a bloques previos
**limpios** — la asimetría contexto→target está cableada en la máscara de
atención, no delegada al azar del masking. Predicción falsable: el binding
(measurable como C4 drop y ctxnec gen_exact) debería crecer a igual dosis.

## 1. Cambio técnico mínimo sobre `train_mdlm_moe_v2.py`

Modelo actual: MdLMMoE full-bidireccional (sin causal mask), seq 768.

1. **Partición en bloques**: tamaño B=32 (o 64) tokens. seq 768 → 24/12
   bloques. En docs esqueleto las etapas no respetan bloques — OK: el
   bloque es unidad de *condicionamiento*, no semántica.
2. **Máscara de atención 3-clases** por bloque b:
   - tokens de bloques < b (clean): atención completa
   - tokens dentro de b (noisy): atención mutua completa (bidir intra-bloque)
   - tokens de bloques > b: sin atención (causal entre bloques)
   Implementación: attention_mask aditiva por pares (i,j) = 0 si
   block(i)<=block(j), −inf si block(i)>block(j). El Block actual toma
   h y no tiene mask — añadir `attn_mask` param al forward.
3. **Objetivo por ejemplo**: elegir bloque b~U, denoising MDLM interno
   (t~U(0,1), SUBS/CE sobre el bloque) con bloques <b limpios y ≥b
   "contexto futuro no disponible" — o sea, se enmascaran/ocultan los
   bloques futuros (BD3-LM los trunca; más barato: máscara de atención
   -inf y mantener el tensor completo).
4. **Compatibilidad**: mismo corpus pretokenizado (train_ids_v5pdb), mismo
   optimizador/EMA/zero1/olas; `--block_size 0` = off (default = modelo
   actual, bit-identidad verificable).

## 2. Protocolo experimental (espejo del brazo fgmode)

- **Warm-start**: `--init_from` desde un ckpt EMA del mainline (p.ej.
  el pineado g495501) — los tensores son idénticos salvo la máscara de
  atención (nuevo comportamiento, no params). Loss de arranque esperado
  mayor que un warm-start normal (el modelo nunca vio atención causal
  por bloques) pero ≪ init random.
- **Dosis**: 30K steps (~276M tok) — idéntica a fgmode → comparación
  pareada directa.
- **Eval**: batería idéntica (`battery_point.sh` con EVAL_CFG propio):
  dense + cons_inf + cons_ctxnec + remask. El punto de discriminación
  principal: **ctxnec gen_exact_ok y C4 drop** (binding), no cons_inf.
- **GO (pre-registrado)**: ctxnec gen_exact_ok o pairwise > mainline a
  dosis igualada por ≥2σ (≈+2.5pt ctxnec pairwise, ~+2pt gen_exact), o
  C4 drop más profundo por ≥2pt. Plano → cerrar eje arquitectura y
  consolidar el negativo controlado.

## 3. Costo y riesgos

- ~200 líneas nuevas (máscara + partición + flag), test de
  bit-identidad con `--block_size 0`. Riesgo principal: el warm-start
  desde atención-full puede converger mal — mitigación: 5K-step smoke
  con gate "loss < init esperado + decay visible" antes de los 30K.
- Compute: ~1.5-2 días de isla (igual que fgmode). No compite con el
  mainline salvo por slots sixhour.
- Si gana: abre el camino a generación semi-AR (bloques L2R con
  KV-cache por bloques — backlog §1 del ROADMAP ya lo tenía fichado
  para decode largo).

## 4. Referencias ancla

BD3-LM 2503.09573 (kuleshov-group/bd3lms) · Ssd-LM 2210.17432 ·
LLaDA2.0 2512.15745 (WSD block-size) · Fast-dLLM v2 (KV por bloques) ·
PARD 2607.24306 (matiz orden-en-bloque) · "Masks Can Be Distracting"
2511.21338 (MDLMs usan contexto más global — nuestro binding quiere
exactamente eso; si block-diffusion lo gana es evidencia de que la
globalidad sin estructura desperdicia la señal).
