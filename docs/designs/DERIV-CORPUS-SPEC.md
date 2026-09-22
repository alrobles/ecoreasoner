# SPEC — corpus `deriv`: slots forzados por contexto

> Issue #8 · Milestone M2 · 2026-09-22
> Diagnóstico: `docs/designs/REPLANTEO-2026-09-22.md` §1.2 — el corpus tiene
> hechos; le faltan **derivaciones**. Denoising sobre hechos arbitrarios
> enseña "los slots son impredecibles"; razonar exige practicar "este token
> está FORZADO por el contexto".

## 0. Definición operativa

Un item `deriv` es un documento donde al menos un span de contenido está
**determinado** por el resto del documento (ctx o candidato), verificable
programáticamente. "Forzado" = existe una única respuesta correcta dado el
texto; un lector razonador no tiene libertad.

## 1. Familias (4)

### F1 — Cloze aritmético verificable (`deriv_arith`)
Documento con valores observados + cantidad derivada.

- Plantillas: cambio porcentual ("from {a} to {b} → {c}%"), diferencia,
  razón, media de dos valores, tasa (N eventos / T unidades), conversión
  de unidades (ha→km², °C→K, mg→g).
- El generador COMPUTE la respuesta → forzada por construcción.
- Anti-template: ≥20 moldes de frase × vocabulario de dominios
  (eco/física/medicina/social) × órdenes (derivación antes/después de los
  datos) × ruido (oraciones distractoras sin relación numérica).
- Variante load-bearing: la conclusión depende del valor derivado
  ("the decline of {c}% exceeds the threshold → significant").

### F2 — QA extractiva (`deriv_qa`)
Pasaje real + pregunta cuya respuesta es un span literal del pasaje.

- Fuente: abstracts de `papers_db` (EN only — regla ES-crudo).
- Generación: teacher OLMo-2-13B (pipeline `qa_gen` existente) con la
  restricción "answer must be a contiguous span of the passage" +
  verificación programática (span ∈ pasaje).
- Formato esqueleto-compatible: `[PASAJE] ... [PREGUNTA] ... [RESPUESTA]
  <span>` — reusa el machinery de etapas (CF=1.0 enmascara la etapa
  candidata = la respuesta forzada).

### F3 — Contraste in-doc (`deriv_contrast`)
Oraciones donde el valor aparece Y se niega la alternativa.

- "abundance increased by 30%, not decreased" / "the effect was positive
  (not null)" / "N=42 individuals (not 21)".
- Fuente híbrida: (a) mutación dirigida de frases reales del corpus
  esqueleto insertando la negación explícita, (b) generación sintética.
- Enseña binding polaridad↔contexto y valor↔mención — el caso
  `number:copia` del split inferable como dato de entrenamiento.

### F4 — Silogismos con diversidad de superficie (`deriv_syll`)
premisa→paso→conclusión donde número/negación son load-bearing.

- Extiende `build_logic_synth.py`: combinatorial en {entidades, relaciones,
  valores, moldes de superficie} — NUNCA dos items con el mismo patrón
  superficial (anti-memorización de template: lección B3/B4).
- Incluir casos negativos honestos ("no conclusion follows" válidos) para
  que el modelo no aprenda "siempre hay conclusión".

## 2. Formato y mezcla

- JSONL `{text, lang:"en", src:"deriv_<fam>", domain:"deriv_<fam>"}`.
- Los docs con etiquetas `[ETAPA]` participan en CF/whole_stage como
  esqueletos; los de prosa reciben masking random-denso (receta hinrcf10).
- Share objetivo: **15–25% de tokens** de la mezcla deriv-v1
  (B3/B4/logic/hneg fallaron a <10% + eval rota; aquí el share es parte
  del diseño).
- Cap por familia: ninguna >8% del total.

## 3. Auditoría obligatoria (gate de calidad)

`audit_deriv.py` antes del pretok:
1. **Forcedness**: ≥90% de items F1/F3/F4 verificados por el propio
   generador (la respuesta es función del ctx); F2: respuesta es substring
   del pasaje.
2. **Anti-template**: ningún molde de superficie >5% de su familia.
3. **Leak**: dedup vs `pairs_hard_v4_holdout_clean` y `eval_devin_hard`
   (embeddings e5-small como en el leak-check previo, umbral >0.9).
4. Estadísticas: n docs, tok estimado, distribución de subtipo de slot
   forzado (number/negation/direction/entity/value-copy).

## 4. Integración con el experimento (B2/B3)

- `train_ids_deriv.npz` = mezcla base (v4en o v5pdb-share reducido) + deriv.
- Brazo experimental vs control pareado (mismo init, steps, seeds).
- Eval: `pairs_L3_inferable` + modo `consistency` (A1/A2) — la frontera
  corregida. Control secundario: `provenance` debe quedar ~plano (si sube
  provenance también, es memorización, no derivación).
- Gate pre-registrado (issue #10): inferable-L3 > control +2σ.

## 5. No-objetivos

- NO generar razonamiento largo estilo CoT todavía (eso es C2/trace-SFT).
- NO mezclar ES crudo.
- NO re-usar templates de build_b3_synthetic sin diversificación (fueron
  absorbidos por template, no por operación).
