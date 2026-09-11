# Plan de rescate para dLLM que razone

> Fecha: 2026-09-11.
> Contexto: V3.3 NO-GO. El usuario pide re-explorar opciones antes de archivar
> definitivamente la línea dLLM-pura.

## 1. Diagnóstico del fracaso

Los 7+1 runs (v2, ablaciones, V3.1, V3.3) muestran un patrón estable:

- **L2 alto** (0.56–0.63): el modelo aprende estructura/etapas.
- **L3 chance** (0.48–0.51): el modelo no distingue contenido inferencial.
- El ranking L3 del trainer (0.77) no se traduce en discriminación por
  denoising en la batería final.

Conclusión: la **señal de entrenamiento es insuficiente**. El objetivo MLM,
masking, curriculum, multi-resolution, ranking y contrastivo offline no
inyectan una distinción correcto/incorrecto a nivel de inferencia.

## 2. Opciones restantes (viabilidad honesta)

| Opción | Costo | Evidencia | Viabilidad |
|---|---|---|---|
| **A. LogicDiff-style inference scheduler** | Muy bajo | Paper mejora LLaDA-8B 22%→61% en GSM8K con cabeza 4.2M | **Alta**: no reentrena base; prueba si el problema es orden de generación. |
| **B. DAG-aware training (dLLM-Reason)** | Medio | Repo con abstracción sólida, pero requiere adaptar modelo propio | **Media-Alta**: entrena enmascaramiento según DAG OBS/HIP/PRED/EVID/CONC. |
| **C. Structured-reasoning tokens** | Medio-Alto | Ninguna garantía | **Media**: redefine el vocabulario con tokens de razonamiento, entrena desde cero. |
| **D. SCoTD: distilar CoT de teacher** | Medio | 125M-1.3B mejoran con trazas de teacher | **Media**: requiere teacher ecológico y miles de trazas. |
| **E. RL post-training (d1/PADRE/ReNCE/LLaDOU)** | Alto | Requieren base preentrenada decente y RL online | **Baja-Media**: con 155M base débil, el RL no tiene señal. |
| **F. HDLM multi-resolution** | Medio | Mejora coherencia semántica, no razonamiento | **Baja**: no ataca la señal de inferencia. |

## 3. Plan recomendado: A → B → C

### Fase A — LogicDiff para EcoReasoner (2-3 días, 1 GPU)

**Objetivo:** probar si el orden de desenmascaramiento es la barrera real.

**Idea:**
- Entrenar una cabeza ligera (~4M) sobre los hidden states del dLLM 155M
  (checkpoint v2 o V3.3).
- La cabeza clasifica cada token enmascarado en roles lógicos:
  `PREMISE` (OBS/HIP), `CONNECTIVE` (because/despite/therefore),
  `DERIVED` (PRED), `CONCLUSION` (CONC), `FILLER`.
- El scheduler desenmascara en orden de dependencia: PREMISE → CONNECTIVE →
  DERIVED → CONCLUSION → FILLER.
- Evaluar L3 con `suite_smoke_v2`.

**Datos:**
- Etiquetar tokens de esqueletos con reglas:
  - Tokens dentro de `[OBSERVACION]...[/OBSERVACION]` y
    `[HIPOTESIS]...[/HIPOTESIS]` → PREMISE.
  - Conectivas (`because`, `despite`, `therefore`, `thus`, `however`) →
    CONNECTIVE.
  - `[PREDICCION]` → DERIVED.
  - `[CONCLUSION]` → CONCLUSION.
  - Resto → FILLER.
- Entrenar la cabeza con cross-entropy sobre pares (token enmascarado, rol).

**Por qué es honesto:**
- No reentrena el dLLM, así que no puede atribuirse a overfitting del training.
- Si mejora L3, indica que la representación ya contiene señal y el problema
  era el orden de generación.
- Si no mejora, la representación del dLLM 155M no razona.

**Criterios de GO/NO-GO:**
- GO: L3 ≥ 0.55 con este scheduler.
- DIRECCIONAL: L3 0.52–0.54.
- NO-GO: L3 < 0.52 → pasar a Fase B.

### Fase B — DAG-aware training (5-7 días, 2-4 GPUs)

**Objetivo:** entrenar el dLLM a reconocer y respetar el DAG de etapas.

**Herramienta:** dLLM-Reason (`BDeMo/dLLM_Reason`).

**Modificaciones necesarias:**
- Crear un DAG de 5 niveles (OBS→HIP→PRED→EVID→CONC) con `TokenDAG.from_levels`.
- Integrar el modelo 155M/350M como subclase de `DiffusionLM`.
- Usar `DAG-aware training`: enmascarar tokens de niveles superiores con mayor
  probabilidad, forzando al modelo a predecir niveles inferiores primero.
- Entrenar desde cero o fine-tune desde el checkpoint v2.

**Riesgo:** el repo dLLM-Reason es prototipo; requiere arreglar imports,
`save_pretrained` y offset del prompt.

**Criterios:**
- GO: L3 ≥ 0.55.
- NO-GO: L3 < 0.52 → Fase C (última apuesta) o archivar.

### Fase C — Structured-reasoning tokens + SCoTD (10-15 días)

**Objetivo:** cambiar radicalmente el espacio de representación.

**Idea:**
- Añadir tokens especiales al vocabulario: `[OBS]`, `[HIP]`, `[PRED]`, `[EVID]`,
  `[CONC]`, `[MECH]`, `[CAUSE]`, `[SUPPORT]`, `[CONTRADICT]`.
- El corpus se convierte en "programas de razonamiento" cortos y estructurados
  en lugar de prosa libre.
- Entrenar desde cero con MLM + ranking L3.
- Opcional: destilar trazas de un teacher (SCoTD) para iniciar.

**Riesgo:** es un proyecto largo; la escala del corpus v5 puede no ser
suficiente.

## 4. Recursos necesarios

| Fase | GPU | Tiempo | Datos | Notas |
|---|---|---|---|---|
| A | 1× pro6000 | 2-3 días | esqueletos | Muy barata. |
| B | 2-4× pro6000 | 5-7 días | esqueletos + v5 | Requiere porte de dLLM-Reason. |
| C | 4-8× pro6000 | 10-15 días | v5 + trazas teacher | Ambiciosa. |

## 5. Límites éticos y científicos

- No reclamar "razonamiento" si solo mejora L2 o generación temática.
- L3 ≥ 0.55 y hold-out por seed/topic distinto son necesarios para GO.
- Si A, B y C fallan, el documento de archivado queda como resultado
  negativo publicable.

## 6. Primer paso concreto

Empezar **Fase A** inmediatamente:
1. Extraer hidden states del checkpoint v2/V3.3 sobre esqueletos enmascarados.
2. Etiquetar tokens con roles lógicos.
3. Entrenar cabeza 4M.
4. Implementar scheduler en `suite_smoke_v2` o un sampler propio.
5. Evaluar L0-L3.
