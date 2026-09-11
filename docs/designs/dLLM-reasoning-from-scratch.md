# dLLM reasoning from scratch — Replantamiento

> Fecha: 2026-09-10.
> Contexto: el dLLM puro ha fallado 6+ veces en L3 (STAGE_GRAMMAR). V3.3
> (contrastivo) está corriendo como último intento de la familia MLM-ranking
> (job 29183112). Este documento replantea cómo un dLLM podría razonar desde cero.

## Resumen ejecutivo

- **Diagnóstico:** el dLLM aprende estructura de etapas (L2) pero no contenido
  inferencial (L3). La causa probable es la **señal de entrenamiento**, no la
  arquitectura.
- **V3.3 lanzado** (job 29183112) como experimento paralelo. Es la intervención
  más directa: cambiar MLM por ranking contrastivo sobre pares correcto/incorrecto.
- **DiDal completado** (3 rondas, 4 críticos + orquestador).
- **Ruta única:**
  1. V3.3 → si L3 ≥ 0.55: GO, escalar.
  2. Si 0.52 ≤ L3 < 0.55: V3.2 role-aware + curriculum.
  3. Si L3 < 0.52 en ambos: archivar dLLM-puro, pivotar a controller/verificator.
- **Veredicto DiDal por opción:** A = GO, B = REVISE, C = ARCHIVE, D = REVISE/GO condicional.

---

## 1. Diagnóstico consolidado

### Evidencia acumulada

| Run | L0 | L1 | L2 | L3 | Patrón |
|---|---|---|---|---|---|
| Piloto v2 | 0.50 ns | 0.47 ns | **0.61*** | 0.51 ns | STAGE_GRAMMAR |
| no-whole-stage | 0.55 * | 0.51 ns | **0.60*** | 0.48 ns | STAGE_GRAMMAR |
| weight-tying | 0.50 ns | 0.48 ns | **0.63*** | 0.50 ns | STAGE_GRAMMAR |
| RoPE | 0.54 * | 0.50 ns | **0.60*** | 0.47 ns | STAGE_GRAMMAR |
| 50M | 0.50 ns | 0.46 ns | **0.56** | 0.48 ns | STAGE_GRAMMAR |
| V3.1 curriculum | 0.54 * | 0.49 ns | **0.63*** | 0.50 ns | **STAGE_GRAMMAR** |
| V3.3 contrastivo | — | — | — | — | en evaluación (job 29183112) |

### Qué sabemos

1. **El objetivo MLM no premia la inferencia**. Aprende a rellenar tokens localmente.
2. **La loss baja no implica razonamiento** (RoPE: loss 6.9, L3 0.47).
3. **El orden de etapas (L2) se aprende robustamente**, pero el contenido (L3) no.
4. **La batería v3 cerró el escape del test**; los negativos son semánticamente duros.
5. **El corpus skeleton puede ser demasiado abstracto** para relaciones ecológicas concretas.
6. **V3.1 (curriculum + role-aware) confirma el patrón STAGE_GRAMMAR**: L3 = 0.5043
   (ns), a pesar de 10K steps y loss ~7.3. La curriculum no rompió el techo.

### Diagnóstico raíz

El problema es la **señal de entrenamiento**, no la arquitectura. El objetivo MLM
optimiza perplejidad puntuual: predice un token enmascarado dado el resto. Para
eso, el modelo solo necesita estadísticas de coocurrencia local. No necesita
saber que `increased` es científicamente correcto y `decreased` incorrecto en un
contexto dado; solo necesita saber cuál token es más probable localmente.

---

## 2. Cuatro opciones de replanteamiento

### A. Cambiar el objetivo de entrenamiento

Hipótesis: el problema no es el modelo ni el masking, sino que la pérdida no
pide explícitamente que el modelo discrimine inferencias válidas de inválidas.

| Paper | Clave | Relevancia |
|---|---|---|
| d1 (arXiv 2504.12216) | masked SFT + diffu-GRPO para dLLMs | Post-entrenamiento con RL para razonar. Costoso (rollouts online). |
| PADRE (EACL 2026, arXiv 2605.05226) | pseudo-likelihood para dLLM reasoning | Alternativa a GRPO, más estable; sigue siendo post-entrenamiento. |
| ReNCE (arXiv 2601.22432) | NCE sobre K soluciones correctas/incorrectas | Simple, sin red crítica; aplica a pares de razonamiento. |
| Math-Shepherd (arXiv 2312.08935) | PRM sin anotación humana | Verificador de proceso a partir de rollouts y recompensas verificables. |
| Counterfactual Distillation (EMNLP 2024) | generar contrafactuales con LLM para SFT | Mejora SLM; datos arquitectura-agnósticos. |
| Beyond Random Sampling (EACL 2026) | curriculum learning para pretraining | Ordenar datos por dificultad; ortogonal. |

**Hipótesis:** MLM estándar (Sahoo et al. 2024) equivale a una mezcla de pérdidas
puntuales sin supervisión de secuencia/respuesta. El objetivo más prometedor para
nuestra escala es **contrastivo/ranking sobre pares correcto/incorrecto**, con
curriculum de dificultad. Process/outcome supervision es teóricamente superior
pero más costoso.

### B. Cambiar la representación

Hipótesis: la granularidad token-level espeja la tarea. Necesitamos
representaciones semánticas intermedias (latentes, grafos, bloques de
razonamiento).

| Paper | Clave | Relevancia |
|---|---|---|
| LaDiR (arXiv 2510.04573) | VAE + latent diffusion sobre thought tokens | Razonamiento en bloques semánticos; costoso (VAE+diffusion). |
| Coconut (arXiv 2412.06769) | razonar en last hidden state continuo | Latent feedback; inestable, requiere LLM base capaz. |
| DoT (NeurIPS 2024, arXiv 2402.07754) | diffusion de thoughts en hidden space | Diffusión continua; no masked discrete. |
| VDLM (arXiv 2602.15870) | masked diffusion sobre embeddings de variables semánticas | Variable diffusion; conceptualmente aplicable. |
| TreeDiff (arXiv 2508.01473) | masking sintáctico por AST para código | Estructura de corrupción; adaptar a árboles de razonamiento. |
| LogicDiff (arXiv 2603.26771) | desenmascarar por rol lógico en MDLM | Inference-time; mejora razonamiento sin reentrenar. |

**Hipótesis:** La granularidad token-level diluye la señal. Operar sobre unidades
semánticas (bloques de razonamiento, variables) puede ayudar, pero un VAE
completo es riesgoso a 155M/300K docs. El camino más barato es **MDLM con
corrupción estructurada por rol lógico** (inspirado en LogicDiff/TreeDiff).

### C. Arquitecturas híbridas

Hipótesis: un solo modelo no puede ser a la vez generador fluido y verificador
preciso. Separar componentes.

| Paper | Clave | Relevancia |
|---|---|---|
| LLM² (arXiv 2412.20372) | LLM generador + verifier System 2 | Mejora Llama3-1B GSM8K 50.3→57.8; dos componentes. |
| LM² (EMNLP 2024) | decomposer + solver + verifier | Modular, pero tres modelos; pesado. |
| GenRM (arXiv 2408.15240) | verificador generativo entrenado con NTP | Mejora verificadores; compatible con LLM. |
| RL Tango (arXiv 2505.15034) | entrenar generador y verificador juntos con RL | Co-evolución; inestable/costoso. |
| Think First, Diffuse Fast (arXiv 2603.13243) | AR pequeño planifica, dLLM genera | Training-free híbrido; AR razonador + dLLM fluidez. |
| STAR-LDM (arXiv 2602.20528) | AR + difusión latente de planificación | Planificación semántica antes de decodificación. |

**Hipótesis:** El razonamiento científico se beneficia de separar generación de
verificación. Para nuestros recursos, la opción más viable es **two-tower con
backbone compartido de 155M usado como generador y verificador generativo
(GenRM/LLM² style)**.

### D. Cambiar los datos

Hipótesis: no hay suficientes repeticiones de relaciones causales con variación
léxica. Necesitamos corpus densos, contrafactuales, o anotaciones estructuradas.

| Paper | Clave | Relevancia |
|---|---|---|
| Argument-Annotated Corpus (W18-5206) | ADUs y relaciones support/attack en papers | Esquema de anotación aplicable; corpus manual pequeño. |
| DISCO (ACL 2023, arXiv 2212.10534) | generar contrafactuales con LLM a escala | +6% robustez, +10% consistencia en NLI. |
| TinyStories (arXiv 2305.07759) | corpus sintético denso para SLMs | Modelos <10M aprenden estructura con datos controlados. |
| Don't Stop Pretraining (ACL 2020) | DAPT/TAPT mejora dominio científico | Refuerzo, no solución primaria. |
| SCoTD (ACL 2023, arXiv 2306.14050) | CoT distillation para modelos 125M-1.3B | 155M es viable con datos denso en CoT. |
| Plausible Negative Samples (arXiv 2602.03516) | negativos casi correctos para DPO | Hard negatives son el ingrediente clave. |

**Hipótesis:** El problema es densidad/señal. El corpus skeleton aumenta densidad
pero sigue siendo un único camino. Necesitamos **múltiples ejemplos
contrastivos por transición inferencial**. La opción más barata es usar el
esqueleto existente + pares hard-negative (L2/L3) con objetivo contrastivo.

---

## 3. Evaluación de viabilidad por opción

| Opción | Costo GPU | Costo datos | Riesgo | Primer paso recomendado |
|---|---|---|---|---|
| A. Contrastivo/ranking | Bajo (4-10h) | Bajo (pares existentes) | Puede seguir sin señal si los pares son débiles | V3.3 (en vuelo) |
| B. MDLM + masking estructurado | Medio (6-12h) | Bajo (mismos datos) | Role labeling puede ser imperfecto | LogicRole MDLM |
| C. Two-tower GenRM | Medio (8-16h) | Medio (pares verificación) | Verificador puede aprender atajos | Generador + verifier 155M |
| D. Datos contrafactuales denso | Alto (generación) | Alto (LLM/API) | Calidad de contrafactuales | DISCO sobre esqueletos |

---

## 4. Secuencia de experimentos propuesta (Go/No-Go)

### Fase actual (inmediata)
1. **V3.3 contrastivo** (job 29183112). Si L3 ≥ 0.55 → GO, escalar.
2. Esperar **V3.1 curriculum** verdict.

### Si V3.3 y V3.1 fallan (L3 < 0.52)
3. **Opción B mínima: MDLM con masking estructurado por rol lógico**. Modificar
   `build_mask_indices` para enmascarar/proteger conectivas, verbos de relación,
   entidades y números. 1 run de 10K steps.
4. **Opción C mínima: two-tower generador+verificador**. Usar el mismo backbone
   155M como generador y verificador generativo sobre pares L3. 1 run de 3K-5K
   steps.

### Si B y C fallan
5. **Opción D controlada**: generar contrafactuales con LLM sobre esqueletos
   (estilo DISCO/PNS) y entrenar con ranking. 1 run.
6. Si D falla → **archivar dLLM-puro**, pivotar a controller/verificator.

---

## 5. Preguntas abiertas centrales

1. ¿Es posible que un dLLM puro de ~150M aprenda a razonar con el objetivo
   correcto, o la escala es insuficiente?
2. ¿El razonamiento debe estar en el mismo modelo o en un verificador separado?
3. ¿Qué nivel de representación es el adecuado: tokens, latentes, grafos,
   variables?
4. ¿Qué corpus y qué densidad de relaciones necesitamos?
5. Si V3.3 contrastivo falla, ¿cuál es el siguiente paso mínimo viable?

---

## 6. DiDal — Rondas de discusión multiagente

### Rol de cada participante

- **narrative**: defiende la opción, argumenta por qué es la solución.
- **critic**: expone obstáculos, costos, riesgos y atajos.
- **evidence**: verifica afirmaciones contra literatura y datos del proyecto.
- **orchestrator**: sintetiza, asigna `go / revise / archive`, propone experimento.

### Ronda 1 — Investigación independiente (completada)

Cuatro agentes exploraron opciones A, B, C, D por separado. Sus informes
fueron integrados en las secciones 2-4 de este documento.

### Ronda 2 — Cross-critique (completada)

Cada agente revisó las otras tres opciones y emitió veredicto.

| Crítico | Opción defendida | Veredicto | Crítica central a las otras |
|---|---|---|---|
| A | A | GO | B no cambia la pérdida, C duplica complejidad, D es caro y no cambia objetivo. |
| B | B | GO condicional | A sigue en token-level, C divide capacidad, D no aumenta ejemplos. |
| C | C | GO condicional | A mantiene modelo monolítico, B depende de role-labeler, D no cura arquitectura. |
| D | D | GO condicional | A y C dependen de pares débiles, B no aumenta densidad de relaciones. |

Puntos convergentes:
- El problema es la señal de entrenamiento (L3 ausente, L2 robusto).
- V3.3 (contrastivo) es el experimento más directo y barato.
- Si L3 < 0.52, la línea dLLM-pura debe archivarse o pivotar.

Disputa central: ¿la causa es el **objetivo** (A), la **representación** (B), la
**arquitectura** (C) o los **datos** (D)?

### Ronda 3 — Refinamiento final (completada)

**Orquestador — Recomendación final:**

| Opción | Veredicto | Prioridad |
|---|---|---|
| A. Contrastivo/ranking | **GO condicional** | 1 (inmediata) |
| B. MDLM + masking por rol lógico | **REVISE** | 2 (si A es direccional) |
| C. Two-tower generador+verificador | **ARCHIVE** | — |
| D. Datos contrafactuales | **REVISE/GO condicional** | 3 (reserva) |

**Ruta única propuesta:**

1. **V3.3 contrastivo** (job 29183112, ya en vuelo).
   - **GO** si L3 ≥ 0.55, L2 ≥ 0.55, rank_acc > 0.60 → escalar a 10K steps.
   - **DIRECCIONAL** si 0.52 ≤ L3 < 0.55 → V3.2 role-aware + curriculum.
   - **NO-GO** si L3 < 0.52 → V3.2 como última prueba; si falla, pivotar.
2. **V3.2 role-aware + curriculum** (si A es direccional/falla).
   - **GO** si L3 ≥ 0.55 con ganancia ≥ +0.04 sobre baseline y L2 ≥ 0.58.
   - **NO-GO** si L3 < 0.52 → archivar dLLM-puro.
3. **Pivot a controller/verificator** si A+B fallan.
   - Usar el verificador ya validado (98.2% match_args) como solución productiva.
   - Opción D (contrafactuales) solo si queda presupuesto y se filtra calidad ≥ 80%.
