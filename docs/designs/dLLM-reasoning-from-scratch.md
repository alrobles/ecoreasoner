# dLLM reasoning from scratch — Replantamiento

> Fecha: 2026-09-10 (sesión post-V3.1/V3.2).
> Contexto: el dLLM puro ha fallado 6+ veces en L3 (STAGE_GRAMMAR). V3.3
> (contrastivo) está corriendo como último intento de la familia MLM-ranking.
> Este documento replantea cómo un dLLM podría razonar desde cero.

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
| V3.1 curriculum | — | — | — | — | en evaluación |
| V3.3 contrastivo | — | — | — | — | en evaluación (job 29183112) |

### Qué sabemos

1. **El objetivo MLM no premia la inferencia**. Aprende a rellenar tokens localmente.
2. **La loss baja no implica razonamiento** (RoPE: loss 6.9, L3 0.47).
3. **El orden de etapas (L2) se aprende robustamente**, pero el contenido (L3) no.
4. **La batería v3 cerró el escape del test**; los negativos son semánticamente duros.
5. **El corpus skeleton puede ser demasiado abstracto** para relaciones ecológicas concretas.

---

## 2. Cuatro opciones de replanteamiento

### A. Cambiar el objetivo de entrenamiento

Hipótesis: el problema no es el modelo ni el masking, sino que la pérdida no
pide explícitamente que el modelo discrimine inferencias válidas de inválidas.

Candidatos:
- Ranking contrastivo sobre pares (V3.3 ya en vuelo).
- Outcome / process supervision (PRMs).
- Verifiers entrenados con pares sintéticos.
- Counterfactual distillation.
- Curriculum de dificultad de razonamiento (fácil → difícil).

### B. Cambiar la representación

Hipótesis: la granularidad token-level espeja la tarea. Necesitamos
representaciones semánticas intermedias (latentes, grafos, bloques de
razonamiento).

Candidatos:
- Latent thought tokens / LaDiR.
- Diffusion sobre variables semánticas.
- Graph-conditioned diffusion.
- Syntax/structure-aware dLLM.

### C. Arquitecturas híbridas

Hipótesis: un solo modelo no puede ser a la vez generador fluido y verificador
preciso. Separar componentes.

Candidatos:
- dLLM generador + verificador/torre discriminativa.
- LLM² / System 1 + System 2.
- Decomposer + solver + verifier.
- Generative verifiers (GenRM).
- Controller/verificator (Opción D del proyecto).

### D. Cambiar los datos

Hipótesis: no hay suficientes repeticiones de relaciones causales con variación
léxica. Necesitamos corpus densos, contrafactuales, o anotaciones estructuradas.

Candidatos:
- Corpus eco-fino con relaciones reales (especies, regiones, GBIF).
- Generación sintética de argumentos ecologicos controlados.
- Counterfactual augmentation (DISCO, etc.).
- Curriculum de datos por relación (repetir relación con variación léxica).
- Hard negatives curriculares.

---

## 3. Fuentes preliminares (a expandir con subagentes)

### A. Objetivos de entrenamiento

- **MERIt** (arXiv 2203.00357): meta-path guided contrastive learning para
  logical reasoning. Genera pares de instancias positivas/negativas editando
  relaciones en meta-paths.
- **Let's Verify Step by Step** (OpenAI, arXiv 2305.20050): process supervision
  vence outcome supervision en MATH.
- **LLM²** (arXiv 2412.20372): generator + process-based verifier, mejora
  Llama3-1B en GSM8K de 50.3 a 57.8.
- **Teaching Small Models to Reason through Counterfactual Distillation**
  (EMNLP 2024): genera counterfactuals con LLM y enseña SLM.

### B. Representaciones estructuradas / latentes

- **LaDiR** (arXiv 2510.04573): latent diffusion sobre thought tokens. Mejora
  razonamiento matemático y planning.
- **UTGDiff** (arXiv 2408.09896): graph-conditioned diffusion para moléculas.
- **Form follows Function** (arXiv 2311.00444): LLM fine-tuneado para generar
  grafos desde texto funcional.

### C. Arquitecturas híbridas

- **LLM²** (arXiv 2412.20372): System 1 (LLM) + System 2 (verifier).
- **LM²** (EMNLP 2024): decomposer + solver + verifier.
- **Generative Verifiers** (arXiv 2408.15240): entrenar verificadores con NTP.
- **LLaDA / iLLaDA**: dLLM como base, SFT/RL para tareas.

### D. Datos / contrafactuales

- **DISCO** (ACL 2023): distilling counterfactuals with LLMs.
- **Dually Self-Improved Counterfactual Data Augmentation** (ACL 2025).
- **Reasoning Elicitation via Counterfactual Feedback** (ICLR 2025).
- **Domain-adaptive pretraining** (INDUS, MatSci, OmniScience).

---

## 4. Espacio para rondas DiDal

Las rondas DiDal se ejecutarán como discusión multiagente sobre cada opción,
con roles:

- **narrative**: argumenta a favor de la opción, plantea caso de uso.
- **critic**: expone obstáculos técnicos, costos y riesgos.
- **evidence**: verifica afirmaciones contra literatura/fuentes.
- **orchestrator**: sintetiza veredicto y recomendación operativa.

Resultado esperado: para cada opción un veredicto
`revise / go / archive` y un experimento concreto con Go/No-Go.

---

## 5. Preguntas abiertas centrales

1. ¿Es posible que un dLLM puro de ~150M aprenda a razonar con el objetivo
   correcto, o la escala es insuficiente?
2. ¿El razonamiento debe estar en el mismo modelo o en un verificador separado?
3. ¿Qué nivel de representación es el adecuado: tokens, latentes, grafos,
   variables?
4. ¿Qué corpus y qué densidad de relaciones necesitamos?
5. Si V3.3 contrastivo falla, ¿cuál es el siguiente paso mínimo viable?
