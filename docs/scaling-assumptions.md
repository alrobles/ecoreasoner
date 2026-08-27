# Supuestos de escalado para el dLLM-MoE "Agentic Scientist" (EcoReasoner)

Fecha: 2026-08-26
Autores: A.L. Robles Fernández (con soporte de Hermes Agent)
Estado: supuestos de trabajo — para re-validar empíricamente con el benchmark de Fase B.

## 1. Por qué no aplicamos el ratio Chinchilla "20 tokens/parámetro" como umbral

Chinchilla (Hoffmann et al., 2022) estableció que para **LLM autorregresivos
genéricos entrenados sobre texto web**, el entrenamiento compute-óptimo se da con
~20 tokens por parámetro (relación casi lineal N ∝ D). Esta cifra es una
**referencia empírica de la industria**, no una ley universal, y en nuestro caso
se desvía por **tres factores independientes** que la vuelven poco vinculante:

| Desvío | Implicación |
|---|---|
| **1. No es autorregresivo (es dLLM masked-diffusion)** | El objetivo de loss (denoising / MLM, tipo LLaDA/MDLM) no es next-token causal; la cuenta de FLOPs y la constante 20:1 no están validadas para difusión discreta. |
| **2. Es de dominio científico-ecológico** | 1 token de ecología curado empaca más señal que 1 token de web genérica para un modelo *de ecología*. La "efectividad" por token es mayor. |
| **3. Es un agente, no un generador pasivo** | El objetivo es razonamiento instrumental / tool-call / decisión, no pérdida de lenguaje. El modo de entrenamiento dominante es **destilación de razonamiento desde el teacher**, no pretraining en bruto. |

No hay, hasta donde sabemos, un "Chinchilla-diffusion" ni un scaling estabulado
para *agentes* publicado. Por eso el presupuesto de tokens NO se decide por una
constante a priori, sino **empíricamente** con métricas de tareas de agente.

## 2. Factores de eficiencia de corpus (heurística razonada, ajustable)

Para convertir el tamaño bruto del corpus en "tokens efectivos" de referencia
web-genérica (base Chinchilla), aplicamos factores por fuente/tipo de dato:

| Factor | Valor | Justificación |
|---|---|---|
| `F_dominio` | ×2.0 | Texto científico ecológico vs web genérica para un modelo de ecología. Conservador; podría ser mayor en dominios de nicho. |
| `F_fulltext` | ×1.3 | Full-text (PMC/EcoEvoRxiv) > solo abstract: más contexto y densidad. |
| `F_destilación` | ×3.0 | Tokens generados/guiados por un teacher experto (v4-flash) rinden más que pretraining en bruto (patrón BabyLM/Phi: curado + sintético + destilado → alta eficiencia de muestra). |

### Resultado aplicado al corpus (`train_corpus_v3`, ~1.71 B tokens)

| Modo | Tokens efectivos | vs. óptimo Chinchilla (5.58 B para 279 M activos) |
|---|---|---|
| Pretraining directo | ~4.34 B | **0.78×** (cerca del óptimo) |
| Con destilación (×3) | ~13.0 B | **2.34×** (supera el óptimo) |

**Lectura operativa:** con la calidad/dominio + destilación, el corpus NO es
escaso; es suficiente (e incluso por encima del óptimo) para 279 M params activos.
El "déficit Chinchilla" (0.31×) calculado sobre el ratio crudo 20:1 es artefacto
de comparar texto de dominio contra la referencia de web genérica.

## 3. Referencias que respaldan la desviación por calidad/data

- **Chinchilla / Hoffmann et al. (2022)** — *Training Compute-Optimal Large
  Language Models* (NeurIPS; arXiv 2203.15556). La referencia base del 20:1.
  El propio paper nota: *"we expect scaling to larger datasets is only
  beneficial when the data is high-quality"* — la calidad es condición del
  beneficio de escala.
- **Phi-3 (Abdin et al., 2024)** — un 3.8B con datos *curados + sintéticos* iguala
  a Mixtral-8x7B y roza GPT-3.5, estando muy por debajo del "óptimo" Chinchilla
  en volumen. Muestra que la **calidad/curation desplaza el óptimo hacia menos
  tokens por parámetro**.
- **Baby Llama (Warstadt et al., 2023 / arXiv 2308.02019)** — destilar un
  ensemble de teachers sobre un dataset pequeño (10M words) produce un modelo que
  **supera a sus teachers con mayor eficiencia de muestra**. Respalda el factor
  de destilación para datasets chicos/de dominio.
- **"Prescriptive Scaling Laws for Data-Constrained Training" (arXiv 2605.01640)** —
  scaling laws para regímenes con *datos limitados*, donde el ratio de Chinchilla
  (que asume tokens únicos) no describe bien la asignación; el óptimo se desplaza
  hacia más epochs / menos data cuando el dato es el cuello.
- **Muennighoff et al. (2023)** — *Scaling Data-Constrained Language Models*:
  cada repetición de tokens decae su valor marginal; respalda iterar sobre corpus
  curto en vez de exigir volumen bruto.
- **Robustez de Chinchilla (arXiv 2509.23963)** — confirma que el 20:1 es robusto
  *para el régimen web-LM causal*, reforzando que es una referencia de ese régimen
  y no una ley para dLLM/dominio/agente.

## 4. Implicación para el MVP

- El presupuesto de tokens del prototipo se fija por **medición en tareas de
  agente (Fase B)**, no por Chinchilla.
- Fase A (pretraining base refinado sobre v3) cubre el "esqueleto de dominio";
  Fase B (destilación de razonamiento/tool-call desde v4-flash) aporta la mayor
  parte de la efectividad.
- Los factores de la §2 se tratarán como **supuestos documentados (v1)** y se
  recalibrarán contra el benchmark de agente una vez tengamos resultados.

## 5. Nota de reproducibilidad

Este documento forma parte de la "receta abierta" del proyecto: los supuestos
deben ser explícitos y trazables para revisores y para quien replique el pipeline
de principio a fin.
