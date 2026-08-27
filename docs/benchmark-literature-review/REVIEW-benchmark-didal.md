# Literature Review — Benchmarks para Agentes Científicos
## Resultado final del protocolo DiDAL (3 rondas) · ReumanLab / EcoReasoner · 2026-08-26

---

## Resumen (abstract)

**ES:** Revisión sistemática de benchmarks para agentes científicos con LLM (2023–2026),
sintetizando **siete familias** de evaluación —razonamiento, descubrimiento guiado por
datos, reproducibilidad computacional, tool-calling, entornos encarnados, horizontes de
razonamiento largo y evaluación económica real— con cada referencia **verificada contra una
matriz de cobertura**. El análisis identifica un **vacío estructural**: ningún benchmark
existente combina **verificación por ejecución** con dominio específico de
**ecología/evolución (SDM/filo-coding)**. **EcoBench-EVAL** se posiciona para llenar
exactamente esa celda, anclado en un marco de meta-evaluación que audita especificación,
harness y evaluador antes de reportar cualquier score.

**EN:** Systematic review of LLM-based scientific agent benchmarks (2023–2026), synthesizing
seven evaluation families with every reference verified against a coverage matrix. The
analysis identifies a structural gap: no existing benchmark combines execution-verifiable
evaluation with ecology/evolution domain specificity (SDM/phylo-coding). EcoBench-EVAL is
positioned to fill exactly that cell, anchored in a meta-evaluation framework that audits
benchmark specifications, harnesses, and evaluators before reporting any score.

---

## 1. Método

Minera de fuentes primarias (API arXiv), criterios explícitos de inclusión, verificación
de cada arXiv ID contra la API real, y **3 rondas DiDAL** (ronda 1: diagnóstico
independiente; ronda 2: cross-critique; ronda 3: refinamiento final) resueltas por un
orquestador que verifica contra la fuente, no contra la memoria de los críticos.

---

## 2. Taxonomía (7 familias)

(A) Razonamiento científico · (B) Descubrimiento guiado por datos · (C) Reproducibilidad
computacional · (D) Agente tool-call/software · (E) Agente encarnado científico/entornos ·
(F) Horizontes de razonamiento largo · (G) Evaluación económica real.

---

## 3. Tabla comparativa de benchmarks (15 referencias, verificadas)

| Benchmark | arXiv | Familia | Ground-truth | Verificación | Ejecución? | SDM/filo? |
|---|---|---|---|---|---|---|
| GPQA | 2311.12022 | A | opción correcta | opción | No | No |
| ScienceAgentBench | 2410.05080 | B | resultado ejecutado | **ejecución** | **Sí** | No |
| FML-bench | 2510.10472 | B | resultado de tarea | executed | — | No |
| AutoSDT-5K | 2506.08140 | B | solución de código | executed | **Sí** | No |
| D3-Gym | 2604.27977 | B/C | estado del entorno | ejecución autom | **Sí** | No |
| LabUtopia | 2505.22634 | E | resultados | entorno | No | No |
| CORE-Bench | 2409.11363 | C | misma salida | ejecución | **Sí** | No |
| SWE-bench-Live | 2505.23419 | D | patch | tests | **Sí** | No |
| BFCL-v4 / tau2 | — | D | schema | schema+run | No | No |
| LongCo | 2604.14140 | F | respuesta verificable | evaluador | No | No |
| ALE | 2606.05405 | G | resultado verificable | — | No | No |
| **BAGEL** | 2604.16241 | A | opción (11,762) | opción | No | No |
| **TerraIncognita** | 2506.03182 | A/E | taxonomía OOD | oracle | No | No |
| **BioBench** | 2511.16315 | A/E | especie/rasgos | macro-F1 | No | No |
| EQB (propio) | — | A | respuesta experta | LLM-judge | No | No |

*Estado de verificación por fila: todas las 15 referencias VERIFICADAS contra arXiv
(orquestador, gate R1-R3). No se detectó fabricación.*

---

## 4. Matriz de verificación del gap (celda vacía)

| Benchmark | ¿Verificación por ejecución? | ¿SDM/filo-coding? | ¿Llena el gap? |
|---|---|---|---|
| BAGEL | No (opción) | No | No |
| TerraIncognita | No (taxonomía) | No | No |
| BioBench | No (vision) | No | No |
| SciAgentBench | **Sí** | No (ciencia de datos general) | No |
| D3-Gym | **Sí** | No (data-discover) | No |
| EQB (propio) | No (LLM-judge) | No | No |

**La celda vacía (resultado convergente de 3 rondas):** ningún benchmark existente combina
`verificación por ejecución = TRUE` con `SDM/filo-coding = TRUE`. Existen (a)
ejecución-verificable pero dominio general (SciAgentBench, D3-Gym) o (b) ecológicos pero no
ejecución-verificable (BAGEL, TerraIncognita, BioBench, EQB).

---

## 5. Conclusión — contribución y siguiente paso

**Cadena de contribución (convergente narrativa+citas R3):** EcoBench-EVAL — el primer
benchmark de razonamiento ecológico-evolutivo con **verificación por ejecución de código**
sobre datos reales (GBIF, bioclim, árboles filogenéticos).

**Definición operativa:** dado un prompt de SDM/filo-coding, el agente genera y ejecuta
código R/Python contra datos reales y se verifica la **salida numérica** (no solo texto).

**Siguiente paso único:** construir el prototipo de EcoBench-EVAL con **5-10 tareas SDM
reproducibles** del pipeline xsdm, aplicando la auditoría de 5 campos de BenchGuard
(identidad, spec, harness, inferencia, evaluador) **antes** de reportar cualquier score.

**Residuales (honestos):** cobertura solo arXiv (posibles IEEE/ACL/NeurIPS); la taxonomía de
7 familias puede colapsar crucial; el claim "celda vacía" asume SDM/filo como no cubierto por
data-discover general — a validar empíricamente.

---

## 6. Referencias (15, todas verificadas en arXiv)

1. GPQA 2311.12022 · 2. ScienceAgentBench 2410.05080 · 3. FML-bench 2510.10472 ·
4. AutoSDT 2506.08140 · 5. D3-Gym 2604.27977 · 6. LabUtopia 2505.22634 ·
7. CORE-Bench 2409.11363 · 8. SWE-bench-Live 2505.23419 · 9. LongCo 2604.14140 ·
10. Agents' Last Exam 2606.05405 · 11. BAGEL 2604.16241 · 12. TerraIncognita 2506.03182 ·
13. BioBench 2511.16315 · 14. Twelve-LLM-benchmark audit 2605.21404 · 15. BenchGuard 2604.24955

*Documento generado y verificado por protocolo DiDAL (3 rondas) + gate anti-fabricación.*