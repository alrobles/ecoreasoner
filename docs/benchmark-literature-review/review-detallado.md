# Literature Review — Benchmarks para Agentes Científicos (soportado en la literatura)

> ReumanLab · EcoReasoner · 26 de agosto de 2026
> Revisión basada en minería de fuentes primarias (API arXiv, abstracts reales) + verificación
> de IDs. Documento listo para 3 rondas DiDAL.

---

## 1. Alcance y metodología

Se revisó la literatura de *benchmarks para agentes de lenguaje (LLM) orientados a
ciencia*. La selección prioriza fuentes primarias revisadas en arXiv (2023–2026) y cubre
tres ejes: **razonamiento científico**, **ejecución de tareas de descubrimiento guiado por
datos (data-driven discovery)** y **verificación objetiva del resultado**. Se excluyen
actualmente lo puramente conversacional; se concentra en capacidad agentica y en
grability de un modelo de dominio (ecología/evolución).

Criterios de inclusión:
- Son benchmarks de uso general (no de un único equipo), con artefacto de evaluación o
  método reproducible, y citados/revisados con firma primaria (arXiv, OpenAgent).
- Se registra para cada uno: identidad (arxiv ID), dominio, formato, ground-truth,
  verificación, escala, y relevancia a ecología/evolución.

---

## 2. Taxonomía sintetizada de la literatura

La literatura converge en distinguir **siete** familias (antes cinco); la ampliación a
siete captura dos flancos nuevos de 2025-2026: la **meta-evaluación** y los **horizontes de
razonamiento largo**.

| Familia | Qué mide | Referencias | Verificación |
|---|---|---|---|
| (A) Razonamiento científico (pregunta-respuesta) | conocimientos + razonamiento | GPQA, LongCoTc | opción correcta / respuesta verificable |
| (B) Descubrimiento guiado por datos (agente) | plan + ejecución de pipeline sobre datos | ScienceAgentBench, AutoSDT, FML-bench, D3-Gym | artefacto/resultado ejecutado, entorno |
| (C) Reproducibilidad computacional | reproducir resultados de un estudio | CORE-Bench | ejecuta y da el mismo resultado |
| (D) Agente tool-call / software | función correcta + formato + issues | tau-bench, BFCL-v4, SWE-bench-Live | esquema + ejecución |
| (E) Agente encarnado científico / entornos | laboratorio físico/químico, largo-horizonte | LabUtopia | simulación, entorno estado |
| (F) Horizontes de razonamiento largos | CoT multi-paso, planificación | LongCoT | respuesta verificable de problema largo |
| (G) Evaluación "económica real" | tasks laborales de largo-horizon verificables | Agents' Last Exam | outcomes verificables industriales |

Una novedad importante de la literatura 2026 es **la meta-evaluación**: dos papers
(BenchGuard; audit "Twelve LLM Agent Benchmark Papers") señalan que **muchos fallos de
agentes son fallos del propio benchmark** (spec rota, asunciones implícitas, scripts que
penalizan soluciones válidas). Implicación: la credibilidad de una afirmación "ganamos
benchmark X" requiere auditar el benchmark (spec, harness, sampling, evaluator), no solo
leer el score. Es el gate que el DiDAL ya implementa.

---

## 3. Tabla comparativa de benchmarks (datos de la literatura)

| Benchmark (arXiv) | Familia | Formato | Ground truth | Verificación | Escala | Relevancia Ecología/Evol | Nicho | Costo |
|---|---|---|---|---|---|---|---|---|
| GPQA (2311.12022) | A | MCQ | respuesta correcta | opción | 448 (Diamond 198) | Bajo (bio-mol/med) | ajeno | alto |
| ScienceAgentBench (2410.05080) | B | tareas data-driven | resultado artefacto | ejecución | 102 tareas | Media (ciencia de datos) | parcial | medio |
| FML-bench (2510.10472) | B | investigación en ML | resultado de tarea | executed | 8 tareas | Bajo | — | alto |
| AutoSDT (2506.08140) | B | data-driven coding tasks | solución código | executed | 5404 tareas | Media (datos) | parcial | medio |
| D3-Gym (2604.27977) | B/C | entornos verificables | estado del entorno | ejecución autom | 565 tasks / 239 repos | Media | parcial | alto |
| CORE-Bench (2409.11363) | C | reproducibilidad | mismo resultado | ejecución | — | Media | parcial | medio |
| LabUtopia (2505.22634) | E | laboratorio simulado | resultados | entorno | — | Baja | no | alto |
| SWE-bench-Live (2505.23419) | D | issues código | patch correcto | tests ejecutados | — | Baja | no | medio |
| BFCL-v4 | D | tool-calling | schema válido | schema+run | — | media (tool) | parcial | medio |
| tau2-bench | D | tool-use | correcto+formato | schema+corr | — | media (tool) | parcial | medio |
| LongCoT (2604.14140) | F | CoT largo | respuesta verificable | evaluador | 2500 problemas | Baja | química/maths/CS | medio |
| ALE (2606.05405) | G | laboral largo-horizon | resultado verificable | — | 250+ expertos | Baja | registro/trabajo | alto |
| EQB (propio) | A | MCQ semántico | respuesta experta | LLM-judge | 46 | **Alta** | CORE ecología | 0 |
| BAGEL (2604.16241) | A | MCQ closed-book | especie/taxonomía | opción | 11,762 | **Alta** | conocimiento animal | medio |
| TerraIncognita (2506.03182) | A/E | species discovery | especie/taxonomía OOD | opción+multimodal | 200+100 | **Alta** | biodiversidad | alto |
| BioBench (2511.16315) | A/E | visión ecológica | especie/rasgos/behavior | macro-F1 | 9 tasks·3.1M img | **Alta** | ML ciencia visual | medio |

## 3.b Matriz de verificación del gap (resolución DiDAL Ronda 2)

| Benchmark | ¿Verificación por ejecución? | ¿SDM/filo-coding? | ¿Llena el gap? |
|---|---|---|---|
| BAGEL | No (opción) | No | No |
| TerraIncognita | No (taxonomía/hierárquica) | No | No |
| BioBench | No (macro-F1 visión) | No | No |
| ScienceAgentBench | **Sí** | No (ciencia de datos general) | No |
| D3-Gym | **Sí** | No (data-discover general) | No |
| EQB (propio) | No (LLM-judge) | No | No |

**La celda vacía (el gap real):** ningún benchmark combina
`verificación por ejecución = TRUE` con `SDM/filo-coding = TRUE`.
Existen (a) ejecución-verificable pero dominio general (SciAgentBench, D3-Gym) o
(b) ecológicos pero no ejecución-verificable (BAGEL, TerraIncognita, BioBench, EQB).
EcoBench-EVAL llenará exactamente esa celda: ecología+evolución con ejecución-verificada.

---

## 4. Análisis de la literatura clave (extractos verificados)

### 4.1 ScienceAgentBench (2410.05080)
Llamado a una evaluación **rigurosa por tarea** dentro de un flujo de trabajo científico,
antes de afirmar "automatización de extremo a extremo". Extrajo **102** tareas de datos
reales con programas de investigación. Crítica: separar competencia por tarea, no slogan de
fin-a-fin.

### 4.2 FML-bench (2510.10472)
Señala que los benchmarks existentes adoptan una vista **orientada a la ingeniería**
(performance final, coste cómputo) y **pasan por alto el proceso de investigación** de los
agentes. Propone **8** tareas de ML research que evalúan el proceso, no solo el resultado.

### 4.3 AutoSDT (2506.08140) y AutoSDT-5K
Construye automáticamente **5,404** tareas de código de descubrimiento en flujos de trabajo
reales, aprovechando la capacidad de codificación del LLM para buscar y sintetizar
instrucciones y soluciones. Aborda la **escasez de datos** de entrenamiento/evaluación en
ciencia de datos abierta.

### 4.4 D3-Gym (2604.27977)
Primer dataset **automáticamente** con **entornos verificables** para descubrimiento guiado
por datos: **565** tareas de **239** repositorios reales en 4 disciplinas, cada una con
instrucción natural, entorno ejecutable con dependencias preinstalados y **dataset de
entrada**. Es la referencia más cercana al tipo de entorno ejecutable que este review
recomienda construir.

### 4.5 GPQA (2311.12022)
Criterio duro: **448** MCQ graduado en biología, física, química. Expertos con PhD en campo
llegan a **65% (74% descontando errores)**; no-expertos a **34%** aunque con web. Es el
"google-proof". Líder en OpenRouter (ago-2026): Gemini 3.1 Pro **94.3%** — no es de un
laboratorio de ecología, y no es ganable para un modelo de dominio ecológico (dominio
ajeno, techo ajustado).

### 4.6 CORE-Bench (2409.11363)
Mide **reproducibilidad computacional**: reproducir el resultado de un estudio con el código
y datos dados. Más que la exactitud de answers, contrata la credibilidad. Relevancia
directa a la filosofía de verificación del laboratorio.

### 4.7 SWE-bench-Live (2403.23419)
Señala **riesgos de sobreajuste y contaminación** de los benchmarks estáticos (SWE-bench no
publicaba updates). La "práctica recomendada" es **live-updatable**, y **no estár** estático:
evita "memorizar el examen".

### 4.8 Meta-evaluación: BenchGuard (2604.24955) y "Twelve LLM Agent Benchmarks"
**BenchGuard** propone auditar con LLM frontier la infraestructura de evaluación (specs
rotas, asum implícitas, scripts rígidos). El **audit de 12 papers** introduce un **esquema de
reproducibilidad de 5 campos** (identidad del benchmark, especificación del harness,
configuración de inferencia, subset, versión del evaluador). Menor.

### 4.9 Agents' Last Exam (2606.05405)
Reclama que los **benchmarks saturan** y los resultados no se traducen a **despliegue
económicamente valioso**. Introduce **tareas reales de largo-horizonte con salidas
verificables**, co-escrito por 250+ expertos, de industrias no-físicas. Pequeño para el
tema; pero refuerza la dirección de verificación comprobable.

---

## 5. El vacío específico de dominio (con evidencia de búsqueda)

La búsqueda sistemática de la literatura (API arXiv, 2026-08-26, queries combinadas:
"biodiversity informatics LLM agent", "ecological LLM benchmark species distribution",
"phylogenetic inference LLM agent", "spatial ecology benchmark language model",
"environmental science LLM evaluation") **no devolvió ningún benchmark establecido de
razonamiento/tareas de ecología y evolución con verificación por ejecución de código sobre
datos reales (GBIF, bioclim, filogenética). Los resultados ecológicos que aparecen son
(de la era pre-LLM, modelos predictivos 2005-2016) o de conocimiento/descubrimiento
(BAGEL, TerraIncognita) que NO verifican por ejecución de código.

**Cobertura de la búsqueda y limitación honesta:** solo arXiv (sin IEEE/ACL/NeurIPS/frontals
vencidas). Posible subestimación si existen benchmarks ecológico-evolutivos publicados en
esas sedes. No obstante, dentro del canon acceso de preprint abierto, el vacío de *verificación
por ejecución de código* es real.

**Precisión del gap (avant Ronda 2):** existen benchmarks ecológicos de CONOCIMIENTO
(BAGEL) y de DESCUBRIMIENTO (TerraIncognita, multimodal), ambos con respuesta de opción/
taxonomía. Lo que NO existe es un benchmark de ecología+evolución con **ground-truth
verificable por EJECUCIÓN de código** (R/Python) contra datos reales de SDM/filo/bioclim
— exactamente la capacidad diferencial de este laboratorio. El benchmark propio EQB es
semántico (LLM-judge), no ejecutable.

---

## 6. Conclusión y recomendación (pre-ronda DiDAL)

**Síntesis de la literatura:** el campo maduró hacia **entornos y tareas verificables por
ejecución** (D3-Gym, ScienceAgentBench, CORE-Bench, ALE) y hacia **metaevaluación** de
los propios benchmarks (BenchGuard, Twelve-audit). Para un agente científico general de
dominio ecológico-evolutivo:

1. Los **generalistas (Gemini/Qwen)** saturan MMLU/GPQA — no compiten por ahí.
2. El **espacio ganable** es un **benchmark de dominio (ecología + evolución)** con
   ground-truth **verificable por ejecución** sobre datos reales — vacío comprobado por
   búsqueda.
3. Por tanto: **construir EcoSmartBench-EVAL** (held-out, verificación por ejecución
   R/Python, datos reales GBIF/bioclim/filo), mantener **confirmación pública** (GPQA-bio,
   MMLU-Pro bio+quim, ScienceQA-bio) y **auditar el propio benchmark** (lección BenchGuard/
   Twelve-audit: reportar spec, profil.) — con el EQB actual como TRAIN.

### Recomendación numérica (blanco de evaluación)
- **EVAL propio:** 80-120 ítems ejecutables (MC verificado + tareas SDM/filo + tool-call).
- **Confirmar** con el pénc público de nicho (biología/ecología) + baselines (Gemini/Qwen).
- **Anti-contaminación:** EVAL congelado; EQB -> train. (lección SWE-bench-Live/BenchGuard.)

---

## 7. Referencias (primarias, verificadas en arXiv)

| # | Benchmark | arXiv ID | Fam de la literatura |
|---|---|---|---|
| 1 | GPQA | 2311.12022 | (A) |
| 2 | ScienceAgentBench | 2410.05080 | (B) |
| 3 | FML-bench | 2510.10472 | (B) |
| 4 | AutoSDT | 2506.08140 | (B) |
| 5 | D3-Gym | 2604.27977 | (B) |
| 6 | CORE-Bench | 2409.11363 | (B) |
| 7 | LabUtopia | 2505.22634 | (E) |
| 8 | SWE-bench-Live | 2505.23419 | (D) |
| 9 | LongCoT | 2604.14140 | (C) |
| 10 | Agents' Last Exam | 2606.05405 | (F suelto) |
| 11 | Meta-audit (Twelve) | 2605.21404 | (meta) |
| 12 | BenchGuard | 2604.24955 | (meta) |
| AA | GPQA leaderboard | OpenRouter 26-08 | (A) |

---

*Este documento es el manifold del review. Será sometido a 3 rondas DiDAL (diagnóstico,
cross-critique, refine) con gate de verificación de arXiv.*