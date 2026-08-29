# EcoBench-EVAL — Tabla comparativa multi-modelo (2026-08-28)

Benchmark: 14 ítems GIS ejecutables (eval_holdout, ScienceAgentBench, datos reales en ecobench_raw). Verificación por ejecución real + artifact. Retry-fix ≤2 con stderr realimentado. LLM-visual-judge SKIP (no es verificación pura). Gold data nunca en train (canary).

## Resultados (pass@1)

| Modelo | Vía | pass@1 | Notas |
|---|---|---|---|
| **deepseek-v4-flash-0731** | OpenRouter | **11/14 (79%)** | runner con fix reasoning-first |
| nemotron-3-super (free) | OpenRouter | 9/14 (64%) | baseline original |
| z-ai/glm-4.7-flash | OpenRouter | 6/14 (43%) | código truncado en varios ítems |

## Detalle deepseek-v4-flash-0731 (11/14)

PASS: sab-4, sab-23, sab-32, sab-33, sab-46, sab-48, sab-49, sab-54, sab-76, sab-84, sab-87
EXEC_FAIL: sab-64 (oggm: glacier dir no disponible — dependencia pesada), sab-77 (griddata argumento duplicado — bug de código trivial), sab-86 (cartopy — dependencia pesada)

→ 2/3 fallos son entorno/dependencias (oggm, cartopy); 1 es bug menor de código. Sin fallos de razonamiento ecológico.

## Detalle z-ai/glm-4.7-flash (6/14)

PASS: sab-23, sab-32, sab-33, sab-54, sab-84, sab-86
EXEC_FAIL (8): mayormente SyntaxError/IndentationError (código truncado por extracción del reasoning) + IndexError/KeyError/ValueError de lógica. La extracción de su reasoning produce código incompleto.

## Fix aplicado al runner (commit e00448f)

- deepseek-v4-flash-0731 es reasoning-first: emite contenido en `reasoning`/`reasoning_details` y deja `content` vacío si el budget se agota razonando → 7/14 "empty_code" (falsos negativos por formato, no capacidad).
- Fix: fallback a reasoning/reasoning_details cuando content está vacío + max_tokens 4000→8000.
- Efecto medido: v4flash 4/14 (29%) → 11/14 (79%).

## Pendientes

- Ítems eco-* (núcleo propio): specs no corresponden a datos reales (274 pu / 41 spp / era5-stack no existen tal cual). Decisión de diseño pendiente.
- dLLM-MoE entrenado (bw0): evaluar el checkpoint final cuando termine.
- glm-4.7-flash local Q6000 (:20006): saturado como teacher; reintentar cuando se libere.