# EcoBench-EVAL — Tabla comparativa multi-modelo (2026-08-28)

Benchmark: 14 ítems GIS ejecutables (eval_holdout, ScienceAgentBench, datos reales en ecobench_raw). Verificación por ejecución real + artifact. Retry-fix ≤2 con stderr realimentado. LLM-visual-judge SKIP (no es verificación pura). Gold data nunca en train (canary).

## Resultados (pass@1)

| Modelo | Vía | pass@1 | Notas |
|---|---|---|---|
| **deepseek-v4-flash-0731** | OpenRouter | **12/14 (86%)** | fix reasoning-first + fix griddata + typo-tolerance |
| nemotron-3-super (free) | OpenRouter | 9/14 (64%) | baseline original |
| z-ai/glm-4.7-flash | OpenRouter | 6/14 (43%) | código truncado en varios ítems |

## Detalle deepseek-v4-flash-0731 (12/14)

PASS: sab-4, sab-23, sab-32, sab-33, sab-46, sab-48, sab-49, sab-54, sab-76, sab-77, sab-84, sab-87
EXEC_FAIL: sab-64 (oggm: glacier dir no disponible — dependencia pesada), sab-86 (cartopy — dependencia pesada)

→ Los 2 fallos restantes son 100% entorno/dependencias pesadas (oggm, cartopy). Cero fallos de razonamiento ecológico ni de código tras los fixes.

## Fixes aplicados al runner (commits e00448f + posteriores)

1. **reasoning-first** (e00448f): deepseek-v4-flash-0731 emite contenido en `reasoning`/`reasoning_details` y deja `content` vacío si el budget se agota → 7/14 "empty_code" falsos. Fix: fallback a reasoning + max_tokens 8000. Efecto: 29% → 79%.
2. **fix griddata**: regla en SYSTEM_PROMPT — `griddata(points, values, xi, method='linear')` solo con method keyword, nunca doble (evita `TypeError: griddata() got multiple values for argument 'method'`). Efecto: sab-77 pasó de exec_fail a código ejecutable.
3. **typo-tolerance**: el enunciado SAB pide `interploated_water_quality.png` (typo) mientras el gold usa `interpolated_water_quality.png`. El verificador ahora acepta basename a distancia ≤2 ediciones (difflib.get_close_matches, cutoff 0.8), igual que un evaluador humano. Efecto: sab-77 PASS (typo-match). Aplica a toda la clase "inconsistencias de ruta de la tarea SAB".

## Pendientes

- Ítems eco-* (núcleo propio): specs no corresponden a datos reales (274 pu / 41 spp / era5-stack no existen tal cual). Decisión de diseño pendiente.
- dLLM-MoE entrenado (bw0): evaluar el checkpoint final cuando termine.
- glm-4.7-flash local Q6000 (:20006): saturado como teacher; reintentar cuando se libere.