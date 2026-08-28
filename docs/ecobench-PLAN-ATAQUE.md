# EcoBench-EVAL — Plan de ataque (2026-08-28)

> Objetivo: convertir el prototipo (runner demo, "pending-agent") en un benchmark
> REAL ejecutable que produzca puntuaciones verificables para el paper.

## Estado actual (verificado)

- `ecobench_eval.json`: 30 ítems (17 eval_holdout, 13 train). 27 GIS de
  ScienceAgentBench + 3 propios (sdm/phylo/bioclim).
- `run_eval.py`: runner con `verify_numeric`/`verify_artifact` REALES, pero en
  modo demo — `status="pending-agent"` (no ejecuta nada, no hay agente conectado).
- Datos reales disponibles para los 3 ítems propios:
  - `scratch/xsdm_1000_sp/Breviceps_montanus/phase1_results/` (L1-L4, rds)
  - `scratch/xsdm_env_extraction_19/Breviceps_montanus/` (P12-P19, T10-T19 csv)
  - `scratch/xsdm_results_v6/Breviceps_montanus/` (model_results_v6.rds)
- Modelos: v4-flash teacher (local :20006), glm-4.7-flash, OpenRouter key.

## El problema de fondo (crítico)

**27/30 ítems son de ScienceAgentBench (GIS)** — dependen de sus datasets de
entrada (`dataset_folder_tree`) y sus `eval_script`/`gold_program`, que NO están
en `ecobench_raw/` (solo el CSV). Sin esos datos, esos ítems son **inejecutables**
en este entorno. El 90% del benchmark hoy no se puede correr.

Los únicos ítems REALMENTE ejecutables con datos propios son los 3 del núcleo
(sdm/phylo/bioclim) — que son justo los que diferencían al lab.

## La estrategia en 3 frentes

### Frente 1 — Hacer ejecutable el núcleo propio (lo que importa)
1. Materializar los datos de los 3 ítems en `ecoseek-benchmark/data/`:
   - copiar Breviceps phase1 (para `eco-sdm-001`)
   - copiar env extraction bioclim (para `eco-bioclim-001`)
   - crear un fixture filogenético sintético (para `eco-phylo-001`)
2. Conectar un AGENTE al runner: el runner pide al modelo (v4-flash/glm) que
   genere código R/Python para el ítem, lo ejecuta, y verifica contra expected.
   Esto convierte "pending-agent" en puntuación REAL.
3. Calcular las puntuaciones (AUC val, lambda, bio01).

### Frente 2 — Ampliar el núcleo propio (la celda del gap)
- Añadir más ítems ecológicos EJECUTABLES con datos del lab:
  - `eco-sdm-00N`: otras especies de xsdm_1000_sp (AUC por especie)
  - `eco-bioclim-00N`: más variables BIO (P19, T10...)
  - `eco-phylo-00N`: K de Blomberg, lambda con fixtures
  - `eco-gbif-00N`: query GBIF real con conteo verificable
- Criterio: cada uno con expected_value REAL y ejecutable.

### Frente 3 — Baseline de modelos (el número para el paper)
- Correr los ítems ejecutables con: v4-flash (teacher), glm-4.7-flash, qwen
  (si está), y el dLLM-MoE entrenado (una vez haya checkpoint).
- Reporte: pass@1 por ítem/familia + tabla comparativa.
- Los ítems SAB-GIS quedan **documentados como no-ejecutables en este entorno**
  (dependen de datasets externos); se marcan `skip` con razón, no se eliminan.

## Coste y recursos
- Frente 1-2: local + cluster CPU (copiar datos, generar fixtures, ejecutar R/Python)
- Frente 3: llamadas al teacher v4-flash (:20006) y OpenRouter (gratis/free)
  — el runner llama al endpoint, genera código, ejecuta en sandbox del runner.
- Sin jobs GPU; solo CPU y llamadas API.

## Criterio de éxito (medible)
- `run_eval.py` produce un JSON con: por ítem `status ∈ {pass, fail, skip}`,
  `actual_value`, y un resumen `pass@1` por familia.
- Al menos los 3-10 ítems del núcleo propio con puntuación REAL de ≥3 modelos.

## Orden de ejecución
1. Materializar datos + fixtures (local, sin GPU).
2. Escribir el driver del agente (llama a :20006/OpenRouter → código → ejecuta).
3. Ampliar núcleo a ~10-15 ítems ejecutables.
4. Correr baselines (3 modelos) → reporte JSON + tabla.
5. Actualizar el doc del skill con la metodología ejecutable.

> Regla de labor: los pasos 1-5 se ejecutan con OK del usuario; el driver del
> agente usa endpoints ya autorizados (teacher local) y OpenRouter free.