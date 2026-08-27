# DiDAL — Registro Ronda 3 (polish final)

## role_narrative (alpha) — R3
- Abstract final ES/EN proporcionado (7 familias, gap estructural, EcoBench-EVAL, meta-evaluación).
- Fix estructura: 1) integrar 3.b como panel lateral en Sección 3; 2) unificar Sección 4 con tabla
  (arxiv ID como clave única); 3) columna "Estado de verificación" (verificado/pendiente/NA).
- key_contribution: ninguna benchmark verifica por ejecución un agente en dominio ecología/evolución;
  EcoBench-EVAL llena la celda vacía con meta-evaluación (BenchGuard 5 campos).
- single_next_step: construir prototipo EcoBench-EVAL con 5-10 tareas SDM reproducibles del xsdm
  pipeline, aplicando auditoría BenchGuard (identidad/spec/harness/inf/evaluator).
- residual_risks: cobertura solo arXiv; taxonomía de 7 familias puede colapsar; claim "celda vacía"
  asume SDM/filo como dominio no cubierto por data-discover general.

## role_citations (alpha) — R3
- **verification_pass: "clean"**.
- difference_maker: primer benchmark ecológico-evolutivo con verificación de EJECUCIÓN de código
  sobre datos reales (GBIF/bioclim/árboles filogenéticos) — ni BAGEL ni TerraIncognita lo cubren.
- final_recommendation: (a) tarea de ejecución (prompt SDM/filo -> agente genera+ejecuta R/Python
  contra datos reales -> verificar salida numérica, no solo texto); (b) abrir en arXiv como preprint
  para prioridad, extender luego a revisão por pares.
- not_verified (as-intended, no fabrication): TerraIncognita y BAGEL no verifican por ejecución;
  benchmarks predictivos 2005-2016 pre-LLM no aplican.

## Resolución final orquestador
- Las 15 referencias (12 núcleo + BAGEL + TerraIncognita + BioBench) verificadas contra arXiv. Clean.
- Hallazgo convergente (narrativa + citas en R3): EcoBench-EVAL como difference-maker = descentral en
  la celda ejecución-verificable × dominio ecológico.
- El review final integra la matriz, el abstract pulido y la recomendación concreta.