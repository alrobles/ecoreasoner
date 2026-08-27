# DiDAL — Registro de rondas (literature review benchmarks agentes científicos)

> Proyecto: EcoReasoner · ReumanLab · Fecha: 2026-08-26
> Canal: SSH -> alpha (hermes-agent). 3 rondas planificadas.

## Ronda 1 — Diagnóstico independiente

### role_narrative (alpha) — veredicto "revise"
Fortalezas: search queries trazables; taxonomía refinada 3->7; meta-evaluación (BenchGuard/Twelve) timely;
gap respaldado por evidencia negativa de búsqueda; grounding arXiv; separación EVAL vs CONFIRM.
Observaciones (a resolver en ronda 3): 
 - Incoherencia numérica: dice "cinco familias" pero la tabla enumera A-G (7). Corregir / reagrupar.
 - GAP: sección demasiado corta; falta fecha de búsqueda, criterios de exclusión exactos y cobertura
   (solo arXiv, posible subestimación sin IEEE/ACL/NeurIPS).
 - Columna "Relevancia Ecología/Evol" sin criterio reproducible ("Media/Baja/parcial").
 - La sección CONCLUSIÓN mezcla hallazgo de literatura con plan de proyecto (80-120 ítems), rompiendo
   revisión->implicación->decisión. Separar en (a) síntesis, (b) implicaciones, (c) recomendación.

### role_citations (alpha1) — veredicto "revise" (compact re-run, JSON limpio)
Notas:
 - La tabla comparativa se le pasó con solo cabeceras (bug del build): no pudo cruzar columnas vs refs.
   -> resolver reenviando tabla POBLADA en Ronda 3.
 - Missing_worth_adding (con ground_truth) propone 3 benchmarks ecológicos: 
   - BAGEL (2604.16241): animal knowledge, closed-book. Ground: species ID, morpho, habitat, dist, interactions. Relevancia ecología ALTA.
   - TerraIncognita (2506.03182): dynamic species discovery, taxonomy + OOD, novel taxa. ALTA.
   - BioBench: ecológico visión/interest (species ID, drone behaviour, traits). ALTA.
   -> Verificados en arXiv: BAGEL y TerraIncognita REALES (título/fecha correctos).
 - Metodología gaps: mapeo columna->contenido; gold-standard de los 13 IDs.

### Verification gate (orquestador)
 - 11/11 arXiv IDs núcleo confirmados contra API (GPQA, SciAgentBench, FML, AutoSDT, D3-Gym, LabUtopia,
   CORE-Bench, SWE-bench-Live, Twelve-Audit, BenchGuard, LongCoA).
 - BAGEL y TerraIncognita confirmados (nuevos). 
 - "Gemini 3.1 Pro 94.3% GPQA" confirmado por web_extract OpenRouter (primaria), no de memoria.

## A incorporar en el review (antes de Ronda 2)
1. Corregir "cinco/varias familias" -> homogeneizar (7).
2. Completar GAP con fecha/cobertura/criterios + reconocer subestimación (solo arXiv).
3. Añadir BAGEL/libro, TerraIncognita, Bug Interpretation en tabla como benchmarks ecológicos existentes,
   y matizar el gap: existen de conocimiento/descubrimiento pero NO de ejecución SDM/filo verificable.
4. Separar secciones (a) hallazgo, (b) implicación, (c) decisión.
5. Reenviar a Ronda 2 con martinza completa.