# DiDAL — Registro Ronda 2 (cross-critique)

## role_narrative (alpha) — R2
- Acepta BAGEL. RECHAZA TerraIncovita y BioBench POR MEMORIA ("no verifiable by this name exists").
  -> El orchestrador DESMIENTE: verificación gate confirma TerraIncognita 2506.03182 (species discovery)
  y BioBench 2511.16315 (scientific vision, imagenet beyond) en arXiv. El juicio del crítico narrativo
  era defectuoso (base de memoria), no de hechos.
- Fix narrativo válido: BAGEL fila truncada ("conocim...") -> completar con datos verified (11,852 MCQ,
  4 fuentes: Wikipedia, GloBI, bioRxiv, Xeno-canto; closed-book animal natural-history).
- Narr Ejecutivo valido: separar revisión->implicación->decisión; documentar accept/reject de las 3 props.

## role_citations (alpha) — R2 (MATRIZ CLAVE)
- **gap_accurate: "yes — defensible"**. Cita: neither BAGEL (knowledge) ni TerraIncovita (discovery/vision)
  ni BioBench (ecology vision) ni ningún revisado evalúa EJECUCIÓN de pipeline SDM o código filogenético
  con verificación de ejecución. El gap es real y distinto de los nichos adyacentes.
- confirmed_add con ground_truth y status:
  - BAGEL: correct option, 11,762 items (taxonomy/morpho/habitat/behavior/vocalization/geo-dist/species-inter);
    closed-book animal expertise. NO pipeline exec. status verified.
  - TerraIncognita: labels expertos de taxonomía (Order/Fam/Gen/Spp) para 200 réplicas + abstention OOD de
    100 especies nuevas. status verified.
  - BioBench: species ID + traits + behavior, 9 tasks ecology, 3.1M img, macro-F1. status verified.
- **final_matrix execution_verifiable flag**: SciAgentBench y D3-Gym execution_verifiable=true pero
  sdm_phylo_coding=false (gap_fills=false). BAGEL/Terra/BioBench execution=false. EQB execution=false.
  => NINGÚN benchmark existente combina execution_verifiable=TRUE + sdm_phylo_coding=TRUE. El gap es
  exactamente esa celda vacía.

## Resolución del orquestador (Ronda 2 -> aplicar en Ronda 3)
1. ACEPTAR BAGEL, TerraIncognita, BioBench (verificados en arXiv). Corregir mal rechazo del narrativo.
2. Añadir MATRIZ ejecución-verifiable (columna por fila) que demuestre el gap limpio.
3. BAGEL completar (11,762 items, 4 fuentes). Añadir BioBench a tabla.
4. Documentar verificación.  Fecha cobertura búsqueda ya añadida R1.

## Referencias verificadas totales: 15 (12 núcleo + BAGEL + Terra + BioBench)