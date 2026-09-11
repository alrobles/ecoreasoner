# Pivot a Controller/Verificator — Plan de MVP

> Fecha: 2026-09-11.
> Trigger: V3.3 NO-GO (L3 = 0.486 ns). Se archiva la línea dLLM-puro y se
> pivotar al sistema controller/verificator, que ya demostró 98.2% match_args
> en replicación de 500 tool-calls.

## 1. Estado del artefacto actual

Ya disponemos de:

- `ecobench/run_controller_verificator.py`: controller con retry + verificator.
- `scripts/verify_toolcall.py`: reparadores M1-M5 + fuzzy match.
- `ecobench/tools_resolver.py`: resolución real de GBIF, iNaturalist, NCBI;
  mock/partial para IUCN, SRTM, TRY, TimeTree, OpenTree.
- 10 herramientas definidas: GBIF, bioclim, MaxEnt, IUCN, SRTM, iNaturalist,
  TRY, TimeTree, OpenTree, NCBI.
- Replicación 500 tool-calls: 99.6% match_func, 98.2% match_args.

## 2. Definición del MVP productivo

El mínimo prototipo viable que justifica fondos/tarjetas es un **sistema que
reciba preguntas ecológicas en lenguaje natural y ejecute secuencias de
herramientas científicas verificadas**, devolviendo un resultado real o un
plan lanzable.

Ejemplo de pipeline MVP:

> "Haz un modelo de nicho para *Panthera onca* en la península de Yucatán."

1. **Controller** emite `gbif_occurrence`.
2. **Verificator** valida argumentos.
3. **Resolver** real consulta GBIF y devuelve N registros.
4. **Controller** emite `bioclim_download` para la región y año.
5. **Resolver** localiza capas CHELSA/ERA5 en HPC.
6. **Controller** emite `maxent_train`.
7. **Resolver** genera spec de job HPC para MaxEnt.

## 3. Hitos del pivot

| Hito | Qué entrega | Estado | Esfuerzo |
|---|---|---|---|
| H1 | Tool-calls individuales 98%+ con verificator | ✅ listo | - |
| H2 | Pipeline multi-step SDM (gbif→bioclim→maxent) | 🔄 en diseño | 1-2 días |
| H3 | Resolver real de bioclim + maxent spec | 🔄 en diseño | 1-2 días |
| H4 | `EcoToolBench`: 30-50 tareas multi-step hold-out | 🔄 pendiente | 2-3 días |
| H5 | Demo reproducible para fondos/GPUs | 🔄 pendiente | 1 día |

## 4. Tareas inmediatas

1. **Mapear stacks reales en `tools_resolver.py`**.
   - El bioclim actual apunta a rutas vacías. Los datos reales están en:
     `/beegfs/a474r867/stage1_conus_hybrid_bioclim_pr_temporal` (CONUS, años
     1960-2024, rasters `.tif`).
   - Para regiones globales hay que construir o encontrar el stack global
     correspondiente (o marcar `partial` con instrucciones claras).
   - MaxEnt: preparar un launcher `moe_v4/species_dm` que tome GBIF CSV +
     capas y lance entrenamiento.

2. **Implementar `ecobench/run_sdm_pipeline.py`**.
   - Recibe una pregunta.
   - Itera controller→verificator→resolver hasta resolver o fallar.
   - Registra traza en JSON.
   - Soporta `--real` y `--mock`.

3. **Crear benchmark multi-step `data/l1/ecotoolbench.jsonl`**.
   - 30 tareas de 1-3 pasos (SDM, filogenia, biodiversidad).
   - Cada ítem: question, gold (secuencia de tool-calls), split eval/train.

4. **Documentar el demo**.
   - Notebook o script que corra 5 ejemplos end-to-end con salida real.
   - Métricas: % tareas resueltas, % tool-calls válidas, % resultados reales.

## 5. Notas sobre el dLLM

El dLLM 155M no se descarta completamente: puede quedar como **generador de
borrador de razonamiento/texto** dentro de un sistema híbrido, pero el
componente crítico de verificación y ejecución pasa al controller/verificator.

## 6. Riesgos

- Resolver de bioclim/maxent requiere acceso a datos HPC y posiblemente
  preproceso de regiones.
- Pipelines multi-step exponen más oportunidades de error acumulado.
- EcoToolBench requiere anotación de gold, pero se puede derivar de las
  tool-calls reales.

## 7. Primer paso concreto

Comenzar con H2+H3: implementar y probar el pipeline SDM sobre una sola
especie (`Panthera onca`, `Yucatán`) en modo `--real` para identificar qué
partes del stack faltan.
