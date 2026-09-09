# Expansión de tool-calls EcoReasoner — Minar literatura para más dominios

Fecha: 2026-09-09 · Autor: Devin (SWE-1.7) · Rama: `main`

## Estado actual

- 3 herramientas implementadas y verificadas: `gbif_occurrence`, `bioclim_download`, `maxent_train`.
- 120 pares `prompt → tool-call` en `data/l1/toolcalls_lit_gold.jsonl`.
- `verify_toolcall.py` valida schemas empíricos.
- `tools_resolver.py` resuelve las 3 con datos reales (GBIF, HPC bioclim, MaxEnt spec).
- El controller de `run_controller_verificator.py` presenta 3 tools con descripción fija.

## Objetivo

Ampliar el benchmark de tool-calls con nuevas herramientas reales de la literatura ecológica, sin contaminar el gold con análisis compuestos. Las herramientas nuevas deben seguir siendo **llamadas directas** a una única fuente de datos, tener **schemas simples** y ser **resolvibles** (al menos en mock; real cuando sea factible).

## Dominios a ampliar (propuesta)

### 1. Datos de biodiversidad adicionales

- `inaturalist_obsurrences`: `{"species": str, "region": str}`
- `obis_occurrence`: `{"species": str, "region": str}`  ( Ocean Biodiversity Information System )
- `ebird_occurrence`: `{"species": str, "region": str, "year": str}`
- `bien_occurrence`: `{"species": str, "region": str}`  ( Botanical Information and Ecology Network )

### 2. Capas ambientales y geoespaciales

- `srtm_elevation`: `{"region": str, "resolution": str}`
- `soilgrids_download`: `{"region": str, "layer": str}`
- `landcover_download`: `{"region": str, "year": str}`  ( ESA WorldCover / Copernicus )
- `modis_ndvi`: `{"region": str, "year": str, "product": str}`
- `human_footprint`: `{"region": str, "year": str}`
- `protected_area_download`: `{"region": str}`  ( WDPA )
- `road_density_download`: `{"region": str}`

### 3. Variables climáticas históricas/futuras

- `worldclim_download`: `{"region": str, "year": str, "resolution": str}`
- `chelsa_download`: `{"region": str, "year": str}`
- `era5_download`: `{"region": str, "year": str, "variable": str}`
- `daymet_download`: `{"region": str, "year": str}`  ( limitado a Norteamérica )

### 4. Rasgos funcionales y bases taxonómicas

- `try_traits`: `{"species": str, "trait": str}`
- `elton_traits`: `{"species": str}`
- `amphiBio_traits`: `{"species": str}`
- `fishbase_traits`: `{"species": str}`
- `iucn_status`: `{"species": str}`
- `timetree_phylogeny`: `{"species": str}`
- `opentree_phylogeny`: `{"species": str}`
- `ncbi_taxonomy`: `{"species": str}`

### 5. Ejecución de análisis simples (controlados, 1 operación)

No análisis compuestos. Ejemplos de 1 solo paso:

- `raster_extract`: `{"raster": str, "points": str}`
- `buffer_shapefile`: `{"layer": str, "distance_km": str}`
- `reclassify_raster`: `{"raster": str, "rules": str}`

## Criterios de selección de nuevas herramientas

1. **Literalidad en la literatura**: el paper menciona descargar X capa de Y fuente.
2. **Resolubilidad**: debe existir una API pública, un archivo en `/beegfs` o un endpoint real que el resolver pueda llamar.
3. **Schema simple**: 1-3 argumentos, todos strings, con nombres alineados al vocabulario del corpus.
4. **No análisis compuesto**: "descargar" y "entrenar" sí; "calcular", "comparar", "proyectar" no.
5. **Evitar dependencias legales**: preferir abiertas (GBIF, OBIS, iNaturalist, ESA, WorldClim, TRY, EltonTraits, NCBI).

## Minado de literatura (pipeline propuesto)

### Fuentas disponibles

- `ecoseek-litdump` (`/beegfs/a474r867/ecoseek-litdump/`).
- PMC/arXiv descargados en `/beegfs/a474r867/ecoreasoner/data/`.
- `docs/benchmark-literature-review/` y `docs/ecobench/` (ya curados).
- `ecobench_eval.json` (30 ítems con referencias a especies/regiones/datos).

### Pasos

1. **Extracción de entidades**:
   - Especies (NCBI Taxonomy / names): regex de nombres científicos.
   - Regiones: lista de regiones conocidas (GADM, ecoregions, países, biomas).
   - Datasets mencionados: GBIF, iNaturalist, OBIS, eBird, TRY, WorldClim, CHELSA, ERA5, SoilGrids, SRTM, ESA WorldCover, MODIS, WDPA, etc.

2. **Clasificación de menciones en tool-calls**:
   - Regla heurística: frase "descargamos X para Y" → `bioclim_download`/`soilgrids_download`/etc.
   - Frase "ocurrencias de X en Y" → `gbif_occurrence` u otra base de biodiversidad.
   - Frase "entrenamos MaxEnt con X" → `maxent_train`.
   - Frase "obtuvimos rasgos de X de TRY" → `try_traits`.

3. **Generación asistida**:
   - Plantillas por tool (como `gen_toolcalls_lit.py`).
   - Muestreo estratificado por dominio para mantener variedad.
   - Verificación con `verify_toolcall.py`.
   - Auto-check de consistencia prompt↔args.

4. **Curación humana**:
   - Muestra aleatoria revisada por Hermes (sobre todo las nuevas tools).
   - Validar que el prompt es directo y el resolver puede ejecutarlo.

## Ejemplo de oro ampliado

```json
{
  "prompt": "Descarga las capas de elevación SRTM para la región andina a 30 m de resolución.",
  "gold": [{"tool": "srtm_elevation", "args": {"region": "andina", "resolution": "30m"}}],
  "source": "literatura: modelos SDM de camélidos andinos usan SRTM como covariable topográfica"
}
```

```json
{
  "prompt": "Consulta el estado de conservación de Puma concolor en la Lista Roja de la UICN.",
  "gold": [{"tool": "iucn_status", "args": {"species": "Puma concolor"}}],
  "source": "literatura: análisis de riesgo de extinción de felinos neotropicales"
}
```

## Cambios técnicos necesarios

### `scripts/verify_toolcall.py`

- Añadir nuevos schemas a `SCHEMAS` y `ARG_TYPES`.
- Extender `KNOWN_VALUES` con regiones, especies, capas, productos, resoluciones.
- Mantener validación de requeridos/opcionales y reparaciones M1-M5.

### `ecobench/tools_resolver.py`

- Añadir `RESOLVERS` para cada nueva tool.
- Para datos abiertos, usar `urllib` (GBIF-like APIs).
- Para datasets en HPC, verificar paths en `/beegfs`.
- Para mocks, devolver `{"ok": true, "mock": true, "args": ...}` si no hay resolver real todavía.

### `ecobench/run_controller_verificator.py`

- Actualizar `TOOLS_DEF` y `SYSTEM_PROMPT` con las nuevas tools.
- Asegurar que el prompt del controller no mezcla análisis compuestos.

### `scripts/gen_toolcalls_lit.py`

- Refactorizar: leer listas curadas desde archivos JSON (`data/l1/species_regions.json`?).
- Añadir generadores por dominio.
- Añadir `--domain` para generar un subconjunto.

## Fases propuestas

1. **Fase 1 (piloto)**: 3-4 nuevas tools de datos abiertos y resolvibles.
   - `srtm_elevation`, `soilgrids_download`, `iucn_status`, `try_traits`.
   - Generar 60 pares adicionales (30 por tool aproximadamente).

2. **Fase 2 (dominios)**: añadir biodiversidad alternativa y capas climáticas.
   - `inaturalist_occurrence`, `obis_occurrence`, `worldclim_download`, `landcover_download`.
   - Generar 120 pares más.

3. **Fase 3 (minería)**: pipeline de extracción automática desde ecoseek-litdump + PMC.
   - Scrapear 1K papers, extraer entidades, proponer tool-calls, curar 20%.
   - Meta: 500 pares de tool-calls de múltiples dominios.

## Riesgos y mitigaciones

- **Resolver de IUCN**: la API requiere token. Mitigación: empezar con mock o usar la API pública de `http://apiv3.iucnredlist.org` si se tiene token.
- **TRY**: no tiene API abierta. Mitigación: mock o limitar a casos con datos descargados manualmente en `/beegfs`.
- **SRTM**: resolución y tile. Mitigación: resolver devuelve el DEM global para la región o un enlace al tile; no descargar en tiempo real.
- **Sobrecarga de herramientas**: más de 10 tools puede confundir al controller. Mitigación: agrupar por dominio y entrenar por fases.

## Siguiente paso inmediato

Decidir con Hermes cuáles son las 3-4 primeras tools a añadir, validar que tengan resolver factible, y generar el primer bloque de 60 pares con la misma metodología del gold de 120.
