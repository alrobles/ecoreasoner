# INSTRUCCIONES — Generación de tool-calls ecológicas basadas en literatura

Para: Devin (SWE-1.7) · De: Hermes/Ángel · Fecha: 2026-09-09
Repo: `/home/reumanlab/ecoreasoner` (rama main, push a origin/main)

## Objetivo

Generar NUEVOS pares `prompt → tool-call` (gold) para la línea controller/verificator
(opción D) de EcoReasoner. Se necesitan más datos de entrenamiento/evaluación de
tool-calls REALISTAS basados en literatura ecológica (especies reales, regiones
reales, preguntas de investigación reales que se resuelven llamando una herramienta).

## Contexto (por qué pedimos esto AHORA)

El corpus actual de tool-calls (533 trazas distiladas del teacher) tiene el gold
CONTAMINADO: solo 14/60 prompts (~23%) son llamadas directas a las 3 herramientas;
el resto son análisis (RF/filogenia/diversidad funcional), infraestructura
(slurm/watchdog/HPC/github) o literatura (PubMed/artículos) que el teacher etiquetó
incorrectamente como `gbif_occurrence`. Limpiamos el subset y el controller quedó
validado (100% función, 86% args sobre 14 puros). Ahora necesitamos AMPLIAR ese
subset limpio con más casos.

## Las 3 herramientas (ÚNICAS permitidas, schemas empíricos de verify_toolcall.py)

```python
gbif_occurrence : {"species": str (requerido), "region": str (opcional)}
  # buscar registros de presencia/ocurrencias/distribución/taxonomía de una especie
bioclim_download: {"region": str (requerido), "year": str (opcional)}
  # descargar capas bioclimáticas (CHELSA/ERA5) para una región/año
maxent_train    : {"species": str (requerido), "layers": str (requerido)}
  # entrenar modelo de nicho (MaxEnt) para una especie con capas
```

## Reglas de calidad (CRÍTICAS — la lección del gold contaminado)

1. **SOLO estas 3 herramientas.** Prohibido: knowledgebase, scripts/slurm/HPC,
   GitHub/issues, PubMed/artículos revisados, web, sesiones DB, watchdogs, APIs
   de terceros (IUCN, Ollama), fitogeografía, etc. Si el prompt no se resuelve
   con una de las 3 → NO lo generes.

2. **El prompt DEBE ser una llamada de herramienta directa** (una acción de
   búsqueda/descarga/entrenamiento de datos ecológicos sobre UNA entidad).
   Prohibidos los prompts de análisis compuesto: "analiza", "escala", "ejecuta
   una regresión", "carga un árbol filogenético", "calcula métricas", "estima
   idoneidad", "proyecta a 2050" (esto ya no es la herramienta sola).

3. **Los args DEBEN ser consistentes con el prompt.** La lección del corpus:
   el item 2 pedía "península de Yucatán" pero el gold decía region="neotropico";
   el item 4 pedía "proyección a 2050" pero el gold decía year="2020". ERROR.
   Si el prompt dice una región/año/especie, el gold DEBE tenerla.

4. **Especies y regiones REALES, basadas en literatura.** Usa especies que
   aparecen en estudios ecológicos de nicho/distribución (mamíferos, aves,
   anfibios, peces, plantas, insectos vectores, ungulados, etc.) con sus nombres
   científicos correctos (Panthera onca, Danaus plexippus, Ambystoma mexicanum,
   Ursus arctos, Rangifer tarandus, Aedes aegypti, Quercus robur, Pinus
   sylvestris, Loxodonta africana, Centrocercus urophasianus, ...). Regiones
   geográficas reales (neotropico, paleartico, amazonia, sahara, alpes,
   mediterraneo, gran barrera de coral, himalaya, patagonia, peninsula de
   yucatan, sudeste asiatico, africa oriental...). NO inventes especies.

5. **Variedad realista**: mezcla ~70% gbif_occurrence, ~20% bioclim_download,
   ~10% maxent_train. Varía especies Y regiones (no repitas el mismo patrón).

6. **Verificación automática OBLIGATORIA**: cada `bad`... cada par generado debe
   pasar `python3 scripts/verify_toolcall.py --text '<tool-call json>'` → `ok:
   true`. NO entregues tool-calls que el verificator rechace.

## Formato de salida

JSONL, UNA línea por par, ruta: `data/l1/toolcalls_lit_gold.jsonl`

```json
{"prompt": "Busca registros de presencia de la especie Ursus arctos en la región paleartica para un modelo de nicho.",
 "gold": [{"tool": "gbif_occurrence", "args": {"species": "Ursus arctos", "region": "paleartico"}}],
 "source": "literatura: osos pardos eurasiáticos (estudios de distribución)"}
```

- `prompt`: la tarea (en español, estilo del corpus, 1-3 frases, natural).
- `gold[0].tool` y `gold[0].args`: la tool-call correcta.
- `source`: una nota corta de la base literaria (opcional pero valorada — hace la
  generación auditable y "basada en literatura" de verdad).

OBJETIVO: **120 pares** (≈84 gbif, ≈24 bioclim, ≈12 maxent), sin repetir
especie+región dentro de lo razonable.

## Cómo trabajar

1. `git checkout main && git pull` (regla: trabajar SIEMPRE sobre main fresco).
2. Crea `scripts/gen_toolcalls_lit.py` SOLO si necesitas un generador auxiliar
   (muestreo de especies/regiones reales desde una lista curada local). Si
   prefieres, genera directamente y valida.
3. Valida TODOS los pares con `verify_toolcall.py` (100% deben pasar).
4. Auto-check de consistencia prompt↔args (regla 3): escribe un mini-check que
   detecte si el prompt menciona una región/año/especie distinta del gold.
5. Escribe `data/l1/toolcalls_lit_gold.jsonl` + un `REPORT-SOURCE.txt` corto
   (cuántos generaste, distribución por tool, cómo lo validaste).
6. Commit + push a main con mensaje claro.

## NO HAGAS

- NO toques `scripts/verify_toolcall.py`, `ecobench/run_controller_verificator.py`,
  `ecobench/tools_resolver.py` ni `runs/` (línea viva de la opción D).
- NO lances jobs HPC ni uses GPU (esto es generación de datos, CPU/nada).
- NO generes prompts que mezclen herramientas o análisis compuestos.
- NO uses 127.0.0.1:20006 (teacher) a menos que sea estrictamente necesario
  para una consulta puntual — el objetivo es generar con TU conocimiento, no
  re-distilar del teacher.

## Criterio de éxito

- `data/l1/toolcalls_lit_gold.jsonl` con ~120 líneas válidas.
- 100% pasan `verify_toolcall.py --text` (reporta el comando y el conteo).
- Consistencia prompt↔args verificada por tu propio check.
- Docs del proceso en `REPORT-SOURCE.txt`.