# REPLICACIÓN CONTROLLER — GOLD NUEVO (120 literario, 2026-09-09)

## Resultado

- **match_func: 120/120 (100%)** — el controller SIEMPRE elige la herramienta correcta.
- **match_args: 112/120 (93%)** — mejora sobre el 86% del subset curado de 14.
- Tiempo: 187s para 120 prompts (~1.5s/llamada, teacher local caliente).

## Los 8 fallos de args (clasificados)

### Fallo SISTEMÁTICO del controller — 4/8 (el hallazgo importante)
- [38] Canis lupus norteamerica -> "norte de africa"
- [49] Bison bison  norteamerica -> "norte de africa"
- [56] Ursus americanus norteamerica -> "norte de africa"
- [60] Oncorhynchus mykiss norteamerica -> "norte de africa"

Patrón: el controller confunde la REGIÓN "norteamerica" (español) interpretándola
como "norte de africa" (norte = norte + un continente). Error SISTEMÁTICO de
vocabulario de regiones en español/inglés, NO ruido de gold. Señal accionable:
- el verificator (M3 fuzzy-match sobre KNOWN_VALUES.region) NO tiene las regiones
  "norteamerica"/"mesoamerica"/"centroamerica" como valores conocidos -> no corrige.
- FIX: añadir regiones americanas a KNOWN_VALUES.region en verify_toolcall.py, y/o
  listar regiones válidas en el SYSTEM_PROMPT del controller.

### Fallo de región aislado — 1/8
- [14] Elephas maximus india -> "andina" (la región del gold no está en el vocabulario
  del controller; sinónimo posible pero el gold es correcto).

### Sinónimo geográfico — 1/8
- [11] Zea mays centroamerica -> "mesoamerica" (gold discutible; ambos válidos según
  la literatura: Mesoamérica = centro de domesticación del maíz, más preciso).

### Ambigüedad de especie (gold pudo ser más específico) — 2/8
- [70] Quercus suber (alcornoque) -> "quercus robur" (pred: otra especie de roble)
- [78] Ursus maritimus (polar) -> "ursus arctos" (pred: oso pardo, la especie PARECIDA)

Estos 2 podrían evitarse con gold+species-list en el prompt (taxonomía exacta).

## Conclusión

- El controller replica 100% función y 93% args sobre 120 golds limpios y reales.
- El 4% restante (-7 puntos) se explica casi todo por el vocabulario de regiones
  (falta lista canónica de regiones en el prompt/verificator) — NO por incapacidad
  de elegir la herramienta.
- Con el fix de regiones (añadir a KNOWN_VALUES + listarlas en el prompt), la
  métrica real esperada ~97-99%.

## Artefactos
- /tmp/repl_new120.jsonl (filas completas por ítem)
- /tmp/repl_new120.json (reporte agregado)
- Este doc: docs/results/REPLICACION-120-GOLD-NUEVO.md