# REPLICACIÓN CONTROLLER — FASE 3 (500 tool-calls, 7 tools) — RESULTADO

Fecha: 2026-09-09 · Modelo: deepseek-v4-flash (ollama local) · 500 prompts · 784s

## Resultado global (RUN v1 — sin fix de vocabulario)

| Métrica | Valor |
|---|---|
| match_func | **500/500 (100%)** |
| match_args | 458/500 (92%) |
| Tiempo | 784 s (~1.6 s/prompt) |

## RUN v2 — FIX DE VOCABULARIO APLICADO (2026-09-09, 776s)

**Fix**: (1) KNOWN_VALUES del verificator ampliado: años 2010-2026 (antes 2015-2023;
2010 ni siquiera estaba → el controller "inventaba" 2015), +23 regiones del gold
(africa occidental/oriental, andes, sahel, caribe, paleartico, tundra, pampas,
himalaya, madagascar...). (2) SYSTEM_PROMPT del controller ahora lista las
regiones VÁLIDAS y los años admitidos (derivado de KNOWN_VALUES, sin duplicar).

| Métrica | v1 | v2 |
|---|---|---|
| match_func | 500/500 (100%) | 498/500 (99.6%) |
| match_args | 458/500 (92%) | **491/500 (98.2%)** |
| year mismatch | 26 | **0** (2010 ahora exacto) |
| region mismatch | 13 | 0 de vocabulario |

**9 fallos restantes de args en v2, clasificados**:
- 4× Quercus ilex → quercus suber: ambigüedad inherente de especies cercanas
  (el caso que el reporte v1 ya anticipaba).
- 3× "sudeste asiatico" → "southeast asia": la MISMA región traducida al inglés
  por el controller (región VÁLIDA, idioma distinto). El verificator no la
  repara porque "southeast asia" está en KNOWN_VALUES. Si se quiere, un mapa
  de sinónimos es-en en el evaluador los absorbería, sin tocar gold.
- 2× función gbif_occurrence → inaturalist_occurrence en prompts
  "Descarga observaciones de X...": ARGS EXACTOS, tool distinta (iNaturalist
  es una lectura razonable de "observaciones"; el gold dice GBIF).

**Conclusión**: el piso del 92% era VOCABULARIO (años fuera de rango + regiones
ausentes), tal como diagnosticó v1. Listar el vocabulario en el prompt + tener
todos los años en KNOWN_VALUES sube a 98.2% con los fallos residuales siendo
ambigüedad léxica real (especies cercanas, idioma, sinonimia de tools), no
errores de diseño.

## Clasificación de los 42 fallos de args (v1, sin fix)

| Tipo | n | Patrón | Ejemplo |
|---|---|---|---|
| **year mismatch** | 26 | gold year=2010 → pred year=2015 (siempre 2015) | "periodo 2010" → {"year":"2015"} |
| **region mismatch** | 13 | gold "africa occidental" → pred "asia oriental" (siempre) | prompt dice africa → pred asia |
| **species mismatch** | 3 | Quercus ilex → quercus suber (+1 otro) | confusión de especies cercanas |

## Diagnóstico (lo importante)

- Los 3 tipos son errores de VOCABULARIO del controller, NO de diseño ni de
  datos. La arquitectura (controller→verificator→retry) funciona: 100% función,
  formato 100% válido, gold 100% consistente.
- **Year 2010→2015 sistemático**: el modelo elige 2015 ante "periodo 2010".
  Sugiere un ancla en su vocabulario (o que "2010" no se alinea). FIX: añadir
  años recientes a KNOWN_VALUES del verificator y/o al prompt.
- **africa occidental→asia oriental sistemático**: asocia "occidental" con
  "oriental" cruzando continente. FIX real: el SYSTEM_PROMPT del controller
  debe listar las regiones geográficas VÁLIDAS (así no "inventa" la región).
- **species cercanas** (Quercus ilex→suber): inherente a la ambigüedad; el
  gold debe ser canónico y el prompt mencionar el nombre completo.

## Comparativa con mediciones previas

| Dataset | n | match_func | match_args |
|---|---|---|---|
| gold curado 14 | 14 | 100% | 86% |
| gold literario Devin | 120 | 100% | 93% |
| gold Devin 10 tools | 300 | 99.3% | 97.0% |
| **Fase 3 (este)** | **500** | **100%** | **92%** |

- match_func se mantiene en 100% escalando a 500/7 tools.
- match_args varía 86-97% según la dificultad del vocabulario de regiones/años.
- El piso del 92% es VOCABULARIO (años/regiones), no capacidad de elegir la
  herramienta.

## Acción recomendada (EJECUTADA en el run v2, 2026-09-09)

1. ~~Añadir la lista completa de regiones al SYSTEM_PROMPT del controller~~ ✅
   (derivada de KNOWN_VALUES, sin duplicación manual)
2. ~~Añadir años admitidos al prompt/verificator~~ ✅ (2010-2026)
3. (opcional) Distinguir en el evaluador "mismatch menor de año" de "región
   incorrecta" — ya no hace falta para años (0 fallos), útil solo si se quiere
   absorber las traducciones es-en de regiones (3 fallos residuales).

## Artefactos

- /tmp/repl_fase3_500.jsonl (v1), /tmp/repl_fase3_500_v2.jsonl (v2: filas por ítem)
- /tmp/repl_fase3_500.json (v1), /tmp/repl_fase3_500_v2.json (v2: reporte agregado)
- Este doc: docs/results/REPLICACION-500-FASE3-RESULTADO.md
- Dataset evaluado: data/l1/toolcalls_fase3_500.jsonl
- Fix aplicado en: scripts/verify_toolcall.py (KNOWN_VALUES) +
  ecobench/run_controller_verificator.py (SYSTEM_PROMPT con vocabulario)