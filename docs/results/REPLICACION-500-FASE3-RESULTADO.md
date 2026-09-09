# REPLICACIÓN CONTROLLER — FASE 3 (500 tool-calls, 7 tools) — RESULTADO

Fecha: 2026-09-09 · Modelo: deepseek-v4-flash (ollama local) · 500 prompts · 784s

## Resultado global

| Métrica | Valor |
|---|---|
| match_func | **500/500 (100%)** |
| match_args | 458/500 (92%) |
| Tiempo | 784 s (~1.6 s/prompt) |

- El controller **nunca elige la herramienta equivocada** en 500 casos, 7 tools.
- El 92% de los argumentos son exactos (oro limpio, sin gold inconsistente aquí:
  este dataset fue generado con consistencia prompt↔args verificada 0 fallos).

## Clasificación de los 42 fallos de args (8%)

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

## Acción recomendada (para subir de 92% a ~98%)

1. Añadir la lista completa de regiones al SYSTEM_PROMPT del controller
   (KNOW_VALUES ya la tiene en el verificator; el PROMPT no).
2. Añadir años admitidos al prompt/verificator.
3. (opcional) En el evaluador, distinguir "mismatch menor de año" de "región
   incorrecta" — el año es un rango, la región no.

## Artefactos

- /tmp/repl_fase3_500.jsonl (filas por ítem)
- /tmp/repl_fase3_500.json (reporte agregado)
- Este doc: docs/results/REPLICACION-500-FASE3-RESULTADO.md
- Dataset evaluado: data/l1/toolcalls_fase3_500.jsonl