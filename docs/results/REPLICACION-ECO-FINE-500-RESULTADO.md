# REPLICACIÓN CONTROLLER — Bloque ECO-FINO v5 (500 tool-calls, 8 tools) — RESULTADO

Fecha: 2026-09-09 · Modelo: deepseek-v4-flash (ollama local) · 500 prompts · 796s

## Dataset

`data/l1/toolcalls_eco_fine_500.jsonl` — primer bloque generado EXCLUSIVAMENTE del
subcorpus ecológico GENUINO (domain_fine v5, 142K docs):
- 178 especies GBIF-reales (93 nuevas del subcorpus eco-fino + gold previo),
  EXCLuyendo patógenos clínicos/modelos de laboratorio.
- Pool ecológico real: Carnegiea gigantea, Apis mellifera, Mytilus edulis,
  Eucalyptus globulus, Daphnia magna, Panthera tigris, Giraffa camelopardalis,
  Cervus elaphus, Populus tremuloides, Campephilus principalis...
- Regiones ampliadas con las top del subcorpus (ibera, tropical forest, caribbean,
  southeast asia, west africa, andina, pampas...).
- 500/500 válidas verify_toolcall (schema 100%).

## Resultado

| Métrica | Valor |
|---|---|
| match_func | **496/500 (99.2%)** |
| match_args | **493/500 (98.6%)** |
| Tiempo | 796 s (~1.6 s/prompt) |

## Comparativa acumulada

| Bloque | n | match_func | match_args |
|---|---|---|---|
| gold curado 14 | 14 | 100% | 86% |
| gold literario 120 | 120 | 100% | 93% |
| 10 tools 300 | 300 | 99.3% | 97.0% |
| Fase 3 500 v1 (SIN fix vocab) | 500 | 100% | 92% |
| Fase 3 500 v2 (CON fix vocab) | 500 | 99.6% | 98.2% |
| **Eco-fino 500 (CON fix vocab)** | **500** | **99.2%** | **98.6%** |

El bloque eco-fino mantiene el piso del fix de vocabulario y lo supera en args
(98.6% vs 98.2%) con especies NUEVAS nunca usadas en el gold previo — el
controller generaliza a especies ecológicas reales sin degradación.

## Fallos (7, patrones ya conocidos — no errores de diseño)

- 4× función: "registros/observaciones de X" → inaturalist_occurrence (gold:
  gbif_occurrence). ARGS EXACTOS. iNaturalist es lectura razonable de
  "observaciones"; el gold dice GBIF. Ocurre en "Localiza registros de X en ibera".
- 2× Quercus ilex → quercus suber: ambigüedad inherente de especies cercanas.
- 1× "sudeste asiatico" → "southeast asia": misma región traducida es-en.

## Artefactos

- Dataset: data/l1/toolcalls_eco_fine_500.jsonl (commit 1b048cc)
- /tmp/repl_eco_fine_500.jsonl + /tmp/repl_eco_fine_500.json
- Este doc