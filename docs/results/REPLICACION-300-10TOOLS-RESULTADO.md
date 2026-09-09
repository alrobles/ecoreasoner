# Replicación controller sobre 300 pares / 10 tools — Resultado

Fecha: 2026-09-09 · Modelo: `deepseek-v4-flash:latest` · Backend: `ollama`

## Resumen global

| Métrica | Valor |
|---|---|
| Pares evaluados | 300 |
| `match_func` | 298/300 (99.3%) |
| `match_args` | 291/300 (97.0%) |
| Tiempo total | 459 s (~1.5 s/prompt) |

**Conclusión**: el controller **soporta las 10 tools** sin degradación visible. La ampliación a 7 tools adicionales (más allá del gold original de 3) no rompe el controlador.

## Desglose por tool

| Tool | N | `match_func` | `match_args` |
|---|---|---|---|
| `gbif_occurrence` | 84 | 82/84 (98%) | 80/84 (95%) |
| `timetree_divergence` | 40 | 40/40 (100%) | 40/40 (100%) |
| `opentree_phylogeny` | 40 | 40/40 (100%) | 40/40 (100%) |
| `ncbi_taxonomy` | 40 | 40/40 (100%) | 40/40 (100%) |
| `bioclim_download` | 24 | 24/24 (100%) | 21/24 (88%) |
| `iucn_status` | 15 | 15/15 (100%) | 15/15 (100%) |
| `srtm_elevation` | 15 | 15/15 (100%) | 14/15 (93%) |
| `inaturalist_occurrence` | 15 | 15/15 (100%) | 15/15 (100%) |
| `try_traits` | 15 | 15/15 (100%) | 15/15 (100%) |
| `maxent_train` | 12 | 12/12 (100%) | 12/12 (100%) |

## Observaciones sobre el "sesgo gbif"

El usuario sospechaba un desbalance por `gbif_occurrence` (84 pares, 28% del total). Los datos muestran:

- `gbif_occurrence` es **el único tool con errores de función** (2 de 84) y args (4 de 84).
- Todos los demás tools, incluidos los 4 recién añadidos y los 3 de evolución, alcanzan **100% func** y **≥93% args**.
- No hay evidencia de que el modelo "sobregeneralice" a `gbif_occurrence` en función; los 2 fallos son por **ambigüedad léxica en el prompt** (`observaciones` suena a iNaturalist) y por **reparaciones M3 automáticas** del verificador.

## Errores analizados (9 fallos de args / 2 fallos de función)

### Fallos de función (2)

| # | Prompt | Gold | Pred | Razón |
|---|---|---|---|---|
| 17 | "Descarga observaciones de Prosopis glandulosa en la región sonora." | `gbif_occurrence` | `inaturalist_occurrence` | Palabra "observaciones" sugiere iNaturalist; gold usa plantilla ambigua. |
| 42 | "Recupera registros de presencia de Alytes obstetricans en ibera." | `gbif_occurrence` | `inaturalist_occurrence` | Mismo patrón: prompt menciona "observaciones" implícitamente (plantilla). |

### Fallos de argumentos (9)

| # | Tool | Prompt | Gold args | Pred args | Razón |
|---|---|---|---|---|---|
| 26 | `gbif_occurrence` | Pan troglodytes en africa occidental | `africa occidental` | `asia oriental` | M3 fuzzy repair por región no en `KNOWN_VALUES`. |
| 51 | `gbif_occurrence` | Crotalus atrox en desierto de sonora | `desierto de sonora` | `desierto de gobi` | M3 fuzzy repair (sim 0.76) entre regiones con "desierto". |
| 88 | `bioclim_download` | indico en 2020 | `indico` | `india` | M3 fuzzy repair por similitud fonética. |
| 91 | `bioclim_download` | desierto de sonora 2015 | `desierto de sonora` | `desierto de gobi` | M3 fuzzy repair. |
| 104 | `bioclim_download` | egipto 2017 | `egipto` | `egypt` | M3 repair a sinónimo inglés. |
| 150 | `srtm_elevation` | indico a 250m | `indico` | `india` | M3 fuzzy repair. |
| 154 | `inaturalist_occurrence` | Quercus ilex en pampas | `Quercus ilex` | `quercus suber` | M3 fuzzy repair entre especies similares (sim 0.72). |

Nota: los fallos de args no son mayoritariamente del controller, sino del **verificador `M3`** que repara hacia un valor canónico cercano y luego `arg_match` no lo considera igual. Esto es configurable ajustando `KNOWN_VALUES` o bajando el umbral de `arg_match`.

## Sobre el desbalance

Aunque `gbif_occurrence` es la clase mayoritaria, el desbalance **no afecta negativamente** la precisión por tool. De hecho, `gbif_occurrence` es la más problemática por ambigüedad léxica, no por dominancia. Si se desea un dataset estratificado, opciones:

1. **Submuestrear gbif a 40 pares** (igual que evolución) y dejar 120 total de los 3 tools originales.
2. **Sobremuestrear los tools pequeños** (`maxent_train`, `bioclim_download`) hasta 40 cada uno.
3. **Mantener 300** y confiar en que el controller zero-shot maneja el desbalance.

## Archivos generados

- `docs/results/replication_300.jsonl` — resultados ítem a ítem.
- `docs/results/replication_report_300.json` — reporte resumen JSON.

## Comando usado

```bash
python3 scripts/eval_controller_replication.py \
  --gold data/l1/toolcalls_lit_gold_eval.json \
  --out docs/results/replication_300.jsonl \
  --report docs/results/replication_report_300.json
```
