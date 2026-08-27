# PRE-VERSIÓN de ScienceAgentBench para EcoBench — plantilla para crear preguntas similares

> ReumanLab · EcoReasoner · 2026-08-26
> Propósito: recorte de las tareas GIS/espacial-ecológicas de ScienceAgentBench (CC BY 4.0,
> abierto, del `osunlp/ScienceAgentBench` version `verified`), listo para **extender** con
> preguntas propias del mismo estilo.
> NO se evalúa todavía (sin baseline). El dataset completo vive en `ecobench_eval.json`.

---

## Cómo construir una pregunta similar (receta)

Cada pregunta de ScienceAgentBench tiene esta anatomía:

| Campo | Qué es | Ejemplo |
|---|---|---|
| `task_inst` | la instrucción natural en 1-3 frases | "Calcula el % de deforestación en Rondônia en un buffer de 5.5km de carretera" |
| `domain` | el área | "Geographical Information Science" |
| `data` | el dataset de entrada real | dataset_folder_tree |
| `gold_program` | el programa que produce la salida (solución) | `elk_new.py` |
| `output_fname` | el artefacto esperado | `pred_results/Elk_Analysis.png` |
| `eval_script` | el evaluador de la salida | `eval_elk_analysis.py` |

**Patrón de pregunta ecológica (lo que el lab puede replicar):**
1. **Entorno + animal/ecosistema + pregunta operable.** Ej: "para evaluar aptitud de hábitat de puma…"
2. **Acción espacial + técnica.** "calcular ruggedness desde elevación / distancia a caminos / integrar en costo"
3. **Salida verificable (artefacto).** `pred_results/<X>.png|csv|tif`
4. **Dato real.** un dataset GIS (GBIF, bioclim, ocurrencias).

**Formato de ítem (esquema EcoBench):**
```json
{
  "id": "eco-<código>-NNN",
  "source": "own",
  "family": "GIS-eco|SDM|phylo|bioclim|tool-call",
  "question": "Instrucción natural de la tarea.",
  "execution": {
    "lang": "R|python",
    "code_hint": "esqueleto/paquete (NO la solución)",
    "data": "ruta o ref al dataset real",
    "expected": {"type":"file_artifact|numeric|schema|state",
                 "value":"<artefacto o valor>", "tolerance":0.0}
  },
  "verification": "execution_code",
  "license": "own",
  "split": "eval_holdout|train"
}
```

---

## Pre-versión: las 10 tareas GIS-ecológicas (extraídas de ScienceAgentBench verified)

| ID | Sistema/animal | Tarea (resumen) | Artefacto de salida | Split actual |
|---|---|---|---|---|
| sab-4 | Elk (movements) | Estimar home ranges + preferencia hábitat + clusters espaciales (geopandas) | `Elk_Analysis.png` | eval |
| sab-21 | Rondônia (deforestación) | % deforestación en buffer 5.5km de carretera | `deforestation_rate.csv` | train |
| sab-23 | Rondônia (deforestación) | Predecir deforestación con futuras carreteras | `predictedRiskyArea.png` | train |
| sab-32 | Coral & esponja | Análisis raster elevación/factores ambientales (rasterio) | `CoralandSponge.png` | eval |
| sab-46 | Mountain lion | Rugosidad de terreno desde elevación (rasterio) | `ruggedness.png` | eval |
| sab-47 | Mountain lion | Distancia de hábitat a carreteras (vector, Euclidiana) | `distance_to_habitat.png` | train |
| sab-53 | Mountain lion | Reclasificar cobertura de suelo / estado protegido | `landCover_reclassified.tif` | train |
| sab-54 | Mountain lion | Integrar rugosidad+distancia+cobertura+costo en corredor (pesos) | `mountainLionCorridor.png` | eval |
| sab-87 | Clima (N. America) | NetCDF temperatura, ajuste polinómico cuadrático (SciTools/iris) | `polynomial_fit_pred.csv` | eval |
| sab-89 | Street trees (SF) | % NULL de especies por región (geopandas/geoplot) | `trees_count_vis.png` | train |

**Cómo usar esta pre-versión:** copia una fila, cambia el sistema (animal/región) y tus propios datos,
mantén el mismo esquema. Ej. reemplaza Elk→`<mi especie>`, Rondônia→ `mi región`, rasterio→ tu
SDM. Así generas preguntas DEL MISMO ESTILO con el formato de verificación por ejecución listo.

---

*Pre-versión sin medir. Para extender con más preguntas, rellena el formato del §1.*