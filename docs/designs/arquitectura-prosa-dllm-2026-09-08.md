# ARQUITECTURA DE SOFTWARE — Prosa dLLM EcoReasoner

Fecha: 2026-09-08 · Autor: Devin (SWE-1.7) · Rama: `devin/prosa-dllm-software-design`

Este documento es la **contraparte de software** del plan
`plan-prosa-dllm-2026-09-08.md`. Define módulos, flujos de datos, formatos y
**implementación** de la Fase B.1/B.2/E.

> Estado: build_prosa_v8, mask_schedule/whole_stage, suite_smoke v2 y los
> techos LLaDA-8B/MDLM-OWT están implementados y pasan `python3 -m py_compile`.
> Pendiente: correr en HPC y ajustar hiperparámetros.

---

## 0. Principios de diseño

1. **Reutilizar el harness existente** (`harness/configs/`, `harness/run_micro.py`,
   `harness/report.py`, `harness/suite_smoke.py`). Cada fase es un micro-run
   declarativo con criterio pre-registrado.
2. **Config-driven**: cada fase tiene un YAML en `harness/configs/`; no tocar
   `scripts/*.slurm` salvo para adaptar nombres de variables.
3. **No romper la línea Fase A-E**: los cambios a `train_mdlm_moe.py` deben
   ser **retro-compatibles** (flags desactivables; `warmup` ya es así).
4. **Prosa y esqueleto son dos datasets separados**, pero con el mismo
   tokenizador y `max_len`. Un mismo modelo se entrena sobre uno u otro o un
   mix según YAML.
5. **Techo de medición separado**: `scripts/ceiling_*` no son parte del
   entrenamiento; evalúan modelos públicos en el mismo `suite_smoke`.

---

## 1. Componentes y flujo de datos

```
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                │
│  data/train_corpus_v7_clean.jsonl                          │
│  data/fulltext_*.jsonl (PMC)                               │
│  data/arxiv/fulltext/fulltext_corpus.jsonl                 │
│  external: ecoseek-litdump/*                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
            scripts/build_prosa_v8.py
            (dedup, filter, EOS packing, tokenize)
                       │
            data/prosa_v8/train_corpus_v8.jsonl
            data/prosa_v8/train_ids_v8.npy
            data/prosa_v8/v8_report.json
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  TRAINER (modificado)                                       │
│  scripts/train_mdlm_moe.py (+ hetero)                       │
│  · mask_type: "random" | "span" | "schedule"                │
│  · mask_schedule: "uniform" | "cosine" | "mutate"           │
│  · whole_stage: true | false                                │
│  · stage_labels: [...]                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
            checkpoints en runs/<tag>/
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  EVALUACIÓN                                                 │
│  harness/suite_smoke.py (extensión)                         │
│  · sampling: "random" | "low_confidence" | "semi_ar"         │
│  · best_of_n: int                                            │
│  · ceiling: llada8b | mdlm-owt                               │
│  scripts/verdict.py (ya listo)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Módulo `scripts/build_prosa_v8.py`

**Responsabilidad**: convertir todo el prosa crudo en un corpus entrenable.

### Inputs (configurables por YAML)

```yaml
sources:
  - path: data/train_corpus_v7_clean.jsonl
    weight: 0.50
  - path: data/arxiv/fulltext/fulltext_corpus.jsonl
    weight: 0.25
  - path: /beegfs/.../ecoseek-litdump/litdump.jsonl
    weight: 0.25
dedup:
  keys: [pmid, doi, pmcid]
  fuzzy_title_threshold: 0.95
filter:
  min_chars: 256
  max_chars: 8192
  quality: "heuristic"  # "heuristic" | "perplexity" (futuro)
packing:
  eos_token: "<|endoftext|>"
  max_seq: 768
tokenizer: /beegfs/.../LLaDA
outdir: data/prosa_v8
```

### Pipeline

1. **Load** de todas las fuentes, normalizando a `{doc_id, title, abstract, text, year, domain, pmid, doi, pmcid}`.
2. **Dedup** por claves exactas; luego fuzzy por título (similitud coseno sobre hashes).
3. **Limpieza**: quitar encabezados de copyright, disclaimers, referencias
   bibliográficas planas, secciones tipo "Acknowledgements".
4. **Boilerplate removal**: identificar plantillas de Methods (e.g. frases
   repetidas >=N veces) vía n-gramas y eliminarlas.
5. **Quality filter** (fase 1 heurística, fase 2 con modelo referencia):
   - longitud, ratio de palabras reales, ratio de tokens repetidos,
   - entropía de caracteres.
6. **EOS packing**: concatenar documentos con `eos_token` entre ellos y cortar
   en ventanas de `max_seq` tokens. Cada ventana debe empezar con el token
   EOS del documento previo si cortamos a mitad — esto es crítico para que
   el modelo aprenda fronteras.
7. **Tokenizar** con LLaDA tokenizer, guardar `train_ids_v8.npy` (uint32 o uint16
   si cabe, pero vocab es 126080 → uint32).
8. **Reporte** `v8_report.json`: n docs, n tokens, n windows, dominios, años,
   métricas de calidad.

### Output schema

```
data/prosa_v8/
  train_corpus_v8.jsonl
  train_ids_v8.npy
  v8_report.json
  v8_config.yaml
```

---

## 3. Trainer v2 (`scripts/train_mdlm_moe.py`)

Añadir flags y modificar **solo** las funciones de masking y LR. No tocar DDP,
resume, olas.

### Nuevos argumentos

```python
ap.add_argument("--mask_schedule", default="fixed",
                choices=["fixed", "uniform", "cosine", "mutate"],
                help="como samplear t (fraccion de tokens enmascarados)")
ap.add_argument("--mask_schedule_args", type=str, default="",
                help="JSON con args (p.ej. '{\"b_l\":0.0,\"b_h\":0.8}')")
ap.add_argument("--whole_stage", action="store_true",
                help="si los datos son esqueleto, enmascarar etapa entera")
ap.add_argument("--stage_labels", type=str,
                default="OBSERVACION,HIPOTESIS,PREDICCION,EVIDENCIA,CONCLUSION",
                help="etiquetas de esqueleto para whole_stage")
```

### Funciones a modificar

- `build_span_mask(...)`: aceptar `t` variable en vez de fijo.
- `build_mask(batch, schedule)`: samplear `t` por ejemplo si
  `mask_schedule="uniform"`; construir `n_masked = int(seq_len * t)` y luego
  span o random.
- `main()`: en cada step llamar a `sample_mask_schedule(step)`.
- `whole_stage`: parsear texto de tokens, encontrar índices de
  `[<stage_label>]`, enmascarar desde el token posterior hasta el siguiente
  `[`.
- LR: ya está `_set_lr` en hetero; propagar a `train_mdlm_moe.py` (función
  `_set_lr` idéntica, llamada en `opt.step()`).

### Backward compatibility

- `mask_schedule=fixed` + `--mask_p 0.15` reproduce exactamente el
  comportamiento anterior.
- `whole_stage=false` no cambia nada.

---

## 4. Decoder v2 (`harness/suite_smoke.py`)

Extender con argumentos:

```python
ap.add_argument("--sampling", default="random",
                choices=["random", "low_confidence", "semi_ar"])
ap.add_argument("--best_of_n", type=int, default=1)
ap.add_argument("--rerank", action="store_true",
                help="usar denoise-loss para rerankar best_of_n")
```

### Implementación deseada

1. **low_confidence unmasking** (LLaDA/Fast-dLLM):
   - Por paso, para las posiciones enmascaradas, predecir logits.
   - Calcular probabilidad del token elegido: `p = softmax(logits)[argmax]`.
   - Desenmascarar las `k` posiciones con mayor `p`.
   - Remascarar las posiciones con menor `p` (o dejarlas para siguiente paso).

2. **best-of-N rerank**:
   - Generar `N` continuaciones con temperatura/stochasticidad.
   - Para cada una, computar denoise-loss promedio sobre posiciones enmascaradas.
   - Elegir la de menor loss.

3. **semi_ar** (futuro, post-BD3-LM):
   - Generar bloques L2R; dentro de cada bloque, difusión.

### Riesgo

`harness/suite_smoke.py` se usa en producción. Para no romper PR #1, la
extensión se hace **agregando opciones** con defaults a los valores actuales.

---

## 5. Techos de medición (`scripts/ceiling_*.py`)

### LLaDA-8B (`scripts/ceiling_llada8b.py`)

- Cargar `GSAI-ML/LLaDA-8B-Base` con `trust_remote_code=True`.
- Recibir `--pairs` y `--config` (aunque el config sea solo por formato; el
  modelo no se recarga).
- Implementar la lógica de `suite_smoke`: para cada par, enmascarar la
  continuación y computar `loss(ok)` vs `loss(bad)`.
- LLaDA-8B no cabe en 1 GPU pro6000 si es fp32; requiere `bf16` y quizás
  `device_map="auto"` / cpu offloading. Especificar `--device cuda` por
  defecto y `--max_len` más corto si hace falta.

### MDLM-OWT (`scripts/ceiling_mdlm_owt.py`)

- Cargar `kuleshov-group/mdlm-owt`.
- Misma interface que LLaDA-8B.
- MDLM-OWT es ~130M (parecido a nuestro modelo) y debería caber en 1 GPU.

### Slurm wrappers

- `scripts/ceiling_llada8b.slurm` (pide 1 pro6000, bf16, >=80GB VRAM).
- `scripts/ceiling_mdlm_owt.slurm` (1 pro6000, ~30 min).

---

## 6. Micro-sweep harness

Cada fase B/C es un micro-run con YAML en `harness/configs/`:

- `harness/configs/prosa-v8-155M.yaml`
- `harness/configs/skeleton-v2-155M.yaml`
- `harness/configs/sketch2text-155M.yaml`

Cada YAML especifica `data`, `mask_schedule`, `whole_stage`, `sampling`, y
`steps`. `harness/run_micro.py` valida, arma el slurm y ejecuta.

---

## 7. Orden de implementación para Hermes

### Día 1 (mañana)
- [ ] Revisar/mergear PRs abiertos.
- [ ] Ejecutar `scripts/vendor_clone.sh` en beegfs.
- [ ] Correr Fase A (batería L0-L3) si no se ha hecho.
- [ ] Empezar `scripts/build_prosa_v8.py` (stub presente en esta rama).

### Día 2-3
- [ ] Terminar `build_prosa_v8.py` y generar `data/prosa_v8/train_ids_v8.npy`.
- [ ] Implementar `mask_schedule=uniform` en `train_mdlm_moe.py`.
- [ ] Crear `harness/configs/prosa-v8-155M.yaml`.

### Día 4-5
- [ ] Lanzar micro-run prosa v8 (10K steps, ~4h GPU).
- [ ] Implementar `low_confidence` en `suite_smoke.py`.
- [ ] Evaluar con `suite_smoke` + `verdict.py`.

### Semana 2
- [ ] `whole_stage` para esqueletos.
- [ ] `sketch-to-text` dataset.
- [ ] LLaDA-8B ceiling.

### Semana 3-4
- [ ] Si no hay señal: decisión BD3-LM vs controller/verificator.

---

## 8. Apéndice: archivos a crear/ver en esta rama

| Archivo | Estado | Nota |
|---|---|---|
| `scripts/build_prosa_v8.py` | implementado | pipeline completo + selftest CPU |
| `scripts/build_prosa_v8.slurm` | implementado | slurm CPU |
| `harness/configs/prosa-v8-155M.yaml` | implementado | config declarativa del micro-run |
| `harness/configs/skeleton-v2-155M.yaml` | implementado | config con whole_stage |
| `scripts/train_mdlm_moe.py` | implementado | mask_schedule, whole_stage, lr warmup |
| `scripts/train_mdlm_moe_hetero.py` | implementado | mask_schedule, whole_stage |
| `harness/suite_smoke.py` | implementado | low_confidence, best_of_n, rerank |
| `scripts/ceiling_llada8b.py` | implementado | carga LLaDA-8B y evalúa pares |
| `scripts/ceiling_llada8b.slurm` | implementado | slurm 1x pro6000 bf16 |
| `scripts/ceiling_mdlm_owt.py` | implementado | carga MDLM-OWT |
| `scripts/ceiling_mdlm_owt.slurm` | implementado | slurm 1x pro6000 |
| `scripts/moe_v4_micro.slurm` | actualizado | acepta MASK_SCHEDULE, WHOLE_STAGE, etc. |
| `docs/designs/arquitectura-prosa-dllm-2026-09-08.md` | completo | este doc |

*Documento de arquitectura rastreable. La implementación está lista para
validación en HPC por Hermes.*
