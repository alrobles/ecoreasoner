# Protocolo experimental para dLLM en KUHPC

## 1. Objetivo

Determinar si un modelo de difusión discreta puede ofrecer una ventaja
medible de calidad/latencia frente a un modelo autoregresivo (AR) bajo un
presupuesto de entrenamiento comparable y sobre el hardware disponible en
KUHPC. El primer experimento debe producir una conclusión reproducible, no
intentar replicar Mercury o LLaDA a escala comercial.

El protocolo usa como base:

- `paper/main_v2.pdf`: revisión DiDAL y sus 22 referencias verificadas.
- `research/paper_graph.json` y `research/literature_map.md`: minería
  recursiva de 40 papers únicos en tres niveles.

## 2. Preguntas de investigación

### RQ1 — Entrenabilidad

¿Puede un masked dLLM entrenarse de forma estable en MI210 y alcanzar calidad
comparable a un AR de la misma arquitectura, tokenizer, datos y número de
parámetros?

### RQ2 — Tipo de ruido

Con parámetros, tokens únicos y presupuesto de FLOPs igualados, ¿qué diferencia
hay entre ruido absorbing-mask y ruido uniforme?

### RQ3 — Ventaja de inferencia

¿La generación paralela reduce latencia o aumenta throughput sin degradar
calidad funcional cuando se mide con el mismo hardware, longitud, batch,
precisión y criterio de parada?

### RQ4 — Escalado en MI210

¿Cómo cambia la eficiencia al pasar de 1 GPU a varias GPUs y varios nodos?
La memoria HBM agregada no se considerará evidencia de escalado hasta medir
comunicación, utilización y tokens/s.

## 3. Hipótesis y criterios de decisión

Estas son hipótesis experimentales, no resultados publicados:

| ID | Hipótesis | Criterio mínimo para continuar |
|---|---|---|
| H1 | MDLM/masked diffusion es el primer baseline reproducible más seguro. | Pérdida estable, sin NaN/Inf, y checkpoint recuperable durante 3 evaluaciones consecutivas. |
| H2 | El dLLM puede acercarse al AR pequeño con coste adicional acotado. | En validación, diferencia de perplexity o pérdida normalizada ≤15% y ninguna caída funcional >10% frente al AR. |
| H3 | La ventaja principal aparecerá en throughput, no necesariamente en coste de entrenamiento. | ≥1.5× tokens/s a calidad equivalente o una reducción ≥25% de latencia p50. |
| H4 | Uniform diffusion merece escalar solo si supera a mask en un régimen controlado. | Mejora ≥3% en calidad a FLOPs equivalentes, o mejora ≥20% en calidad/FLOP. |

Un resultado negativo es válido. No se declarará viabilidad por velocidad si la
calidad, la memoria o la estabilidad no son reproducibles.

## 4. Diseño mínimo controlado

### 4.1 Variables constantes

- Tokenizer y vocabulario idénticos en todos los tratamientos.
- Corpus, orden de datos, deduplicación y partición train/validation/test
  idénticos.
- Arquitectura Transformer, número de capas, hidden size, cabezas y contexto
  idénticos entre AR y dLLM.
- Semilla, inicialización, optimizer, weight decay, warmup y clipping fijados.
- Precisión BF16 si pasa la prueba numérica; FP32 para diagnóstico.
- Mismo número de tokens únicos vistos y misma longitud máxima.
- Evaluación con el mismo número de muestras, batch y hardware.

### 4.2 Tratamientos

| Tratamiento | Objetivo | Propósito |
|---|---|---|
| AR | Causal next-token prediction | Control de calidad y coste de entrenamiento |
| MDLM-mask | Absorbing-mask/SUBS o implementación equivalente | Baseline dLLM reproducible |
| dLLM-uniform | Corrupción uniforme discreta | Comparar el tipo de ruido |
| MDLM-mask + decoding acelerado | Masked dLLM con caching/parallel decoding | Medir la ventaja de inferencia |

La variante acelerada no se entrenará como un tratamiento separado. Se aplicará
al mismo checkpoint MDLM y se comparará contra su sampler no acelerado.

### 4.3 Tamaños y réplicas

| Fase | Modelo nominal | GPUs | Réplicas |
|---|---:|---:|---:|
| Calibración | 10M–50M | 1 MI210 | 1 |
| Baseline | 110M | 1 MI210 | 3 semillas |
| Confirmación | 350M–1.3B | 1–3 MI210 | 2 semillas |
| Escalado | checkpoint que pase gates | 3–24 GPUs | 1 por configuración |

Los 110M se eligen por su conexión con los experimentos pequeños de MDLM y
porque permiten depurar ROCm sin convertir la primera iteración en un problema
de infraestructura. El salto a 1.3B no se autoriza por disponibilidad de HBM;
requiere que la fase anterior pase los gates de estabilidad y calidad.

## 5. Fases de ejecución

### Fase 0 — Compatibilidad y presupuesto

Registrar en cada nodo y job:

- GPU, memoria libre, ROCm, driver, PyTorch y versión de kernels.
- BF16/FP16/FP32, SDPA y kernels de atención disponibles.
- interconexión, topología visible, RCCL/NCCL y ancho de banda efectivo.
- longitud de secuencia, microbatch, gradient accumulation y peak memory.
- tokens/s, step time, utilización de GPU y consumo energético si está
  disponible.

Ejecutar una prueba sintética de atención y un forward/backward pequeño antes de
usar datos reales. Guardar el resultado como `hardware_manifest.json`.

### Fase 1 — Baseline 110M

Entrenar AR y MDLM-mask con el mismo corpus y presupuesto de tokens. Evaluar
cada intervalo fijo en tokens, no solo en steps, y conservar todos los
configuraciones en un manifiesto. Repetir con tres semillas para estimar
variabilidad.

Medir:

- pérdida de entrenamiento y validación;
- tiempo por step, tokens/s y memoria pico;
- pérdida frente a tokens vistos y frente a FLOPs estimadas;
- estabilidad numérica y capacidad de reanudar desde checkpoint.

### Fase 2 — Comparación de ruido

Entrenar MDLM-mask y dLLM-uniform con la misma arquitectura, tokens únicos,
semillas y presupuesto de FLOPs. Reportar también el coste del sampler y la
calidad con 1, 2, 4, 8, 16 y 32 pasos de denoising cuando cada configuración
sea válida.

No comparar solo perplexity: las cotas variacionales de difusión no deben
presentarse como perplexity AR exacta.

### Fase 3 — Escalado

Usar únicamente el tratamiento que haya pasado las fases anteriores. Medir
1, 3, 6, 12 y, si el scheduler lo permite, 24 GPUs. Separar:

- escalado dentro de un nodo;
- escalado entre nodos;
- data parallelism;
- sharding/FSDP o equivalente.

Para cada punto registrar:

```text
global_batch, sequence_length, precision, world_size, nodes,
tokens_per_second, step_time, scaling_efficiency, peak_memory,
communication_fraction, validation_loss
```

La eficiencia se calculará como `throughput(N) / (N * throughput(1))`.

### Fase 4 — Inferencia y código

Aplicar el sampler baseline y una sola técnica de aceleración por vez. Para
código, usar HumanEval y MBPP con ejecución sandboxed, además de un conjunto
posterior a la fecha de corte para reducir contaminación. Reportar pass@1 y
pass@k con el mismo número de muestras.

Cada punto de throughput debe incluir:

- GPU y versión de software;
- batch, longitud de prompt y longitud generada;
- número de pasos de denoising;
- temperatura y criterio de parada;
- warmup, número de repeticiones, p50 y p95;
- tokens/s de salida y memoria pico.

No usar las cifras H100 de Mercury como objetivo directo. Sirven como evidencia
de posibilidad industrial, no como baseline comparable para MI210.

## 6. Gates go/no-go

### Gate A — Hardware

Continuar solo si el forward/backward BF16 es estable, el job puede reanudar y
la medición de memoria es repetible en dos ejecuciones.

### Gate B — Entrenamiento

Continuar al estudio de ruido solo si MDLM-mask completa el presupuesto de
tokens sin divergencia y mejora sobre un modelo aleatorio claramente por debajo
de la pérdida inicial.

### Gate C — Calidad

Continuar al escalado solo si el tratamiento seleccionado queda dentro de H2 o
la diferencia está explicada por un trade-off explícito de coste/latencia.

### Gate D — Inferencia

Declarar ventaja solo si se mantiene la calidad funcional dentro del margen
predefinido y la medición incluye el coste total del sampler, no únicamente el
forward de una iteración.

## 7. Artefactos obligatorios

Cada ejecución debe producir:

```text
configs/<run>.yaml
manifests/<run>.json
logs/<run>.jsonl
checkpoints/<run>/
metrics/<run>.parquet
reports/<run>.md
```

El manifiesto debe contener commit de código, dataset/versionado, tokenizer,
semilla, hiperparámetros, hardware, job ID, duración, checkpoints y comandos
exactos. Los datos grandes y checkpoints no se deben subir al repositorio; el
repositorio debe conservar metadatos, hashes y rutas de almacenamiento.

## 8. Resultado esperado

El primer resultado debe ser una curva calidad–tokens–FLOPs y una curva
calidad–latencia–pasos de denoising para AR, MDLM-mask y, si pasa los gates,
uniform diffusion. La conclusión debe clasificar el proyecto como:

1. **viable como piloto**, si es estable y produce una ventaja medible;
2. **viable con restricciones**, si la ventaja depende de un régimen concreto;
3. **no viable en la configuración probada**, si el coste o la calidad impiden
   una comparación favorable.

La clasificación se aplica a la configuración medida, no se extrapola
automáticamente a todos los tamaños, datasets o topologías de KUHPC.
