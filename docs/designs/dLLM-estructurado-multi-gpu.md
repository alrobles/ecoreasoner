# dLLM con representación estructurada — Análisis de viabilidad multi-GPU

> Fecha: 2026-09-10.
> Contexto: V3.1 (curriculum+role) y la familia MLM no rompen L3. El usuario
> pregunta por la Opción 3 (representación estructurada) y enfatiza que hay
> muchas tarjetas GPU disponibles, por lo que el prototipo requiere más
> ingeniería en el uso del hardware.

---

## 1. Resumen ejecutivo

**La representación estructurada no es una solución mágica al problema L3.**
El diagnóstico raíz sigue siendo que la **señal de entrenamiento** (token-level
MLM) no premia la discriminación inferencial. Cambiar la representación puede
ayudar a condensar la señal, pero solo funciona si la pérdida y los datos
piden explícitamente razonamiento.

**Si tienes muchas GPUs, el cuello deja de ser la memoria y pasa a ser:**
- Eficiencia de escalado (NCCL, topología, all-reduce).
- Estrategia de paralelismo adecuada.
- Cantidad y calidad del corpus.
- Diseño del objetivo de entrenamiento.

**Conclusión anticipada:** la ruta más provechosa con multi-GPU no es entrenar
un dLLM más grande con la misma pérdida, sino combinar **representación
estructurada + objetivo contrastivo + datos denso** en un modelo de mayor
capacidad (350M–1B), usando FSDP2 en 2-4 pro6000.

---

## 2. Tres sub-opciones de representación estructurada

### 2.1 HDLM — Hierarchical Diffusion Language Model

| Aspecto | Detalle |
|---|---|
| Idea | Añadir tokens de cluster entre palabra y `[MASK]`. El modelo predice la siguiente escala semántica. |
| Papers | Zhou et al., *Next Semantic Scale Prediction via Hierarchical Diffusion Language Models*, arXiv 2510.08632, NeurIPS 2025. |
| Datos | 10-30B tokens para 155M (paper usa 131B para 170M/425M). Mínimo smoke: 2-5B tokens. |
| GPU | 1 pro6000 entrena 155M en 2-3 días a 5B tokens; 5 pro6000 ~12h. |
| Multi-GPU | DDP suficiente; FSDP solo >1B. |
| Ventaja | Más cercano a MDLM, no requiere VAE. Mejora PPL y coherencia. |
| Desventaja | No introduce señal de discriminación inferencial. Sigue siendo token-level CE. |
| Viabilidad | Técnica, pero dudosa para L3 sin más cambios. |

### 2.2 Multi-resolution masking

| Aspecto | Detalle |
|---|---|
| Idea | Enmascarar unidades de diferentes escalas: token, span, oración, párrafo, etapa, bloque. Forzar reconstrucción de unidades semánticas grandes. |
| Papers | MDLM (NeurIPS 2024), HDLM (NeurIPS 2025), NDGC (arXiv 2606.21802), TreeDiff (arXiv 2508.01473). |
| Datos | Misma cantidad que MDLM (~100-500M tokens pueden bastar para smoke), pero el corpus necesita unidades reales de esas escalas. El skeleton ya tiene etapas. |
| GPU | Igual que MDLM. 1 pro6000, 5-6h para 10K steps. Multi-GPU DDP trivial. |
| Multi-GPU | DDP sin cambios. Cada rank samplea sus propias máscaras. |
| Ventaja | Muy barato, reusa infraestructura, no añade parámetros. Conecta con L2/L3. |
| Desventaja | Riesgo de seguir aprendiendo estructura sin contenido. Niveles imprecisos. |
| Viabilidad | **La opción más barata y directa para probar representación estructurada.** |

### 2.3 VDLM / LaDiR — latent diffusion sobre variables o thought tokens

| Aspecto | Detalle |
|---|---|
| Idea | Separar planificación semántica (diffusion en espacio latente/embedding) del renderizado de texto. |
| Papers | VDLM (arXiv 2602.15870), LaDiR (arXiv 2510.04573), LDLM (arXiv 2605.07933). |
| Datos | Muchos más. Necesita entrenar VAE/encoder + renderer + diffusion planner. Mínimo smoke: 5-10B tokens. |
| GPU | 2-3 fases de entrenamiento. 1 pro6000 insuficiente para entrenar VAE desde cero. Requiere 2-4 GPUs y días. |
| Multi-GPU | FSDP2 para VAE y diffusion. Pipeline separado. |
| Ventaja | Ataque más profundo: razonar en variables semánticas, no tokens. |
| Desventaja | Muy caro, riesgo de colapso de VAE, necesita LLM base o muchos datos. |
| Viabilidad | **Prototipo ambicioso, no primera prueba.** |

---

## 3. ¿Qué más datos requiere?

| Modelo | Parámetros | Tokens mínimos para smoke | Tokens recomendados | Corpus actual EcoReasoner |
|---|---|---|---|---|
| MDLM (v2) | 155M | 100-500M | 2-5B | ~129M esqueleto |
| Multi-resolution masking | 155M | 100-500M | 2-5B | ~129M esqueleto (suficiente para smoke) |
| HDLM | 155M | 2-5B | 10-30B | Necesita 10-20x más |
| HDLM | 350M | 5-10B | 20-50B | Necesita ampliar corpus |
| VDLM/LaDiR | 155M-350M | 5-10B | 20-100B | Necesita VAE + datos masivos |
| Denso 1B | 1B | 10-20B | 20-100B | No tenemos |

**Fuente de datos potencial:**
- Esqueleto actual: 315K docs, 129M tokens.
- arXiv eco/evo, EcoEvoRxiv, PMC, bioRxiv: podrían aportar 5-50M docs.
- Generación sintética controlada de razonamientos ecológicos.
- Pares contrafactuales (DISCO/PNS) para aumentar densidad de relaciones.

**Cálculo de Chinchilla simplificado:** ~20 tokens por parámetro para entrenamiento
eficiente. Para 1B params → 20B tokens. Para 350M → 7B tokens. Para 155M → 3B tokens.

---

## 4. ¿Qué más computo GPU requiere?

### 4.1 Modelos 155M-350M

| Setup | GPUs | Estrategia | Tiempo 10K steps | Tiempo 100K steps / 5B tokens |
|---|---|---|---|---|
| 155M | 1 pro6000 | Single | ~4h | ~2-3 días |
| 155M | 5 pro6000 | DDP | ~1h | ~12h |
| 350M | 1 pro6000 | Single | ~8h | ~5-7 días |
| 350M | 2 pro6000 | DDP/FSDP2 | ~4h | ~2-3 días |
| 350M | 4 pro6000 | FSDP2 | ~2h | ~1-1.5 días |
| 1B denso | 1 pro6000 | Single (imposible batch) | — | No viable |
| 1B denso | 2 pro6000 | FSDP2 | ~6h | ~3-4 días |
| 1B denso | 4 pro6000 | FSDP2 | ~3h | ~2 días |

### 4.2 Estimación de FLOPs

Fórmula aproximada: `FLOPs = 6 × P × T` (forward+backward).

| Modelo | Tokens | FLOPs | 2× pro6000 efectivo* | Tiempo |
|---|---|---|---|---|
| 155M | 5B | 4.7e18 | ~1.5e14 FLOP/s | ~30h |
| 350M | 7B | 1.5e19 | ~2.5e14 FLOP/s | ~17h |
| 1B | 20B | 1.2e20 | ~3.5e14 FLOP/s | ~95h (~4 días) |

*Asumiendo 30-40% MFU y comunicación intra-nodo.

### 4.3 Memoria

| Modelo | Parámetros | FP16/bf16 | Adam estados | Activaciones (batch 8, seq 768) | Total aprox por GPU |
|---|---|---|---|---|---|
| 155M | 155M | 310 MB | 930 MB | 1-2 GB | 3-5 GB |
| 350M | 350M | 700 MB | 2.1 GB | 2-4 GB | 6-10 GB |
| 1B | 1B | 2 GB | 6 GB | 4-8 GB | 15-25 GB |

155M y 350M caben en 1 pro6000 (96GB). 1B requiere FSDP2 en 2-4 GPUs o gradient checkpointing.

---

## 5. Multi-GPU: estrategias y qué funciona

### 5.1 Paralelismo disponible

| Estrategia | Uso recomendado | Overhead | Complejidad |
|---|---|---|---|
| DDP | 155M-350M en 2-8 GPUs | Bajo | Baja |
| FSDP2 | 350M-1B en 2-8 GPUs | Medio (~1.5× comunicación) | Media |
| DeepSpeed ZeRO-3 | 1B+ o MoE con EP/offload | Medio-Alto | Alta |
| Tensor Parallelism | >1B, hidden ≥ 2048 | Alto si no hay NVLink | Alta |
| Pipeline Parallelism | >32 capas | Bubble de pipeline | Media-Alta |
| Sequence Parallelism | seq_len ≥ 4K | Bajo | Media |

### 5.2 Recomendación para EcoReasoner

- **155M-350M:** DDP es suficiente y más eficiente.
- **1B denso:** FSDP2 en 2-4 pro6000.
- **MoE:** Expert Parallelism si se escala a ≥4 GPUs.
- **No mezclar generaciones:** pro6000 (cu128/Blackwell) no puede correr con A100/L40/Q6000 (cu121) en el mismo job.
- **Multi-nodo:** si no hay InfiniBand, usar `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, y considerar `sync_every` (post-localSGD) para reducir all-reduce.

### 5.3 Ingredientes de ingeniería para multi-GPU

1. **Prueba NCCL:** `scripts/nccl_test.slurm` en 2, 4, 8 pro6000.
2. **FSDP2 wrapper:** envolver cada bloque con `torch.distributed.fsdp.fully_shard` y `MixedPrecisionPolicy(bf16)`.
3. **Sharded checkpoints:** `torch.distributed.checkpoint.save_state_dict` para evitar all-gather en cada save.
4. **Gradient accumulation:** mantener global batch adecuado sin explotar memoria.
5. **Gradient checkpointing:** para 1B si la memoria pico supera 80 GB.
6. **Auto-resubmit:** mantener el patrón de olas con SIGUSR1 para walltime 6h.

---

## 6. Prototipo viable recomendado

Dado el presupuesto y el hardware, propongo un prototipo escalonado:

### Fase A — Multi-resolution masking a 155M (1-2 días, bajo riesgo)

**Hipótesis:** forzar al modelo a reconstruir unidades de razonamiento (etapas y
bloques de etapas) en lugar de spans arbitrarios puede condensar la señal
estructural y acercarlo a L3.

**Setup:**
- Modelo 155M, 1-2 pro6000.
- `multi_res: true` con niveles `token, span, sent, stage, block`.
- Pesos: token 0.5, span 1.0, sent 1.5, stage 2.0, block 2.5.
- Bloque = 2 etapas consecutivas.
- 10K steps, evaluación L0-L3.

**Costo:** 5-10h de GPU.

**Go/No-Go:**
- L3 ≥ 0.55 → escalar esta ruta.
- L3 0.52-0.54 → combinar con ranking contrastivo (V3.3) y escalar a 350M.
- L3 < 0.52 → la representación no es suficiente; pivotar a híbrido/VDLM o controller.

### Fase B — Denso 350M con FSDP2 + multi-resolution + contrastivo (3-5 días)

**Hipótesis:** con más capacidad y un objetivo que combine MLM multi-resolución
con ranking L3, el modelo puede internalizar relaciones inferenciales.

**Setup:**
- Arquitectura: hidden 768, layers 12, heads 12, ff_mult 4 (~350M params).
- 2-4 pro6000 con FSDP2.
- Corpus: 2-5B tokens (esqueleto + arXiv eco + contrafactuales).
- Objetivo: `alpha * CE_multi_res + (1-alpha) * ranking_L3`.
- 50K-100K steps.

**Costo:** 2-4 días de 2-4 GPUs.

**Go/No-Go:**
- L3 ≥ 0.55 → tenemos una ruta dLLM escalable.
- L3 < 0.52 → el dLLM puro no razona a esta escala; pivotar a híbrido.

### Fase C — VDLM/LaDiR a 350M-1B (1-2 semanas, alto riesgo)

**Hipótesis:** si el problema es la granularidad token-level, operar sobre
variables semánticas latentes puede permitir razonamiento.

**Setup:**
- Entrenar VAE ligero sobre bloques de razonamiento.
- Entrenar diffusion en espacio latente.
- Entrenar renderer Vec2Text.
- 2-4 pro6000, FSDP2, corpus 5-10B tokens.

**Costo:** 1-2 semanas.

**Go/No-Go:**
- L3 ≥ 0.55 → ruta latente prometedora.
- L3 < 0.52 → archivar dLLM-puro.

---

## 7. Riesgos

1. **Más GPU no compensa mala señal.** Si la pérdida no premia razonamiento, un 1B params seguirá fallando en L3.
2. **Escasez de datos.** Para 350M/1B necesitamos 10-100x más tokens de los que tenemos.
3. **Overhead de FSDP2.** En multi-nodo con red lenta, el overhead de comunicación puede comerse el speedup.
4. **Contaminación de contrafactuales.** Si generamos pares con LLM, el modelo puede aprender artefactos.
5. **Cuello de preprocesamiento.** Construir jerarquías, VAE, o variables semánticas requiere mucha ingeniería.

---

## 8. Próximos pasos concretos

1. **Esperar V3.3 y V3.2.** Si alguno cruza 0.55, escalar esa ruta antes de representación estructurada.
2. **Si fallan:** implementar Fase A (multi-resolution masking 155M) en 1-2 días.
3. **Preparar corpus para Fase B:** ampliar a 2-5B tokens con arXiv eco/EcoEvoRxiv y contrafactuales.
4. **Preparar FSDP2:** crear rama con `fully_shard` y probar con 2 pro6000.
5. **Solo si Fase A y Fase B fallan:** considerar VDLM/LaDiR.
