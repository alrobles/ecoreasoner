# Estrategia LLaDA-8B (dLLM Core de EcoReasoner)

**Fecha:** 25 ago 2026 · **Repo:** github.com/alrobles/ecoreasoner · **Estado:** aprobada, Fase A pendiente de lanzamiento

---

## Resumen ejecutivo

LLaDA-8B **no se entrena desde cero** en nuestro hardware (ver §4: ~24 años en 2 MI210, ~4 meses solo con acceso exclusivo a todo el cluster). La estrategia correcta es **servir el modelo base preentrenado + fine-tuning (SFT/LoRA) sobre datos ecológicos**, conectando con M1 (deploy), M2 (benchmark) y M8 (fine-tune) del ROADMAP.

---

## Fase A — Poner LLaDA-8B a servir (M1)

**Hardware:** 1 MI210 (64GB) — el modelo BF16 ocupa ~16GB, cabe con holgura.

- Script: `llada_serve_1gpu.slurm` (ya existe, `gres:gpu:mi210:1`)
- Modelo: ya descargado en `/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/` (6 shards safetensors)
- Serve: apptainer + transformers (sin vLLM), OpenAI-compatible en :8080
- Patrón producción igual a v4serve: `node:port` → `.current_llada_node` → túnel SSH :20014 → provider `hpc-llada` en Hermes
- Smoke: `/v1/models`, `/v1/chat/completions`, medir throughput base (tok/s)

**Criterio de éxito (ROADMAP):** LLaDA-8B responde en <10s para 256 tokens en MI210.

## Fase B — Baseline de referencia (M2)

Comparar LLaDA-8B (dLLM) vs Qwen3.5-35B (AR) vs GLM-4.7-Flash (AR) en las 20 tareas estratificadas del protocolo, sobre el mismo hardware MI210:

- Tareas no causales: summarization, draft generation, code scaffolding
- Tareas causales: JSON schema, tool_call format, multi-step reasoning
- Criterios: dLLM ≥3× más rápido en no-causales; AR ≥90% format validity en tool-calling

## Fase C — Fine-tuning del dLLM (M8)

- El fine-tune de LLaDA-8B (SFT/LoRA) **sí es barato**: 10-30K ejemplos ecológicos ≈ horas en 1-2 MI210.
- Adaptar trazas B1/B2 (contexto/reasoning/code + tool_calls) al formato LLaDA (mask + denoise).
- Eval: LLaDA-ecoreasoner vs LLaDA-base en tareas científicas.
- **Criterio de éxito:** mejora ≥10% en CoT ecológico vs LLaDA-base.

**Por qué funciona:** no declaramos viabilidad por velocidad de entrenamiento desde cero (inviable), sino por **servicio + fine-tune de un dLLM de 8B ya preentrenado**, que es el rol del dLLM Core (fast draft, summarization, no-causal): tareas donde el dLLM no pierde calidad y gana throughput.

---

## §4. ¿Y si quisiéramos entrenar LLaDA-8B DESDE CERO con nuestro hardware?

**Coste de referencia:** LLaDA-8B (8B params, 2.3T tokens) = 6·P·T = **1.10e23 FLOP**.

### Hardware real disponible (inventario scontrol, 25-ago-2026)

| Tarjeta | GPUs usables | BF16 TF | Nota |
|---|---|---|---|
| MI210 | 54 (27 nodos × 2, tope scheduler) | 181 | nuestra columna |
| A100 | 20 | 312 | compartida |
| Q6000 | 29 | 149 | mesh |
| V100 | 36 | 112 | BF16 no nativo (FP16) |
| PRO6000 | 5 | 238 | teacher |
| A40/L40/Q8000 | 10 | ~150 | minoritaria |
| **Total cluster** | **~154** | **~2.7e16 FLOP/s** | teórico, nadie lo tiene exclusivo |

### Escenarios de tiempo para LLaDA-8B completo (2.3T tokens, MFU 40%)

| Escenario | FLOP/s útiles | Tiempo |
|---|---|---|
| **2 MI210 (nuestro caso real)** | 1.45e14 | **~24 años** ❌ |
| 54 MI210 (acceso exclusivo a todas las MI210) | 3.9e15 | ~11 meses ⚠️ |
| Todo el cluster (todas las tarjetas, teórico) | 1.08e16 | ~4 meses ⚠️ |

### Variantes reducidas (si se quisiera insistir)

| Objetivo | Coste | 2 MI210 | 54 MI210 | Todo cluster |
|---|---|---|---|---|
| LLaDA-8B-lite (8B, 300B tok) | 1.44e22 | ~3 años ❌ | ~43 días ⚠️ | ~15 días |
| LLaDA-8B mínimo útil (8B, 100B tok) | 4.8e21 | ~1 año ❌ | ~14 días | ~4.5 días |
| LLaDA-MoE-7B-A1B (1B activo, 2.3T tok) | ~1.4e22 | ~3 años ❌ | ~6 semanas | ~2 semanas |

### Barreras adicionales (además de los FLOPs)

1. **Walltime 6h:** entrenar durante meses requiere checkpoint/resume perfecto cada 6h durante meses — riesgo operacional enorme (drift, corrupción de checkpoint, colas).
2. **Cluster compartido:** nadie tiene 54 MI210 exclusivas; los `alphaGGAU.sh` de otros usuarios ya demostraron que las MI210 se reparten.
3. **ROCm/MI210 no es la plataforma objetivo** de LLaDA (paper usa NVIDIA); MFU realista probablemente < 40%, empeorando todo.

### Conclusión

- **2 MI210: entrenar 8B desde cero es imposible en plazo útil** (~24 años el completo, ~1 año incluso el mínimo útil de 100B tokens).
- Solo sería "posible" con acceso **exclusivo a TODO el cluster durante ~4 meses**, que no es nuestro caso.
- **El techo realista de un dLLM entrenado desde cero con nuestro setup sigue siendo ~700M-1B / 20-30B tokens** (Fase 2 del `experimental_protocol.md`, ~3-5 días en 2 MI210).
- Por eso la estrategia correcta es **Fases A/B/C (servir + benchmark + fine-tune)**, no entrenamiento desde cero.