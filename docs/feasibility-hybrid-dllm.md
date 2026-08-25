# EcoReasoner-dLLM: Viabilidad de Arquitectura Hibrida para Agentes Cientificos

**Autor:** A.L. Robles Fernandez  
**Fecha:** 25 ago 2026  
**Estado:** Borrador para discusion  
**Repos:** github.com/alrobles/ecoreasoner (principal), github.com/alrobles/dLLM (lit review)

---

## 1. Motivacion

El EcoReasoner actual entrena LoRA sobre Qwen3.5-35B-A3B (AR) en MI210. Funciona, pero hereda las limitaciones del paradigma autoregresivo: decodificacion secuencial, latencia creciente con longitud, y sin capacidad de revision bidireccional.

Los dLLMs (diffusion language models) prometen generacion paralela, velocidad 5-10x mayor, y capacidad de editar/insertar tokens en cualquier posicion. La pregunta es: podemos construir un modelo hibrido AR-difusion orientado a agentes cientificos que aproveche nuestras MI210 y lo aprendido con Nemotron/EcoReasoner?

## 2. Estado del Arte (ago 2026)

### 2.1 LLaDA 2.2 — Primer dLLM agenticico (Jul 2026)

- **Modelo:** 100B MoE (256 expertos, Top-48 activos), 128K contexto nativo
- **Innovacion:** Levenshtein Editing (KEEP/SUBSTITUTE/DELETE/INSERT)
  - Permite al modelo corregirse durante la generacion sin reiniciar
  - Entrenado en 3 fases: pretraining continuo + SFT + RL (L-EBPO)
- **Resultados:** 703 TPS en BFCL-V4 function calling, 519 TPS en SWE-bench
- **Limitaciones:** SWE-bench Verified y Pro por debajo de modelos AR equivalentes
- **Licencia:** Apache 2.0 (pesos abiertos en HuggingFace)

### 2.2 "Bitter Lesson" (Lu et al., Ene 2026, arXiv:2601.12979)

Evaluacion sistematica de dLLMs en workflows agenticicos. Hallazgos clave:

1. **Embodied agents:** dLLMs fallan en branching bajo feedback temporal
2. **Tool-calling:** dLLMs fallan en precision simbolica (JSON schemas) bajo ruido de difusion
3. **Pero son efectivos en roles no causales:** resumen de memoria, seleccion de herramientas
4. **Recomendacion:** integrar mecanismos de razonamiento causal en el proceso de denoising

### 2.3 Hibridos AR-Difusion (Yuan et al., Feb 2026)

- **Collaborative Thoughts:** AR para razonamiento logico + difusion para generacion paralela
- **Resultado:** 100% accuracy en topologia vs 58% AR-only, 42% difusion-only
- **Tendencia:** la industria se mueve hacia hibridos (difusion para draft + AR para refinamiento)

### 2.4 Aceleracion de inference

- **Fast-dLLM:** KV cache + parallel decoding (training-free, 535 TPS)
- **Efficient-DLM:** step distillation, 10-20 steps con perdida minima
- **DyLLM:** saliency-based token selection, partial attention
- **EntropyCache:** entropy-guided KV cache refresh

## 3. Hardware: MI210 (gfx90a / CDNA2)

### 3.1 Restricciones

| Parametro | Valor | Impacto |
|---|---|---|
| Arquitectura | gfx90a (CDNA2) | No FP8, no MXFP4 |
| VRAM | 64 GB HBM2 | BF16: ~32B params por GPU |
| Compute | 181 TFLOPs FP16 | Suficiente para 8B-35B |
| Bandwidth | 1.6 TB/s | Memory-bound en decode |
| vLLM | rocm/vllm:rocm6.4.1 | Funciona con AITER=0 |
| Quantizacion | INT4 (AWQ/GPTQ) | W4A16 factible |

### 3.2 Lo que ya sabemos (lecciones Nemotron/EcoReasoner)

1. **Qwen3.5-35B-A3B (MoE) funciona en MI210** con LoRA r16 + device_map balanced
2. **FSDP-MoE = SIGSEGV** en ROCm 6.1; ROCm 7.2.2 funciona (SIF qwen35-rocm-v2)
3. **DeepSeek-V4-Flash (MXFP4) NO funciona** en MI210 (solo Blackwell)
4. **GLM-4.7-Flash Q4_K_M (19GB) funciona** en Q6000 via Ollama
5. **vLLM en MI210** requiere enforce-eager, Triton flash-attn, AITER=0
6. **Training swarm** con 3 replicas (m1/w1/w2) completado a 400 pasos

## 4. Arquitectura Propuesta: EcoReasoner-Hibrido

### 4.1 Diseno conceptual

```
                    EcoReasoner-Hibrido
                    ┌─────────────────────────────────────┐
                    │         Router / Dispatcher          │
                    │  (decide AR vs difusion por tarea)   │
                    └──────┬──────────────────┬───────────┘
                           │                  │
                    ┌──────▼──────┐    ┌──────▼──────┐
                    │   AR Core   │    │  dLLM Core  │
                    │  (preciso)  │    │  (rapido)   │
                    │             │    │             │
                    │ Qwen3.5-35B │    │ LLaDA-8B    │
                    │  + EcoReason│    │  + Levenshtn│
                    │   LoRA      │    │   Editing   │
                    └──────┬──────┘    └──────┬──────┘
                           │                  │
                    ┌──────▼──────────────────▼───────┐
                    │      Integrador / Validador       │
                    │  (verifica formato, tool schema,  │
                    │   consistencia causal)            │
                    └───────────────────────────────────┘
```

### 4.2 Componentes

#### A) AR Core (Razonamiento Causal)
- **Base:** Qwen3.5-35B-A3B (ya entrenado) o GLM-4.7-Flash (19GB Q4)
- **Funcion:** tool-calling preciso, JSON schema, razonamiento paso-a-paso
- **Rol:** Generar plan, validar output del dLLM, ejecutar tool calls
- **Hardware:** 2x MI210 (BF16) o 1x Q6000 (Q4_K_M via Ollama)

#### B) dLLM Core (Generacion Rapida)
- **Base:** LLaDA-8B-Instruct (MIT license, 8B, cabe en 1x MI210 en BF16)
- **Funcion:** draft generation, resumen de memoria, seleccion de herramientas, generacion de codigo
- **Rol:** Todo lo no-causal: context summarization, draft CoT, code scaffolding
- **Mejora futura:** LLaDA-MoE-7B-A1B (1B activo, ~10x mas rapido)
- **Hardware:** 1x MI210 (BF16, vLLM con enforce-eager)

#### C) Router/Dispatcher
- Heuristica simple: si la tarea requiere JSON preciso o tool_call -> AR Core
- Si la tarea es generacion de texto/codigo/resumen -> dLLM Core
- Si es mixto: dLLM genera draft, AR valida y corrige

#### D) Integrador/Validador
- Verifica que el output del dLLM cumpla schema
- Si falla, pasa al AR Core para correccion
- Metricas: format_validity_rate, correction_rate, latency_total

### 4.3 Por que NO montar sobre Nemotron

1. **Nemotron-3-Super es AR puro** — no tiene infraestructura de difusion
2. **El EcoReasoner actual (Qwen3.5 LoRA) ya es AR** — reusarlo como AR Core
3. **LLaDA-8B es open-source MIT** — podemos afinarlo con datos del EcoReasoner
4. **El valor esta en el hibrido**, no en otro modelo AR afinado

### 4.4 Por que LLaDA-8B y no LLaDA-2.2 (100B)

1. **100B no cabe en MI210** (necesita ~200GB BF16 = 4+ GPUs)
2. **LLaDA-8B-Instruct** cabe en 1x MI210 (16GB BF16 + KV cache)
3. **LLaDA-MoE-7B-A1B** es mejor alternativa (1B activo, mas rapido)
4. **Escala:** podemos empezar con 8B y escalar a MoE-7B cuando este disponible

## 5. Plan de Implementacion

### Fase 0: Viabilidad (2 semanas)

1. **Desplegar LLaDA-8B-Instruct en MI210** via vLLM
   - Imagen: rocm/vllm:rocm6.4.1_vllm_0.10.1
   - Config: --enforce-eager, VLLM_USE_AITER=0, Triton flash-attn
   - Verificar: throughput, calidad, context window

2. **Benchmark agenticico**
   - BFCL-V4 subset (tool-calling)
   - Agentboard subset (embodied)
   - Comparar vs Qwen3.5-35B AR Core

3. **Medir ganancia de velocidad**
   - Latencia AR vs dLLM en tareas no causales
   - Calidad de draft generation

### Fase 1: Arquitectura Hibrida (4 semanas)

1. **Implementar Router** — heuristica + scoring
2. **Implementar Integrador** — schema validation + fallback AR
3. **Pipeline:** user query -> Router -> (dLLM draft | AR direct) -> Integrador -> output
4. **Servir como endpoint OpenAI-compatible** para Hermes

### Fase 2: Fine-tuning del dLLM Core (6 semanas)

1. **Destilar trazas B1 del EcoReasoner** en formato LLaDA (mask + denoise)
2. **SFT sobre datos ecologicos** (sci_v2_b1.jsonl, 10K trazas)
3. **Eval:** comparar hibrido vs AR-only en tareas cientificas

### Fase 3: Levenshtein Editing (futuro)

1. **Adaptar L-EBPO** del paper LLaDA 2.2 para datos ecologicos
2. **Entrenar edit operations** (KEEP/SUB/DEL/INS) sobre trazas agenticas
3. **Eval en Hermes real** — usar como backend de agentes

## 6. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigacion |
|---|---|---|---|
| LLaDA-8B no corre bien en MI210 | Media | Alto | Probar LLaDA-MoE-7B-A1B (mas ligero) |
| dLLM falla en tool-calling | Alto (documentado) | Medio | AR Core como fallback obligatorio |
| Latencia hibrida > AR solo | Media | Alto | dLLM solo para drafts, no para output final |
| Fine-tuning dLLM en MI210 falla | Media | Medio | Usar A100 o Blackwell para training, MI210 para inference |
| Context window limitado (32K) | Bajo | Bajo | LLaDA soporta 128K nativo |

## 7. Diferenciacion vs LLaDA 2.2

| Aspecto | LLaDA 2.2 | EcoReasoner-Hibrido |
|---|---|---|
| Paradigma | dLLM puro | Hibrido AR + dLLM |
| Escala | 100B MoE | 8B-35B (MI210) |
| Agentes | Levenshtein editing | Router + validador AR |
| Tool-calling | Difusion nativa (ruidosa) | AR Core preciso |
| Dominio | General | Cientifico (ecologia) |
| Hardware | H100/B200 | MI210 (ROCm) |

## 8. Conclusion

**La arquitectura hibrida es viable y ventajosa** para nuestro caso de uso:

1. El "bitter lesson" confirma que dLLMs solos no sirven para agentes — pero en roles no causales si
2. LLaDA-8B cabe en MI210 y es open-source MIT
3. Nuestro EcoReasoner (AR) ya funciona como Core causal
4. El Router + Integrador resuelve el problema de precision simbolica
5. El dominio cientifico se beneficia mas de drafts rapidos que de JSON perfecto

**Proximo paso:** Desplegar LLaDA-8B-Instruct en MI210 y medir throughput vs Qwen3.5-35B.
