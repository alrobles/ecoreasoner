# EcoReasoner ROADMAP

**Ultima actualizacion:** 25 ago 2026  
**Repo:** github.com/alrobles/ecoreasoner

---

## Vision

Construir un modelo de lenguaje hibrido AR-difusion orientado a agentes cientificos (ecologia, SDM, biologia computacional) que aproveche el hardware MI210 del HPC de KU y lo aprendido con Nemotron/EcoReasoner.

---

## Milestones

### M0 — Infraestructura y Viabilidad (COMPLETADO)
**Fecha objetivo:** 25 ago 2026

- [x] Audit de 21 endpoints HPC, 4 maquinas mesh sincronizadas
- [x] Inventario de 53 GPUs en HPC (5 PRO6000, 14 Q6000, 6 MI210, 6 A100, 12 V100, 8 A40/L40)
- [x] Catalogo de modelos en beegfs (deepseek-v4-flash, gpt-oss:120b, nemotron-3-super)
- [x] Q6000 persistente con GLM-4.7-Flash (:20010)
- [x] Repo principal creado: github.com/alrobles/ecoreasoner
- [x] Analisis de viabilidad: docs/feasibility-hybrid-dllm.md
- [x] Lit review dLLM integrada: alrobles/dLLM (v3, DiDAL round-3 + corpus integration: 34 refs verificadas)
- [x] Protocolo experimental dLLM integrado: docs/dllm-experimental-protocol.md (RQ1-RQ4, gates go/no-go, fases 0-4)
- [x] Mapa literario dLLM: docs/dllm-literature-map.md (paper_graph.json, 40 papers en 3 niveles)

### M1 — Despliegue LLaDA-8B en MI210 (EN PROGRESO)
**Fecha objetivo:** 28 ago 2026

- [ ] LLaDA-8B-Instruct descargado al HPC (COMPLETADO)
- [ ] Script SLURM llada_serve.slurm (COMPLETADO)
- [ ] Lanzar job en MI210 y verificar carga del modelo
- [ ] Tunel SSH :20012 al nodo de compute
- [ ] Smoke test: /v1/models, /v1/chat/completions
- [ ] Medir throughput base (tok/s) en MI210
- [ ] Configurar provider hpc-llada en Hermes config

**Criterio de exito:** LLaDA-8B responde en <10s para 256 tokens en MI210

### M2 — Benchmark AR vs dLLM
**Fecha objetivo:** 01 sep 2026

- [ ] Suite de tests: 20 prompts estratificados (tool-calling, code gen, summarization, CoT)
- [ ] Comparar LLaDA-8B (dLLM) vs Qwen3.5-35B (AR) vs GLM-4.7-Flash (AR)
- [ ] Metricas: latencia, throughput, calidad (humana eval), format_validity
- [ ] BFCL-V4 subset (function calling) — replicar del paper "bitter lesson"
- [ ] Tareas no causales: summarization, draft generation, code scaffolding
- [ ] Tareas causales: JSON schema, tool_call format, multi-step reasoning
- [ ] Seguir el protocolo dLLM: gates A (hardware) y B (entrenamiento) antes de comparar ruido; Fase 1 (110M baseline AR+MDLM-mask) en MI210
- [ ] Documento: docs/benchmark-m1-vs-m2.md

**Criterio de exito:** dLLM >=3x mas rapido en tareas no causales; AR >=90% format validity en tool-calling

### M3 — Bloque A: Evaluacion Baseline EcoReasoner (EN PROGRESO)
**Fecha objetivo:** 28 ago 2026

- [x] 3 replicas LoRA entrenadas a 400 pasos (m1/w1/w2) en MI210
- [x] Script eval_baseline_A.py + eval_baseline_A.slurm
- [ ] Job eval completado en MI210 (actualmente PENDING)
- [ ] Metricas: tool_call_format (gate >=0.70), memory_verbatim, regression_vs_base
- [ ] Documento: docs/eval-baseline-A-results.md

**Criterio de exito:** Al menos 1 replica con tool_call_format >=0.70

### M4 — Bloque B1: Destilacion Multi-Teacher (EN PROGRESO)
**Fecha objetivo:** 05 sep 2026

- [x] Script distill_b1_multiteacher.py (PubMed + GBIF sources)
- [x] System prompt con maxentcpp como motor SDM preferido
- [x] 40 queries estratificadas
- [x] 4287 trazas existentes consolidadas de litdump
- [ ] 10K trazas totales (objetivo: ~5700 nuevas)
- [ ] Code_valid rate >=95%
- [ ] Rotacion del teacher (DeepSeek-V4-Flash) cada 6h
- [ ] Documento: docs/b1-distillation-report.md

**Criterio de exito:** 10K trazas con code_valid >=95%, diversidad de topics >=20 categorias

### M5 — Arquitectura Hibrida: Router + Integrador
**Fecha objetivo:** 15 sep 2026

- [ ] Diseno del Router (heuristic + scoring)
  - Tareas causales -> AR Core (Qwen3.5-35B)
  - Tareas no causales -> dLLM Core (LLaDA-8B)
  - Mixto -> dLLM draft + AR validation
- [ ] Implementar Integrador (schema validation + AR fallback)
- [ ] Pipeline: query -> Router -> (dLLM | AR) -> Integrador -> output
- [ ] Endpoint OpenAI-compatible en :20014
- [ ] Test en Hermes: configurar como provider hpc-hibrido
- [ ] Documento: docs/hybrid-architecture-design.md

**Criterio de exito:** Router dispatch correcto >=90%; Integrador corrige 100% de JSON invalidos

### M6 — Bloque C: Re-entrenamiento con Curriculum
**Fecha objetivo:** 30 sep 2026

- [ ] Curriculum Fase 2: Block A (400 pasos) -> Block B1 (1000 pasos) -> Block B2 (agentic traces)
- [ ] Re-encadenar las 3 replicas (m1/w1/w2) con datos B1 (10K trazas)
- [ ] Eval post-entrenamiento vs baseline (M3)
- [ ] Comparar: AR-only vs AR+dLLM en tareas cientificas
- [ ] Documento: docs/curriculum-c-results.md

**Criterio de exito:** Mejora >=15% en tool_call_format vs M3 baseline

### M7 — Bloque B2: Trazas Agenticas de Sesiones Hermes
**Fecha objetivo:** 15 oct 2026

- [ ] Extraer trazas tool-call de sesiones Hermes reales (session_search)
- [ ] Formatear al esquema context/reasoning/code + tool_calls
- [ ] Muestreo estratificado: SDM, GBIF query, phylogenetic, bioclim
- [ ] 2K trazas agenticas adicionales al dataset B1
- [ ] Documento: docs/b2-agentic-traces.md

**Criterio de exito:** 2K trazas con tool_calls validos, diversidad de herramientas >=10

### M8 — Fine-tuning del dLLM Core
**Fecha objetivo:** 15 nov 2026

- [ ] Adaptar trazas B1 al formato LLaDA (mask + denoise)
- [ ] SFT sobre datos ecologicos (10K trazas)
- [ ] Eval: comparar LLaDA-base vs LLaDA-ecoreasoner en tareas cientificas
- [ ] Si MI210 no es suficiente para training, usar A100 o Blackwell
- [ ] Documento: docs/dllm-finetune-results.md

**Criterio de exito:** LLaDA-ecoreasoner mejora >=10% en CoT ecologico vs LLaDA-base

### M9 — Integracion End-to-End en Hermes
**Fecha objetivo:** 01 dic 2026

- [ ] Router + Integrador como microservicio
- [ ] Provider hpc-hibrido en config.yaml de las 4 maquinas
- [ ] Test en produccion: sesiones reales de EcoSeek
- [ ] Benchmark en produccion: latencia, calidad, coste
- [ ] Documento: docs/production-deployment.md

**Criterio de exito:** Latencia media <3s para drafts; tool_call_format >=95%

### M10 — Levenshtein Editing (Futuro)
**Fecha objetivo:** Q1 2027

- [ ] Adaptar L-EBPO del paper LLaDA 2.2 para datos ecologicos
- [ ] Entrenar edit operations (KEEP/SUB/DEL/INS) sobre trazas agenticas
- [ ] Eval en Hermes real: usar como backend de agentes
- [ ] Documento: docs/levenshtein-editing-results.md

**Criterio de exito:** Self-correction rate >=30% en tool_calls sin fallback AR

---

## Dependencias

```
M0 (done) ──> M1 (in progress) ──> M2 ──> M5 ──> M9
                                        │
M0 (done) ──> M3 (in progress) ──> M4 ──> M6 ──> M8 ──> M9
                                        │          │
                                        M7 ────────┘
```

## Hardware Asignado

| Componente | Hardware | Puerto | Estado |
|---|---|---|---|
| AR Core (Qwen3.5-35B) | 2x MI210 (BF16) | :20012 | Pendiente |
| dLLM Core (LLaDA-8B) | 2x MI210 (BF16) | :20014 | M1 |
| Router + Integrador | Login node / reumanlab | :20016 | M5 |
| Teacher (DeepSeek-V4-Flash) | 2x PRO6000 (Blackwell) | :20006 | Activo |
| Q6000 agent (GLM-4.7-Flash) | 1x Q6000 | :20010 | Activo |
| Local ollama | reumanlab gamma | :11434 | Activo |

## Riesgos Clave

1. **LLaDA-8B no carga en MI210** — Mitigacion: usar device_map="auto" o LLaDA-MoE-7B-A1B
2. **vLLM no soporta LLaDA en gfx90a** — Mitigacion: usar codigo nativo de LLaDA (generate.py)
3. **Teacher muere antes de completar B1** — Mitigacion: chain script con auto-relaunch
4. **Eval A sigue PENDING** — Mitigacion: monitorear squeue, relanzar si necesario
5. **Fine-tuning dLLM en MI210 falla** — Mitigacion: usar A100 o Blackwell para training
