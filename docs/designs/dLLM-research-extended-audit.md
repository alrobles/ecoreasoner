# Auditoría extendida de dLLMs para razonamiento científico

> Fecha: 2026-09-10.
> Contexto: respuesta al pedido de investigación más profunda (papers, repos,
> código, datos) con foco en un mínimo prototipo viable para EcoReasoner.

---

## 1. Resumen ejecutivo

La investigación confirma el diagnóstico local: **la mayoría de los dLLMs de
pre-entrenamiento no implementan una señal de razonamiento explícita.** Los
avances recientes (2024-2026) se concentran en:

1. **Post-entrenamiento RL/contrastivo:** d1, d2, AGRPO, LLaDOU, dLLM-RL,
   ReNCE, PADRE.
2. **Control de orden de desenmascaramiento:** LogicDiff, dLLM-Reason,
   LLaDOU-UPM.
3. **Representación estructurada/jerárquica:** HDLM (clusters), multi-resolution
   masking, LaDiR/VDLM (latente).
4. **Datos contrafactuales/densos:** DISCO, CORE, PNS, SCoTD, RECODE.

**Hallazgo central:** prácticamente **ningún repositorio oficial auditado
implementa una pérdida contrastiva explícita** tipo ranking sobre pares
correcto/incorrecto. EcoReasoner ya tiene una implementación propia en
`scripts/train_mdlm_moe_v3_contrastive.py` (ranking L3). La dirección más
productiva, si V3.3 da señal, es refinar ese objetivo con inspiración de RL para
dLLMs y posiblemente un control de orden de desenmascaramiento.

**Hallazgo de infraestructura:** existen frameworks recientes y bien
estructurados que **sí se pueden auditar, clonar y adaptar** para un MVP:
- `dllm-reasoning/d1` — masked SFT trainer y diffu-GRPO.
- `BDeMo/dLLM_Reason` — DAG-guided unmasking, schedulers, RL, process reward.
- `zhouc20/HDLM` — clustering semántico y jerarquía de tokens.
- `kuleshov-group/mdlm` — MDLM/SUBS con Hydra/Lightning.

**Hallazgo de datos:** el corpus local (v5, ~1.1B tokens) ya es in-domain. La
mejora más barata no es más texto, sino **más estructura, contrafactuales y
metadatos** (MeSH, citas, autores, esqueletos argumentativos). Herramientas como
`arxiv-corpus-builder`, DISCO/CORE, GROBID/SAM/SciCite son reutilizables.

---

## 2. Estado de V3.3 (actualizado al momento del informe)

- Job: 29184152, relanzado por error `KeyError seed` del job 29183112.
- Tiempo: ~8 min, step ~1220/3000.
- `rank_acc` del trainer: 0.875-1.0 (evaluación interna de pares del mismo
  generador; puede ser artefacto).
- Loss rank baja (~0.0001), lo que indica que el modelo aprende a separar las
  parejas del entrenamiento.
- **Veredicto real aún pendiente:** el ranking en training no es evidencia de
  L3. Se requiere la batería hold-out post-entrenamiento.

---

## 3. Papers y repositorios auditados

### 3.1 dLLMs base / pre-entrenamiento

| Paper | arXiv | Repo | Estado del código | Notas para EcoReasoner |
|---|---|---|---|---|
| MDLM | 2406.07524 | kuleshov-group/mdlm | Completo, Hydra+Lightning | Base limpia de SUBS/MLM; samplers, DiT/AR/Mamba. |
| LLaDA | 2502.09992 | ML-GSAI/LLaDA | Oficial, inferencia/SFT | 8B desde cero; útil para comparar, no para reentrenar fácilmente. |
| HDLM (Zhou) | 2510.08632 | zhouc20/HDLM | Completo, training+eval | Multi-resolution por clusters semánticos; requiere pre-entrenado para clustering. |
| HDLM (ServiceNow) | 2504.06416 | ServiceNow/hdlm | Completo, hybrid | Hyperschedules AR/difusión, sampler ACS; más de generación que de reasoning. |
| SEDD | 2310.16834 | louaaron/Score-Entropy-Discrete-Diffusion | Completo | Discrete score-entropy; baseline alternativa. |

### 3.2 Razonamiento / post-entrenamiento

| Paper | arXiv | Repo | Estado | Notas |
|---|---|---|---|---|
| d1 | 2504.12216 | dllm-reasoning/d1 | Completo | Masked SFT + diffu-GRPO. `dLLMTrainer` y `dLLMDataCollator` reutilizables. |
| d2 | 2509.21474 | kuleshov-group/d2 | Completo | Estimación exacta de likelihood any-order para RL. |
| AGRPO | 2510.04019 | probablyabot/agrpo | Completo | RL por paso de denoising, evita sesgo de approx. |
| LLaDOU | 2505.10446 | maple-research-lab/LLaDOU | Completo | Unmasking Policy Module con Plackett–Luce; ranking de orden. |
| dLLM-RL (TraceRL) | 2509.06949 | Gen-Verse/dLLM-RL | Completo | Framework integral con process reward model y value model. |
| LogicDiff | 2603.26771 | No encontrado | Paper-only | Inference-time: cabeza ligera ~4.2M + scheduler por roles lógicos. |
| ReNCE | 2601.22432 | WenzhengZhang/ReNCE | Placeholder | Contrastive NCE online; código no liberado. |
| PADRE | 2605.05226 | No encontrado | Paper-only | Pseudo-likelihood para RL de dLLMs. |

### 3.3 Representación estructurada / latente

| Paper | arXiv | Repo | Estado | Notas |
|---|---|---|---|---|
| LaDiR | 2510.04573 | mk322/LaDiR | Inmaduro | VAE + latent diffusion sobre thought tokens. Riesgo de colapso a pequeña escala. |
| VDLM Variable | 2602.15870 | No encontrado | Paper-only | Planner semántico + Vec2Text renderer. Sin código. |
| Diffusion-of-Thoughts | 2402.07754 | HKUNLP/diffusion-of-thoughts | Antiguo, basado en Plaid/DiffuSeq | Conceptualmente útil, no reusable directamente. |

### 3.4 Datos / contrafactuales

| Paper/Dataset | arXiv | Repo | Uso para EcoReasoner |
|---|---|---|---|
| DISCO | 2212.10534 | epfl-nlp/disco | Generar contrafactuales con LLM + filtro teacher. |
| CORE | 2210.04873 | tanay2001/CORE | Retrieve-then-edit: entrenar retriever de ejemplos opuestos. |
| PNS | 2602.03516 | No encontrado | Plausible negative samples para DPO. |
| SCoTD | 2306.14050 | liunian-harold-li/scotd | Destilación de CoT a modelos 125M-1.3B. |
| RECODE | Zenodo | No encontrado | Corpus ecológico anotado para NER/RE. |

---

## 4. Auditoría de códigos clonados

### 4.1 d1 (`dllm-reasoning/d1`)

**Estructura**
- `SFT/sft_trainer.py`: `dLLMTrainer` (subclase de HuggingFace `Trainer`),
  `dLLMDataCollator`, `dLLMSFTDataset`, `preprocess_dataset`.
- `diffu-grpo/`: scripts y entrenador RL.
- `eval/`: evaluación y parseo.
- `env.yml`: Python 3.10, PyTorch 2.6.0, Transformers 4.49.0, TRL, DeepSpeed.

**Snippet clave (masked SFT loss, `SFT/sft_trainer.py:12-26`)**
```python
class dLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, ...):
        labels, t, num_prompt_tokens = inputs.pop(...)
        outputs = model(**inputs)
        logits = outputs.logits
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        loss = unscaled_loss / t
        loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)
        return loss
```

**Snippet clave (forward noising, `SFT/sft_trainer.py:71-84`)**
```python
def forward_process(self, batch, eps=1e-3):
    input_ids = batch["input_ids"]
    B, N = input_ids.shape
    t = torch.rand((B,), device=input_ids.device)
    t = (1 - eps) * t + eps
    t = t[:, None].repeat(1, N)
    mask_indices = torch.rand((B, N), device=input_ids.device) < t
    noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
    return noisy_batch, t, mask_indices
```

**Juicio:**
- **Muy reutilizable** para masked SFT/completion. La idea de entrenar sobre
  razonamientos con formato `<reasoning>...</reasoning><answer>...</answer>` es
  directamente aplicable a esqueletos ecológicos.
- `preprocess_dataset` está atado a s1K; hay que adaptar.
- diffu-GRPO requiere rewards verificables (outcome). Para ecología no hay
  verificador automático todavía.

### 4.2 dLLM-Reason (`BDeMo/dLLM_Reason`)

**Estructura**
- `src/dllm_reason/scheduler/`: 13 schedulers (confidence, entropy, linear,
  maskgit, curriculum, adaptive, DAG, ...).
- `src/dllm_reason/graph/`: TokenDAG, SpanDAG, templates (COT, skeleton, etc.).
- `src/dllm_reason/inference/`: samplers (DAGSampler, DiffusionSampler).
- `src/dllm_reason/search/`: búsqueda/evolución de DAGs.
- `src/dllm_reason/training/`: pretrain, finetune, DAG-aware, RL, correction.
- `src/dllm_reason/library/`: EpisodeStore, retrieval, feedback.
- `train.py`, `evaluate.py`, configs YAML, scripts Slurm.

**Snippet clave (DAG scheduler, `src/dllm_reason/scheduler/dag_scheduler.py:50-99`)**
```python
def select_positions(self, step, total_steps, current_mask, is_unmasked,
                     logits, confidences, ..., n_to_select=1):
    ...
    ready = self.dag.ready_positions(is_unmasked)
    eligible = ready & current_mask
    if self.sub_strategy == "all_ready":
        return eligible
    elif self.sub_strategy == "confidence_topk":
        return self._confidence_topk(eligible, confidences, n_to_select)
    ...
```

**Snippet clave (templates de DAG, `src/dllm_reason/graph/templates.py:34-70`)**
```python
def chain_of_thought_dag(seq_len, num_steps, prompt_len=0, device="cpu"):
    gen_len = seq_len - prompt_len
    positions_per_step = gen_len // num_steps
    ...
    return TokenDAG.from_levels(levels, seq_len=seq_len, device=device)
```

**Juicio:**
- **Muy reutilizable** para controlar orden de generación y probar hipótesis de
  razonamiento estructurado sin reentrenar.
- Permite definir DAGs de etapas OBS/HIP/PRED/EVID/CONC y forzar al modelo a
  generar primero hipótesis/evidencia y luego conclusiones.
- Incluye RL para entrenar políticas de desenmascaramiento (LLaDOU-style).
- Es el framework más maduro para representación estructurada + RL.

### 4.3 HDLM (`zhouc20/HDLM`)

**Estructura**
- `hdlm/train.py`: DDP, Hydra, DIT, wandb.
- `hdlm/diffusion_process.py`: forward/reverse con clusters.
- `hdlm/loss.py`: GiddLoss / HDLMLoss con ELBO cerrado.
- `hdlm/compute_clusters.py`: clustering semántico K-means++ con restricciones de
  tamaño sobre embeddings de tokens.
- `hdlm/clusters/`: clusters precalculados para GPT-2 / GIDD.
- `hdlm/models/dit.py`: DiT con soporte de cluster tokens.

**Snippet clave (clustering, `hdlm/compute_clusters.py:11-47`)**
```python
def semantic_kmeans_cluster(embeddings, cluster_size, max_iters=100, ...,
                            min_size_ratio=0.1, max_size_ratio=5.0,
                            use_cosine=True, seed=42):
    # embeddings: [vocab_size, embed_dim]
    # cluster_size: number of clusters n
    ...
    # K-means++ initialization + soft size constraints
```

**Juicio:**
- **Reutilizable** para añadir niveles jerárquicos a MDLM.
- Requiere un modelo pre-entrenado del que sacar embeddings de tokens.
- Puede integrarse con el dLLM 155M entrenado (usar sus embeddings para
  clusterizar).
- Es la opción más directa de representación estructurada jerárquica.

### 4.4 MDLM (`kuleshov-group/mdlm`)

**Estructura**
- `main.py`: entrenamiento/evaluación con Hydra.
- `diffusion.py`: `Diffusion` LightningModule, SUBS/D3PM/SEDD/AR.
- `noise_schedule.py`: schedules.
- `models/`: DiT, AR, Mamba.
- `configs/`: Hydra configs.

**Juicio:**
- **Base limpia** para entrenar dLLM desde cero con receta moderna.
- No es específicamente para reasoning, pero la arquitectura y loss son
  referencia sólida.

---

## 5. Hallazgos sobre datos y corpus

El subagente de datos concluyó:

- El corpus v5 local ya tiene ~1.1B tokens en dominio ecológico/evolutivo
  (`docs/ecoreasoner-MINING-REVIEW-V5.md`).
- **La mejora más barata es estructura, no más texto:** añadir MeSH, journal,
  autores, citas, grafo de co-citación, y zonas argumentativas.
- Fuentes de texto adicionales in-domain:
  - arXiv q-bio.PE / q-bio.OT (teóricos, métodos).
  - bioRxiv `ecology` / `evolutionary-biology`.
  - PMC Open Access filtrado por MeSH.
  - OpenAlex `topics.subfield.id:1105` (~$0.01/PDF).
  - BHL OCR (60M+ páginas, ruidoso, para taxonomía).
- Generación de contrafactuales:
  - **DISCO** (más rápido): LLM + teacher filter.
  - **CORE** (más robusto): retrieve-then-edit con DPR entrenado.
  - `build_pairs_hard_v3.py` local ya cubre mutaciones regladas; DISCO/CORE
    pueden enriquecerlas.

**Recomendación de datos para MVP:**
1. No ampliar corpus hasta tener EcoBench-EVAL estable.
2. Enriquecer v5 con metadata estructural (MeSH, citas) mediante joins baratos.
3. Generar `EcoContra` (pares contrafactuales) con DISCO o reglas + LLM local.
4. Construir esqueletos argumentativos (GROBID + SAM/SciCite) para entrenar
   razonamiento estructurado.

---

## 6. Mapa de opciones para MVP

### Eje 1: objetivo de entrenamiento

| Opción | Costo | Riesgo | Recompensa si funciona |
|---|---|---|---|
| Refinar ranking L3 (V3.3 actual) | Bajo | Atajos del generador | Prueba directa de la hipótesis actual |
| DPO sobre pares L3 (DPO-ST / d1) | Medio | Inestabilidad con dLLM | Mejor alineación de preferencias |
| diffu-GRPO / AGRPO / TraceRL | Medio-Alto | Recompensas verificables escasas | RL de razonamiento real |
| NCE online (ReNCE) | Medio | Código no liberado | Contrastivo sin DPO |

### Eje 2: representación / orden de generación

| Opción | Costo | Riesgo | Recompensa |
|---|---|---|---|
| Multi-resolution masking local | Muy bajo | Puede seguir sin señal | Forzar estructura sin cambiar modelo |
| DAG-guided unmasking (dLLM-Reason) | Bajo-Medio | Necesita DAGs de etapas | Controla orden sin reentrenar |
| HDLM clusters (Zhou) | Medio | Requiere pre-entrenado y clustering | Niveles semánticos en vocabulario |
| LogicDiff inference-time | Bajo (sólo cabeza) | No hay repo | Mejora razonamiento en generación |
| LaDiR/VDLM | Alto | VAE frágil a 155M | Representación latente real |

### Eje 3: datos

| Opción | Costo | Riesgo | Recompensa |
|---|---|---|---|
| Enriquecer v5 con metadata | Muy bajo | Ninguno | Más señal sin más FLOPs |
| EcoContra (DISCO/CORE) | Medio | Calidad variable | Pares densos para ranking/DPO |
| Esqueletos argumentativos | Medio | Ruido de extracción | Estructura explícita para entrenar |

---

## 7. Recomendaciones de MVP concretas

### Escenario A: V3.3 da GO (L3 ≥ 0.55)

**MVP:** reentrenar desde cero (o desde checkpoint) con **ranking L3 + DAG-guided
unmasking**.

1. **Datos:**
   - Enriquecer v5 con metadata estructural.
   - Generar `EcoContra` con DISCO/reglas.
2. **Modelo:**
   - 350M denso (o 155M si el presupuesto es corto).
   - FSDP2 en 2-4 pro6000.
3. **Objetivo:**
   - Ranking L3 refinado con modelo de referencia (DPO-style).
   - MLM auxiliar para no olvidar vocabulario.
4. **Inferencia:**
   - DAG de etapas OBS→HIP→PRED→EVID→CONC usando `dLLM_Reason` o un scheduler
     propio.
5. **Evaluación:**
   - EcoBench L0-L3 con pares hold-out.

**Costo estimado:** 3-5 días en 4 pro6000, ~10B tokens.

### Escenario B: V3.3 es direccional (0.52–0.54)

**MVP:** confirmar si el problema es orden de generación más que señal de
entrenamiento.

1. Implementar **LogicDiff-style** scheduler o DAG de etapas sobre el modelo
   entrenado.
2. Evaluar si controlar el orden mejora L3.
3. Si no, archivar dLLM-puro.

**Costo estimado:** 1-2 días de ingeniería, sin reentrenar.

### Escenario C: V3.3 es NO-GO (L3 < 0.52)

**MVP:** pivotar a **controller/verificator** o a **híbrido dLLM+verificador**.

- El verificador estructural ya entrega 98.2-98.6% match_args.
- El dLLM puede quedar como componente generador de un sistema híbrido.
- Publicar el negativo: 7/7 runs con dLLM puro fallan L3, pero el híbrido
  funciona.

**Costo estimado:** 1 día para documentar y pivotar.

---

## 8. Plan de acción inmediato

1. **Esperar V3.3** y aplicar veredicto frío sobre batería hold-out.
2. Si GO o direccional, clonar y probar `dLLM_Reason` scheduler DAG con el
   checkpoint V3.3 (sin reentrenar).
3. Preparar datos `EcoContra` (DISCO/reglas) para entrenamiento ampliado.
4. Preparar FSDP2 wrapper y config para 350M.
5. Si NO-GO, cerrar dLLM-puro y documentar pivot.

---

## 9. Advertencias y riesgos

- **Dependencias frágiles:** `d1`, `dLLM_Reason`, `HDLM` usan versiones
  específicas de PyTorch/Transformers/FlashAttention. El clúster usa PyTorch
  2.7.1+cu128 con pro6000 (Blackwell); puede ser necesario ajustar o recompilar
  kernels.
- **Código inmaduro:** LaDiR, VDLM, LogicDiff no tienen repos oficiales.
- **No hay contrastivo en repos oficiales:** EcoReasoner debe mantener su propio
  `ranking_loss`.
- **Más GPUs no curan señal:** antes de escalar, debe haber evidencia L3 ≥ 0.55.
- **Presupuesto de datos:** 350M requiere ~7B tokens, 1B requiere ~20B tokens.
  El corpus v5 es insuficiente para 1B.
