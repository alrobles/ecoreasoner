# Plan de estudio EcoReasoner — cerrar el gap del vibe-coding

Objetivo: que cada pieza del repo sea algo que puedas **explicar en 90
segundos, reimplementar desde cero, y defender bajo preguntas** — el estándar
de una entrevista research-engineer (Inception/labs dLLM).

Método por pieza (aplicar en orden estricto):

1. **Explicar**: lee el código y dilo en voz alta en inglés, línea a línea,
   sin mirar notas.
2. **Predecir**: antes de correr cualquier cosa, escribe qué esperas que salga.
3. **Reimplementar**: versión mínima desde cero, sin mirar el original.
4. **Romper**: "¿qué pasa si…?" — cada bug real del repo es un ejercicio.
5. **Defender**: pitch de 60-90s + respuesta a "¿por qué no X?"

---

## Etapa 0 — Mapa del proyecto (tu historia de 90 segundos)

Las etapas reales del repo (git log):

| etapa | qué pasó | commits/hitos |
|---|---|---|
| F0 harness | battery de discriminación + eval pairwise | `logicdiff_*`, `eval_baseline_*` |
| corpus | arxiv/PMC/papers_db → esqueletos `[OBSERVACION]..[CONCLUSION]` | `build_*`, `pre_tokenize*` |
| trainer v1→v2 | MDLM + MoE + masking machinery | `train_mdlm_moe*.py` |
| GA G8-G11 | falsificación controlada (hinge contrastivo falsado en G10) | `63fcc21`, `618bb7e` |
| pivot chat | SFT pipeline + prototipo 676M | `eb662b2` |
| v5-1b | escala 1.13B, DDP heterogéneo, ZeRO-1, memmap, watchdog | `8150810`..`e735ef3` |

Ejercicio 0: narra esta historia en 90s en inglés — "I built a masked-diffusion
MoE pipeline from scratch: corpus, trainer, eval battery, distributed training
on heterogeneous GPUs."

---

## Etapa 1 — Fundamentos PyTorch (la brecha principal)

Todo anclado en `scripts/train_mdlm_moe_v2.py`.

### 1.1 Tensores, views, gather/scatter
Código real: `MoEMLP.forward` (líneas 240-263) — `reshape`, `topk`,
`scatter_add_`, indexing booleano `out[sel] += w[sel,None]*experts[e](flat[sel])`.

Ejercicios:
- E1.1 Reimplementa `MoEMLP` desde cero (top-1, 8 expertos, dim 64) y comprueba
  que tu forward coincide con el del repo en tensores aleatorios.
- E1.2 Reescribe el loop `for e in range(n)` con `torch.bincount`/`scatter`
  puro (sin loop python) — mide diferencia de tiempo en CPU y GPU.
- E1.3 Predice sin correr: ¿`flat[sel]` copia o vista? ¿`out[sel] +=` es
  in-place sobre qué storage? ¿`reshape` vs `view` cuándo falla?

### 1.2 autograd / backward
Código real: `(loss/ARGS.grad_accum + aux).backward()` (~línea 1660) y
`balance_loss` (264-288) — nota cómo `_gate_probs` se guarda *diferenciable* y
`_fcount` *stop-grad*, a propósito.

Ejercicios:
- E2.1 Traza a mano el grafo: ¿por qué `f` (fracción de tokens) va detach y
  `P` (prob media del gate) no? ¿Qué pasaría si ambos fueran detach?
- E2.2 Reimplementa `balance_loss` y verifica que `gate.weight.grad` recibe
  gradiente aunque un experto no se use en el batch (el truco `probe_x`).
- E2.3 Explica `no_sync()` (línea 1658): ¿qué comunicación evita y cuándo es
  seguro usarlo con grad accumulation?

### 1.3 nn.Module, buffers, state_dict
Código real: `register_buffer("_fcount", ..., persistent=False)` (línea 236),
`ema_model` dicts, `_save_checkpoint` (1228+).

Ejercicios:
- E3.1 ¿Por qué `_fcount` es `persistent=False`? ¿Qué pasaría al cargar un
  checkpoint viejo si fuera persistent?
- E3.2 Reimplementa `_init_ema`/`_update_ema` (1498-1521) — versión CPU y
  versión GPU. Mide la diferencia de tiempo en un modelo de ~50M params.
- E3.3 Escribe `save_ckpt` atómico: tmp+`os.replace` — explica por qué el
  rename es atómico y por qué el PID en el nombre (comentario línea 1247).

### 1.4 DDP y ZeRO
Código real: `init_process_group("nccl")` (~1410), wrap DDP (~1524),
`ZeroRedundancyOptimizer` (~1408-1421), `_consolidate_zero1` (1348).

Ejercicios:
- E4.1 Mini-DDP a mano: 2 procesos, cada uno con un modelo idéntico;
  sincroniza gradientes con `dist.all_reduce` — compara vs
  `DistributedDataParallel`.
- E4.2 Memoria: calcula a mano bytes para 1.13B params en fp32 con AdamW
  (w + g + m + v = 16B/param). Verifica: 1.13B × 16 ≈ 18GB — por eso q6000
  (21.97GB útiles) requirió ZeRO. Con ZeRO-1 en 3 ranks: ~12GB/rank.
- E4.3 Explica por qué `consolidate_state_dict` es colectivo y qué hace que
  el checkpoint ZeRO sea compatible con AdamW plano (cambio de isla).
- E4.4 ¿Qué es `find_unused_parameters` y por qué el `probe_aux` de
  `balance_loss` lo hace innecesario?

### 1.5 Memoria y performance
Código real: `--grad_ckpt` (374-377), fused AdamW, `np.load(mmap_mode='r')`
(~438-456), el bug EMA-CPU.

Ejercicios:
- E5.1 Reproduce el bug EMA-CPU: modelo ~200M params, `_update_ema` CPU vs GPU
  — mide y explica el 28s→1s (DtoH por-tensor + ops CPU).
- E5.2 `torch.utils.checkpoint.checkpoint(b, h, use_reentrant=False)` — mide
  VRAM y tiempo por step con/sin. Explica el tradeoff recompute-vs-store.
- E5.3 `np.load(mmap_mode='r')` + `torch.from_numpy` — verifica que N ranks en
  el mismo nodo comparten page cache (mide RSS antes/después de 2 procesos).
- E5.4 fused vs foreach AdamW: `torch.optim.AdamW(fused=True)` — ¿qué
  temporales elimina? (el bug `foreach_sqrt` OOM).

### 1.6 Checkpointing / señales / robustez
Código real: `_handle_sig` (1356), `_try_load` con checkpoints corruptos
(1294+), `_save_checkpoint` atómico.

Ejercicios:
- E6.1 Simula SIGUSR1 durante un save y explica por qué existe el lock
  `.save.lock` y el flag anti-reentrada.
- E6.2 Haz un mini-loop de entrenamiento con save/load y demuestra resume
  bit-exacto (mismo loss tras reanudar).

---

## Etapa 2 — Arquitectura del modelo

Código real: `MdLMMoE` (354-384), `Block` (325-341), `RoPEMultiheadAttention`
(290-322), `TiedHead` (344-351).

Ejercicios:
- E7.1 Implementa atención manual (QKV+softmax) y compara contra
  `nn.MultiheadAttention` — mismo init, mismos pesos, mismo output.
- E7.2 Implementa RoPE a mano y verifica `RoPEMultiheadAttention` — explica
  `cos/sin` buffers y por qué `x*cos + rot(x)*sin`.
- E7.3 Cuenta parámetros: verifica `n_params()` ≈ 1,129.6M para
  hidden=768,L=12,ff=4x,16 exp — descompón en dense/experts/embed.
- E7.4 `TiedHead`: explica weight tying y cuándo rompe (vocab+1 por MASK).
- E7.5 Pre-norm vs post-norm: ¿por qué `x + attn(ln(x))` y no `ln(x+attn(x))`?

---

## Etapa 3 — Teoría dLLM (la parte que te preguntarán)

Código real: `sample_mask_fraction` (614), `build_mask_indices` (1198),
`curriculum_state` (580), `build_candidate_mask` (1149), `_corrupt_*` (945+),
loss CE solo en posiciones enmascaradas (1615-1623).

Lecturas (en orden):
1. LLaDA (arXiv 2502.09992) — masked diffusion como lower bound
2. MDLM (Sahoo et al.) — absorbing state, t~U(0,1)
3. SEDD — score entropy discrete diffusion
4. Switch Transformer + DeepSeekMoE — routing/top-k/shared experts
5. Fast-dLLM — cache por bloques (para "¿y el KV cache?")

Preguntas que debes responder sin notas:
- ¿Por qué CE solo en posiciones enmascaradas y no en todas?
- ¿Qué es absorbing-state vs uniform noise? (RQ2 del protocolo — `docs/dllm-experimental-protocol.md`)
- ¿Por qué t ~ U(0,1) y qué implica para la varianza de la loss por step?
- ¿Cómo se genera texto si no hay orden autoregresivo? (pasos de denoising,
  schedule de unmasking, canvas fijo, bloques semi-AR)
- ¿Por qué MoE top-1 y no top-2? (costo, activación 25%, comparar con backlog
  DeepSeekMoE en ROADMAP)
- ¿Qué hace `candidate_focus` / role-weighted masking y por qué existió?
  (etapa G8-G11: foco en regiones inferenciales; G10 falsó el hinge)

Ejercicio E8: escribe un generador de denoising (`chat_proto_v2.py` como
referencia) — desde ruido total, N pasos, revelar top-confianza primero.

---

## Etapa 4 — Datos e infraestructura

- `pre_tokenize_v2.py`: blockwise npy + concat memmap — ¿por qué no listas
  Python? (28B/int × 12B tokens = OOM real ocurrido)
- `build_batches()`/`ids.npy` memmap — N ranks comparten page cache
- `v5_1b_ddp.slurm`: auto-adapt por VRAM, `--prefer`, watchdog, resubmit
- `sft_mdlm.py`: prompt fijo + response enmascarada, CE solo en response —
  la receta portada de LLaDA a MdLMMoE

Ejercicios:
- E9.1 Reimplementa el pretokenizer de dos fases (npy parciales + concat
  memmap) sobre un corpus pequeño y mide RAM vs la versión ingenua.
- E9.2 Explica por qué `_pack_with_eos` con `tolist()` era fatal a 17M docs.
- E9.3 Dibuja el flujo señal→checkpoint→resubmit del slurm y explica cada trap.

---

## Etapa 5 — Drill de entrevista (semana 3-4)

Cada subsistema → pitch 60-90s en inglés + una pregunta hostil:

| tema | pregunta hostil típica |
|---|---|
| MoE | "¿por qué tu load balancing no colapsa expertos?" (balance_loss+probe) |
| dLLM | "¿cómo generas más de seq_len tokens?" (bloques semi-AR, Fast-dLLM) |
| DDP/ZeRO | "¿qué cambia en el checkpoint entre 3 y 4 ranks?" (consolidate) |
| memoria | "¿por qué 1.13B no cabía en 24GB?" (16B/param fp32+AdamW ≈18GB) |
| debugging | "cuéntame un bug difícil" → EMA-CPU/py-spy, foreach_sqrt OOM |
| eval | "¿cómo sabes que aprende inferencia y no perplexity?" (holdout L3/num/neg) |
| GA | "¿cómo falsaste el hinge contrastivo?" (G10 control vs arms) |

War stories a ensayar (son oro en entrevistas): EMA-CPU 28s/step (py-spy),
`foreach_sqrt` OOM, npz 48GB×3 ranks OOM→memmap, `_init_ema` antes del wrap
DDP, SIGPIPE 141 por `head|nvidia-smi`, `eos_id` fuera de vocab (126081>126080).

---

## Ritmo sugerido (4 semanas, paralelo al training v5-1b)

- **S1**: Etapas 0-1 (PyTorch core) — ejercicios E1-E5
- **S2**: Etapa 2 + 3 (arquitectura + teoría dLLM, lecturas) — E6-E8
- **S3**: Etapa 4 + reimplementación guiada de un mini-trainer — E9
- **S4**: drills de entrevista + repo público pulido + writeup v5-1b

Regla: nada se marca "hecho" hasta poder defenderlo en 90s en inglés sin
mirar el código.
