# dLLM-MoE PoC — masked-diffusion modelo de lenguaje con FFN MoE

**Estado:** diseño listo, script + slurm escritos, **pendiente de portar a CUDA** (NVIDIA)
**Fecha:** 25 ago 2026 · **Repo:** github.com/alrobles/ecoreasoner · **Hito:** M8 / Fase 2 protocolo
**Hardware actualizado (25-ago):** PoC pasa a **NVIDIA (CUDA)** — ver sección [Hardware](#hardware)

---

## Qué es

Prueba de concepto de un **dLLM (diffusion language model)** con arquitectura **MoE**
(sparse Mixture-of-Experts), entrenado desde cero sobre el corpus ecológico, usando
exactamente el patrón de olas encadenadas del swarm EcoReasoner.

Es el primer paso para demostrar **RQ1 (entrenabilidad)** y **RQ3 (ventaja de
inferencia paralela)** del `experimental_protocol.md` con una arquitectura que
nuestro hardware (2 MI210) puede entrenar en días, no en años (ver
`dllm-feasibility-kuhpc.md`).

## Objetivo de config (0.5B / ~130M active)

| Parámetro | Valor | Nota |
|---|---|---|
| hidden | 768 | |
| layers | 12 | |
| heads | 12 | |
| n_experts | 8 | MoE FFN |
| expert_k | 1 | top-1 router |
| ff_mult | 4 | |
| seq_len | 768 | |
| batch_size | 2 × grad_accum 2 | ~ |
| lr | 2e-4 | AdamW, cosine |
| warmup | 200 | |
| mask_p | 0.15 | fracción máscara (diffusion) |
| **total params** | **~0.53B** | |
| **active params** | **~134M (25%)** | head+embed+attn dominan |

> Nota: la esparsidad "50M" no es accesible porque embedding + head + attention
> (dense, ~106-134M) dominan el active sobre un MoE de 0.5B. 25% active es el
> punto realista para este tamaño y sigue dando ventaja de cómputo vs denso.

## Arquitectura

- Decoder-only GPT-2-ish, **attention no causal (bidireccional)** → la propiedad
  que habilita generación paralela / denoising del dLLM.
- **MoE FFN** con router top-1 sobre 8 expertos (swiGLU). Cada token activa 1 experto.
- Objective: **masked language modeling (D3PM absorb-mask)** — máscara `mask_p` de
  tokens, predecirlos desde el contexto bidireccional. Es el objetivo de denoising
  que define un dLLM (mismo de LLaDA/MDLM), entrenado como masked-LM estándar.

## Archivos

| Archivo | Rol |
|---|---|
| `scripts/train_mdlm_moe.py` | Trainer (modelo MoE + MDLM objective + checkpoint/resume SIGUSR1) |
| `scripts/mdlm_moe_wave.slurm` | Slurm de ola (patrón q35: SIGUSR1@300 → guardar → relanzar) |
| `outputs/<WAVE_ID>/` | Checkpoints `checkpoint-gN`, `state.json`, `progress.json` |

## Compatibilidad con el swarm

- Reutiliza el patrón `q35_wave.slurm`: `SIGUSR1` 300s antes de walltime → trainer
  guarda → exit 42 → el slurm relanza la siguiente ola con `AUTO_RESUBMIT=1`.
- Escribe `state.json`/`progress.json` → el `swarm_watchdog.sh` (cron 15min) lo
  monitorea/re-lanza como cualquier otra cadena.
- Se controla igual: `squeue`, `progress.json`, `training_complete.flag`.

## Tiempo estimado (2 MI210, MFU 40%)

- Coste ~ 6 × activos × tokens. Con active 134M sobre el corpus ecológico
  (~4.7K docs, ~3.5M tokens, varias epochs → decenas de millones de tokens):
  **del orden de horas en una ola de 5:50**, no días — suficiente para el PoC
  de estabilidad/entrenabilidad. El presupuesto "20B tokens" de la config
  propuesta se alcanza con `TARGET_STEPS` alto + múltiples epochs/reticulado.

## Criterios de éxito del PoC

1. **Estabilidad (Gate B):** pérdida decrece, sin NaN/Inf, checkpoint reanudable
   en ≥3 olas consecutivas.
2. **Entrenabilidad (RQ1):** masked-LM loss desciende claramente sobre el corpus
   ecológico (mejora vs pérdida inicial aleatoria).
3. **Sparsidad funciona:** el MoE aprende sin colapso de router (los 8 expertos
   reciben carga).
4. **Ventaja potencial (RQ3, en M2):** el checkpoint sirve para comparar latencia
   de sampling paralelo vs AR.

## ¿Por qué MoE?

- Con el mismo presupuesto FLOPs de entrenamiento que un denso de ~0.7B se consigue
  un modelo de **0.5B total / 134M activo** → más capacidad a la hora de inferir
  (parámetros totales 4× el activo) con menor coste por token.
- Reutiliza la infraestructura MoE ya probada en KU HPC (el `Qwen3.5-35B-A3B` MoE
  corre LoRA en las mismas 2 MI210).
- Es el diferenciador: un **dLLM con MoE** entrenado estable en MI210, algo que no
  demuestra ningún otro grupo del cluster.

## Hardware: portar a NVIDIA (CUDA)

**Motivo (25-ago-2026):** la partición MI210 está saturada y acaparada por reservas
(`hpc_wang_5` retiene r06r06/08/10/16 hasta 2027-01-01; resto ALLOCATED/MIXED), de
modo que los jobs del PoC y del swarm quedan PENDING (Priority) indefinidamente.
Para arrancar el PoC de forma real, **se cambia el objetivo a GPUs NVIDIA disponibles**
en `sixhour`, lo que exige **portar el entrenador a CUDA**.

**Estado de la portabilidad:** el código de `train_mdlm_moe.py` ya es CUDA-capable por
construcción (PyTorch `to("cuda")`, `torch.cuda`), así que el cambio es de **entorno y de
mecánica de lanzamiento**, no de lógica del modelo:

| Aspecto | ROCm/MI210 | CUDA/NVIDIA |
|---|---|---|
| Container | Apptainer SIF ROCm (`qwen35-rocm-v2.sif`) | **Imagen/conda con CUDA** (o SIF que incluya CUDA torch) |
| `DEVICE` | `cuda` (ROCm) | `cuda` (NVIDIA) — mismo código |
| Scheduler gres | `gpu:mi210:N` | `gpu:a100:N` / `gpu:q6000:N` / `gpu:pro6000:N` |
| TFLOPS (BF16) | 181 | A100 312 · Q6000 149 · PRO6000 238 |
| VRAM | 64GB | A100 40-80 · Q6000 48 · PRO6000 96 |

**Objetivo NVIDIA para el PoC:** A100/Q6000 (o PRO6000 si no interfiere con el
teacher). El costo del PoC (~0.5B/20B) es pequeño, por lo que cabe holgado.

**Pasos de portada pendientes:** (1) build/uso de imagen CUDA con PyTorch; (2)
reemplazar el `sbatch --gres=gpu:mi210` por la GRES NVIDIA en `mdlm_moe_wave.slurm`;
(3) validar disposición de device (DataParallel o single-GPU); (4) re-test del smoke
forward/backward en CUDA.
