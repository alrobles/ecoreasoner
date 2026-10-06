# Estrategia de escala en kuhpc — catálogo GPU + cuello real (2026-10-06)

Pregunta: cómo empujar "entrenamiento masivo y paralelo" dado el hardware del
cluster. Respuesta corta: **el cuello no son las tarjetas, es la
interconexión**. El paralelismo que rinde aquí es por cargas (fleets), y el
único pool "masivo" sin usar es AMD (MI210, vía ROCm).

## 1. Catálogo real (sinfo, sixhour — única partición GPU)

| Pool | Nodos | GPUs/nodo | Total | VRAM | Clase | Estado típico |
|---|---|---:|---:|---:|---|---|
| **mi210** | ~27 | 3 | **~81** | 64GB | ~A100 (gfx90a) | 14 alloc, resto mix — pool grande |
| a100 | 6 | 3 | 18 | 40GB | rápida | mix |
| a40 | 1 | 4 | 4 | 48GB | rápida | mix |
| l40 | 1 | 4 | 4 | 48GB | rápida | mix |
| pro6000 | 3 | 1-2 | 5 | 96GB | muy rápida (sm_120) | mix |
| q8000 | 1 | 2 | 2 | 48GB | Turing (sin bf16) | idle |
| q6000 | 8 | 3-4 | ~26 | 24GB | lenta (~1.2K tok/s/GPU) | idle/mix |
| v100 | ~15 | 2 (+1×4) | ~30 | 16GB | Volta (sin bf16) | mix |
| CPU | ~200 | 0 | — | — | — | pool enorme p/ fleets |

- `kbs`: 4 nodos, 60 días — sin GPU (servicios/watchers).
- Límite de cuenta: MaxJobs/Submit ~5000; sin QOS restrictivo visible.
- NVIDIA rápida total: **~33 GPUs, siempre compartidos**.

## 2. El cuello real: comunicación

Medido antes (`_nccl_test`): **~200 MB/s efectivos inter-nodo** (no es IB
de gama alta). Para v5-1b (1.13B params → grads fp32 ≈ 4.5 GB):

| Config | Comm/sync | Comm amortizado | Veredicto |
|---|---:|---:|---|
| DDP intra-nodo (actual, 3 ranks) | NVLink/PCIe | ~0 | óptimo por isla |
| Multi-nodo vanilla | 4.5 GB/paso | ~22 s/paso | **colapsa** (compute ~1 s) |
| Multi-nodo + `sync_every=8` | 4.5 GB /8 pasos | ~2.8 s/paso | batch efectivo ×8 — cambia receta |
| + bf16 grads | ÷2 volumen | ~1.4 s/paso | igualmente cambia receta |

`train_mdlm_moe_v2.py` ya hace `no_sync` en micro-pasos de acumulación; el
sync por optim-step es irreductible salvo localSGD (`--sync_every`, ya
implementado en `train_mdlm_moe_lsgd.py` para v4 — falta portar a v5).

**Implicación**: escalar el MISMO run a más nodos solo rinde si el batch
efectivo crece ~×30+ (→ ~300K tok/sync, receta grande estándar pero distinta
de la campeona GA). Es una decisión científica, no de infra.

## 3. Lo que sí empuja fuerte (por horizonte)

**A. Hoy — nada bloquea.** La cadena AUTO_RESUBMIT+watchdog ya roba la mejor
isla libre; cuando a40/l40 (4 GPUs) desocupe, `.wd_shape` sube a 4 ranks
(+33%) sin tocar receta. Mainline ETA ~5-6 días solo.

**B. Spike MI210/ROCm (el pool masivo).** `venv_rocm` YA existe
(torch 2.5.1+rocm6.2 en beegfs) → smoke `smoke_mi210.slurm` lanza fwd/bwd
de la arch v5-1b en 1×MI210 (job de 30 min, medir tok/s y memoria).
Si corre: 3 ranks intra-nodo ≈ isla rápida adicional; el pool (~81 GPUs)
es 2.5× toda la NVIDIA rápida junta. Riesgos: SDPA/flash en gfx90a,
RCCL para multi-nodo, dtype (bf16 soportado en MI210).
Sin apptainer — ejecución directa por venv.

**C. Fleets (ya explotado, ampliable).** Todo lo embarazoso-paralelo ya usa
el cluster correcto: mtolmo 16-way, qagen 16-way, batteries 3-job por punto.
Escalar NSHARDS es gratis mientras haya pids pendientes.

**D. Brazo paralelo de exposición (opción honesta a "más tokens").**
Con los pools lentos ociosos (q6000 ×26, v100 ×30) puede correr un SEGUNDO
backbone con receta pensada para la red: batch grande + `sync_every` alto +
clip (patrón del doc TECHO). No acelera v5-1b — duplica throughput
agregado del laboratorio y da un segundo punto de curva.

## 4. Qué NO conviene

- Multi-nodo NVIDIA con batch actual: la comm se come todo el speedup.
- v100 para pretrain: 16GB sin bf16 → bs mínimo, lento; útil solo p/ probes.
- Subir MAXDOCS/fleets contra la cola: los recursos compartidos hacen que
  arrays grandes pendientes solo quemen prioridad.

## 5. Siguiente acción concreta

1. Medir `smoke_mi210` → si ≥ ~2K tok/s/GPU, vale smoke 3-ranks intra-nodo.
2. Si MI210 funciona: clonar `v5_1b_ddp.slurm` → `v5_1b_mi210.slurm`
   (venv en vez de SIF, constraint mi210, bs mayor — 64GB cabe bs8+).
3. Decisión Angel: ¿brazo big-batch en q6000/v100 como segundo backbone?
4. QA→SFT UNAM-EN queda automático (watch_unam_qa armado → flag → SFT-v3
   sobre ckpt terminal).
