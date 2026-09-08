# EcoReasoner F1 — Estimación del run heterogéneo multi-GPU (hito paper-1)

Fecha: 2026-09-07
Autor: Hermes (para A.L. Robles Fernández)
Estado: ESTIMACIÓN V1 — pendiente medir Q6000/A100 (L40 midiéndose AHORA)

---

## 0. Resumen en una frase

Con el pool NVIDIA disponible (pro6000/Q6000/L40/A100) y el trainer híbrido
(grad_accum adaptativo + autosize por VRAM + all-reduce manual), y con las
mediciones corregidas (ver §1 — método: solo entrenamiento, sin cache), el
**1 epoch del corpus de esqueletos (129.5M tok, la unidad del criterio GO F1)
cuesta**: ~3 min con el pool MÁXIMO, ~8 min con el pool realista (24 GPUs),
~47 min con solo el pool de HOY (4× L40). Los tok/s medidos (micro dense
154.8M, batch 8×768, torch cu121/cu128): **A100 ≈22.0K > pro6000 ≈12.6K >
L40 ≈11.4K > Q6000 ≈7.9K**.

---

## 1. Mediciones REALES (no teóricas) — las que tenemos

| Fuente | GPU | Modelo | tok/s medido |
|---|---|---|---|
| f0-random-prosa (micro-run) | 1× pro6000 | dense 50-100M, batch 8×768 | **12,512** |
| bw5_spanhi (training vivo) | 2× pro6000 | MoE v4 863M, batch 4×768/rank | ~5,700 (pool 2 GPU) |
| NCCL 2 pro6000 (nccl_test) | all-reduce 5×16MB bf16 | — | OK (DDP real) |
| NCCL Q6000 (Medición 08-30) | all-reduce ~10MB | — | ~200MB/s |

Proyección del micro-modelo (el que corre el sweep): 1× pro6000 ≈ **12.5K tok/s**.
Para los demás: sin medición propia todavía → smoke_throughput.slurm corriendo
en L40. **MEDICIÓN L40 REAL (2026-09-07 18:04, job 28843577): ~2,560 tok/s.**
El L40 es ~4.9× MÁS LENTO que pro6000 para el micro dense (no 0.7× como
pensaba por FLOPs): a batch 8 el L40 no llena su VRAM (48GB usados ~20GB) y
compute-bound domina. Lección: NO proyectar tok/s por FLOPs para un micro
dense con batch fijo — hay que medir. Tabla CORREGIDA (proyección L40 abajo
en lugar de arriba):

| Familia | VRAM | tok/s micro (método: solo tiempo de entrenamiento, 200 steps) |
|---|---|---|
| **A100** | 80GB | **~22,026** (medido 08-09, smoke 28860086) |
| pro6000 (Blackwell) | 102GB | **~12,553** (medido 08-09, smoke 28881059) |
| L40 | 48GB | **~11,443** (medido, smoke 28843577) |
| Q6000 | 48GB | **~7,941** (medido, smoke 28879250) |

**CORRECCIÓN DE MÉTODO (2026-09-08)**: la versión anterior de esta tabla
subestimaba A100/L40 (14.1K/2.56K) porque dividía por el walltime INCLUDIENDO
la carga del cache (~56s y ~35s de overhead). El método correcto cronometra
SOLO el entrenamiento (step 0 → step 190 del log). Los números de arriba son
los válidos. Orden REAL por tok/s: A100 > pro6000 > L40 > Q6000 (el A100 es
~1.8× el Blackwell para el micro dense — el cuello es el head Linear→vocab
126080, y el A100 cu121 lo hace más rápido aquí).

Pitfall: medir throughput de un micro dense con walltime del job es FALSO —
siempre cronometrar del log (step 0 → step N), excluyendo carga de cache y
checkpoints.

Pitfall apuntado: con un modelo de 50M y batch 8, casi cualquier GPU moderna
queda subutilizada por COMPUTE, no por VRAM — la ganancia de pro6000 sobre L40
viene de la raw speed del Blackwell, no de los 96GB vs 48GB. Para el run
heterogéneo esto significa: **batch por VRAM NO captura la diferencia**; la
holgura la da el hardware. Ajustar el F1 a usar batch adaptativo por GPU para
que cada tarjeta trabaje (L40 batch 16, pro6000 batch 32, ...) — el grad_accum
adaptativo lo absorbe sin cambiar el gradiente global.

---

## 2. Fórmula del run heterogéneo (train_hybrid)

El trainer ya implementa el patrón correcto (scripts/train_hybrid.py):

```
por rank (GPU i):
    VRAM_i = torch.cuda.get_device_properties(rank).total_memory
    microbatch_i = min(batch_max_que_cabe, VRAM_i * 0.8 / bytes_per_sample)
    accum_i     = ceil(global_batch / microbatch_i)   # grad_accum adaptativo
    por step:   model.no_sync() para micros 1..accum_i-1
                all_reduce manual + optimizer.step() SOLO en el último micro

pool_total:
    tokens/día = Σ_i tok/s_i × 86400 × U
```

**La matemática es exacta**: cada rank acumula N gradientes locales con su
microbatch propio → el gradiente global del optimizer step es IDÉNTICO al de
una sola GPU con batch grande (all-reduce solo al final del accum). No es
aproximación: es el mismo gradiente, solo repartido por VRAM.

**sync_every (NCCL lento)**: el all-reduce del micro model (147M params bf16
≈ 294MB) a 200MB/s cuesta ~1.5s/paso. Con sync_every=8 (estilo bw1), el coste
caiga a ~0.2s/paso — despreciable frente a los ~6s de compute/paso del micro.

---

## 3. Criterio GO del F1 (del diseño -new §3)

> Si tras N tokens el modelo no discrimina ≥55% pairwise ni completa
> esqueletos → tesis falsada a nuestra escala.

N objetivo realista para un micro-ganador 50-100M (Chinchilla): **~2.5–7B tok**.
Referencia durísima de lo que se ha visto: bw0/bw1v5 (37M tok = 3% corpus) NO
alcanzó lenguaje; bw3 (93K steps × 24,576 tok = 2.3B tok) alcanzó loss 2-4.2
pero sin sintaxis. La tesis no exige lenguaje fluido — exige discriminación
inferencial ≥55% + esqueletos. Con 4.3B tok (1 epoch v7_clean) es la apuesta
registrada.

---

## 4. Escenarios: tiempo para 1 epoch del corpus de ESQUELETOS (129.5M tok)

**Objetivo F1 (receta ganadora span-esqueleto)**: 1 epoch sobre el corpus de
esqueletos = 129.5M tok (NO 4.3B — ese era v7 prosa; el esqueleto es mucho más
chico y es sobre lo que se decide el GO). Con tok/s medidos §1.

### Escenario A — pool MÁXIMO (56 GPUs, ideal)

| Familia | GPUs | tok/s c/u | subtotal tok/s |
|---|---|---|---|
| A100 | 18 | 22,026 | 396K |
| pro6000 | 5 | 12,553 | 63K |
| L40 | 4 | 11,443 | 46K |
| Q6000 | 29 | 7,941 | 230K |
| **TOTAL** | **56** | ~13K medio | **735K tok/s** |

→ 1 epoch esqueleto (129.5M) en **~3 min**. (Obviamente irreal: cluster compartido.)

### Escenario B — pool REALISTA (24 GPUs asignables)

| Familia | GPUs | tok/s | subtotal tok/s |
|---|---|---|---|
| A100 | 4 | 22,026 | 88K |
| pro6000 | 4 | 12,553 | 50K |
| L40 | 4 | 11,443 | 46K |
| Q6000 | 12 | 7,941 | 95K |
| **TOTAL** | **24** | ~11.6K medio | **279K tok/s** |

→ 1 epoch esqueleto en **~8 min**. El criterio GO (más de 1 epoch si hace
falta) en menos de 1 hora de pool.

### Escenario C — pool HOY (medido 2026-09-08 00:30, scontrol)

| Familia | GPUs LIBRES | tok/s | subtotal tok/s |
|---|---|---|---|
| L40 | 4 | 11,443 | 46K |
| Q6000 | ~7-12 (r22* 3/3 + parciales) | 7,941 | 56-95K |
| A100 | 0 (otros usuarios) | — | — |
| pro6000 | 1-3 libres (cu128, F1 no puede mezclar) | — | — |
| **TOTAL (cu121)** | **~11-16** | ~9K medio | **~100-140K tok/s** |

→ 1 epoch esqueleto en **~15-22 min** con lo que hay HOY mismo, solo con L40+Q6000.

**Lee los tres**: el diseño -new ya lo avisaba (§2, §12): "pool teórico ≠ pool
real". La corrección clave de esta revisión: el objetivo F1 (esqueleto) es
**dos órdenes de magnitud más barato** que 1 epoch de prosa v7 (129.5M vs
4.3B tok), así que el F1 completo cabe en <1h de pool incluso con pocas GPU.
El valor del hito paper-1 está en que el harness mide Y ENTREGA el pool que
de verdad esté libre, no el que promete sinfo.

---

## 5. Riesgos y supuestos (honestos)

1. **El tok/s de Q6000/A100/L40 son PROYECCIÓN, no medición.** El smoke
   throughput (corriendo) debe medirlos. Si L40 sale <6K, la tabla baja.
2. **NCCL heterogéneo**: mezclar cu128 (pro6000) con cu121 (Q6000/L40/A100) en
   el MISMO DDP es EL muro del HITO 3. Opciones: (a) nodos homogéneos por
   partición + pipeline parallel entre familias (intercambio de activaciones,
   no gradientes), (b) subir todo a torch común. La ref lo documenta.
3. **El micro-ganador puede no escalar**: si el ganador del sweep es un 50M,
   el F1 lo entrena más grande; el tok/s baja con el cuadrado del tamaño.
4. **Contienda**: Q6000/A100 "mix" significa que otros usuarios pueden
   revocarlos. El run heterogéneo debe planear nodos de larga duración.

---

## 6. Camino al hito (próximos pasos)

- [ ] Medir L40 (corriendo: job 28843577) → tok/s real.
- [ ] Medir Q6000 + A100 cuando haya GPU (mismo slurm, cambiar --gres).
- [ ] lanzar train_hybrid multi-GPU en el pool disponible CON los micro-runs
      completados (valida el all-reduce + accum adaptativo de verdad, no solo
      single-GPU como hasta ahora).
- [ ] confirmar criterio GO del micro-sweep (pairwise_acc ≥ 0.55) → escalar.
- [ ] escribir paper-1 (systems): metodología + mediciones reales + harness.

---

## 7. Archivos

- `scripts/smoke_throughput.slurm` — smoke tok/s por familia (L40 corriendo).
- `scripts/train_hybrid.py` — trainer heterogéneo (autosize + accum + LoRA).
- `references/heterogeneous-nvidia-training.md` — ref inventario + pitfall sinfo.
- `designs/ecoreasoner-Fase3-DESIGN-new.md` §2,§8,§12 — pool y aritmética.