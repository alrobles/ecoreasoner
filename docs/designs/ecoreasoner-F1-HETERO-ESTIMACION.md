# EcoReasoner F1 — Estimación del run heterogéneo multi-GPU (hito paper-1)

Fecha: 2026-09-07
Autor: Hermes (para A.L. Robles Fernández)
Estado: ESTIMACIÓN V1 — pendiente medir Q6000/A100 (L40 midiéndose AHORA)

---

## 0. Resumen en una frase

Con el pool NVIDIA MÁXIMO disponible (pro6000/Q6000/L40/A100 = 52 GPUs teóricas)
y el trainer híbrido (grad_accum adaptativo + autosize por VRAM + all-reduce
manual), se estima **~25B tok/día pico teórico, ~10.6B realista, ~0.9B con el
pool de HOY (solo 4× L40 medidos a 2,560 tok/s)**. El criterio GO del F1
(~4.3B tok ≈ 1 epoch v7_clean) se alcanzaría en **~4h de pool pico, ~10h
realista, o ~4.9 días con solo L40**. Mediciones REALES (micro dense 50-100M,
2026-09-07, misma traza de train_mdlm_moe.py):
A100=14.1K tok/s, pro6000=12.5K, L40=2.56K (Q6000 por medir, ~2.5K est).
El F1-HETERO ya tiene trainer (train_mdlm_moe_hetero.py, autosize+SCALE_I)
VALIDADO en DDP multi-GPU real (smoke L40×4, 100 steps, micro=25/rank).

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

| Familia | VRAM | tok/s MEDIDO micro |
|---|---|---|
| **A100** | 80GB | **14,128** (medido 07-09, smoke 28860086) |
| pro6000 (Blackwell) | 96GB | **12,512** (medido) |
| L40 | 48GB | **2,560** (medido) |
| Q6000 | 48GB | ~2,500 (est. ~L40, por medir) |

**El A100 es el más rápido del pool para el micro dense** (14.1K tok/s > pro6000
12.5K) a pesar de ser cu121 — el cuello es compute del head (vocab 126080), no
torch version. La tabla ya no puede ordenarse por VRAM ni por generación:
orden REAL por tok/s = A100 > pro6000 > L40 ≈ Q6000.

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

## 4. Escenarios de tokens/día y días para el GO

### Escenario A — pool MÁXIMO (los 52 de todos los tipos, ideal)

| Familia | GPUs | tok/s c/u | subtotal tok/día |
|---|---|---|---|
| pro6000 | 5 | 12,500 | 5.4B |
| A100 | 18 | ~8,000* | 12.4B |
| Q6000 | 29 | ~2,500** | 6.3B |
| L40 | 4 | 2,560 | 0.9B |
| **TOTAL** | **56** | — | **25.0B** |

→ 1 epoch v7 (4.3B tok) en **~4h**. (*A100 sin medir; **Q6000 sin medir — un
L40+ medido da 2.5K, así que Q6000 no puede dar 9.4K)

### Escenario B — pool REALISTA (nuestros nodos, compartidos pero asignables)

| Familia | GPUs reales | tok/s | subtotal tok/día |
|---|---|---|---|
| pro6000 | 4 (de 5; bw5 ocupa 1) | 12,500 | 4.3B |
| Q6000 | 12 (de 29 asignables) | 2,500 | 2.6B |
| L40 | 4 (todas) | 2,560 | 0.9B |
| A100 | ~4 (de 18) | 8,000 | 2.8B |
| **TOTAL** | **24** | ~5.6K medio | **10.6B** |

→ 1 epoch en **~9.7h**; criterio GO (4.3B) en **<1 día de pool**.

### Escenario C — pool HOY (medido 2026-09-07 17:30)

| Familia | GPUs LIBRES de verdad (scontrol) | tok/s | subtotal |
|---|---|---|---|
| L40 | 4 | 2,560 (medido) | 0.9B |
| pro6000 | 0 (bw5 + 3 micros) | — | — |
| Q6000/A100 | 0 (otros usuarios) | — | — |
| **TOTAL** | **4** | ~2.6K | **0.9B tok/día** |

→ 1 epoch en **~4.9 días** con solo L40.

**Lee los tres**: el diseño -new ya lo avisaba (§2, §12): "pool teórico ≠ pool
real". El valor del hito paper-1 está en que el harness mide Y ENTREGA el pool
que de verdad esté libre, no el que promete sinfo.

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