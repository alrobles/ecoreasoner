# Romper el techo multi-nodo Q6000 — estudio de comunicación y gradientes (2026-08-28)

> Objetivo: entrenar el MoE v4 (850M total / ~420M activos, config bw0) en 4 nodos
> Q6000 y partir la barrera que DDP vanilla impone cuando el all-reduce de
> gradientes domina sobre el cómputo local.

## 1. Inventario real de nodos Q6000 (verificado 2026-08-28)

| Nodos | GPUs/nodo | Cores | RAM | Partición |
|---|---|---|---|---|
| r15r05n01, r15r10n01 | **4** | 32 | 190GB | sixhour |
| r22r05n01 … r22r35n01 (7 nodos) | **3** | 24 | 190GB | sixhour |

- **No existen 4 nodos × 4 tarjetas.** El máximo homogéneo de 4 nodos es
  **4×3 = 12 ranks** (4 nodos r22), o **2×4 = 8 ranks** (2 nodos r15, más
  potentes por núcleo). Configuración mixta 4+4+3+3 = 14 ranks posible con
  `--exclude` explícito, pero eso NO es recomendable al inicio (heterogéneo:
  nodos de 3 y 4 GPUs comparten el mismo all-reduce; la partición de datos por
  rank sigue siendo uniforme, así que funciona — el riesgo es solo de
  disponibilidad).
- Todos `mix` (parcialmente ocupados) ahora; los jobs de teachers ocupan
  r30r08n01/r22r05n01/r30r24n01 (no compiten con r22r10+).

## 2. El techo: costo de comunicación vs cómputo

**Medición previa real** (`_nccl_test.py`, 2 nodos, bf16): all_reduce 10MB × 5 =
0.5s → **~50ms/10MB → ~200 MB/s efectivo entre nodos** (interconexión Q6000 no es
InfiniBand de alta gama; esto domina el diseño).

**Modelo 850M** (config bw0: hidden 1024, layers 16, 4 exp top-1, seq 768):
gradientes fp32 ≈ 4×850M = **3.4 GB por all-reduce completo**.

| Escenario | Comunicación por paso | Tiempo red / paso |
|---|---|---|
| DDP vanilla, 1 grad_accum | 3.4 GB | **~17 s** |
| grad_accum 4 (reparte cómputo, NO red: DDP all-reducea por backward) | 3.4 GB | ~17 s |
| **post-localSGD sync_every=8** (no_sync 7 pasos + sync 1) | 3.4 GB cada 8 | **~2.1 s/paso** |
| sync_every=16 | 3.4 GB cada 16 | ~1.1 s/paso |

**Punto clave que se discutió y ahora está cuantificado**: con DDP vanilla el
**grad_accum NO reduce comunicación** — DDP lanza el all-reduce en cada backward.
La única forma de bajar la red es reducir la FRECUENCIA de sync (post-localSGD
con `no_sync()`) o reducir el VOLUMEN (MoE-shard all-to-all, ver §4).

Cómputo estimado por paso (1 nodo Q6000, batch 4, seq 768, 420M activos —
mitad del bw0 actual) ≈ ~0.5-1 s/step. Con DDP vanilla la red (17s) sería
**20-30× el cómputo** → throughput colapsa. Con sync_every=8 la red (~2s) y el
cómputo (~4-8s agregado) quedan en el mismo orden → **escalas real con 12 ranks**.

## 3. Implementación preparada (NO lanzada)

- `scripts/train_mdlm_moe_lsgd.py` — variante del trainer con:
  - `--sync_every K`: los K-1 backwards van en `DDP.no_sync()` (grad local),
    el K-ésimo all-reducea y hace `optim.step()`. Ciclo de acumulación =
    grad_accum × K. Default 1 = comportamiento EXACTO del vanilla.
  - `--clip_grad C`: grad norm clip (robustez multi-nodo).
  - Mismo checkpoint/resume/onda SIGUSR1 del base; checkpoint compatible.
- `scripts/moe_v4_q6000_4n.slurm` — plantilla 4 nodos (por defecto 4×3=12
  ranks, parametrizable con `--export NODES/TASKS/...`):
  - SIF base `pytorch-cuda.sif` SIN overlay (Q6000 sm_86; el overlay blackwell
    es solo para sm_120).
  - `MASTER_ADDR` exportado desde host con scontrol (fix multinodo validado).
  - Ondas SIGUSR1 + AUTO_RESUBMIT hasta TARGET_STEPS (patrón bw0).
  - `NCCL_DEBUG=INFO` en primera ola para diagnóstico.

## 4. Opciones para "romper el techo" (orden de implementación)

1. **post-localSGD (`sync_every`) — YA preparado.** Divide la red por K.
   Ajuste: lr escala ~ √K por paso efectivo (regla empírica local-SGD; validar).
2. **Grad clip — YA preparado.** Elimina picos de grad > 1 que en multi-nodo
   (menos lotes por sync) desestabilizan el entrenamiento.
3. **MoE-shard + all-to-all (GShard/Mixtral)** — siguiente arquitectura cuando
   el vanilla toque techo de modelo más grande: expertos repartidos entre GPUs,
   grad de experto **100% local**, SOLO all-reduce de params densos
   (atención/embeddings/gate ≈ 270M → 1.1 GB → con sync_every=16 ≈ 0.35 s/paso).
   Pendiente de diseño; no es urgente a 850M.
4. **bf16 grad all-reduce** (reduce a la mitad el volumen; NCCL `_BF16` en el
   all-reduce manual) — barato de añadir cuando toque.
5. **LocalSGD de pesos (post-localSGD clásico)**: promediar pesos cada K pasos
   en vez de gradientes — aún más robusto a la red, pero cambia la dinámica de
   optimización; mantener como plan C.

## 5. Criterio de validación (smoke antes de la ola larga)

1. `--max_steps 40 --sync_every 4` en 2 nodos × 2 Q6000 → loss baja, sin
   hang, `(sync)` en log cada 4 pasos, checkpoint ok.
2. Medir tiempo/paso con 1 rank vs 12 ranks para cuantificar la ganancia real.
3. Ola larga solo tras smoke (regla de labor: preguntar antes de lanzar).

## 6. Datos/parámetros de la ola larga (sugeridos, no lanzados)

```
NODES=4 nodos r22 × 3 GPU = 12 ranks
HIDDEN=1024 LAYERS=16 N_EXPERTS=4 EXPERT_K=1 SEQ=768
BATCH=4 GRAD_ACCUM=4 → batch efectivo local 16, global 192 (vs bw0 global 8)
SYNC_EVERY=8 CLIP=1.0 LR=2e-4 (escalar ~√8 si se valida) TARGET_STEPS=6000
```