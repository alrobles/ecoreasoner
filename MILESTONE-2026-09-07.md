# MILESTONE 2026-09-07 — EcoReasoner: micro-sweep F0 6/6 + trainer F1-hetero validado

Date: 2026-09-07 22:5x CDT
Status: F0/F0b cerrados; F1-hetero infra validada pendiente de pool multi-familia.

================================================================================
1. MICRO-SWEEP F0 — 6/6 COMPLETADO Y EVALUADO
================================================================================
Los 6 micro-runs (dense 50-100M, 10K steps, 1 GPU pro6000, seed 7331) entrenaron
y completaron la eval de discriminación inferencial (suite_smoke + report.json +
index.jsonl). Fuente de verdad: /beegfs/a474r867/ecoreasoner/runs/index.jsonl.

  | Run                    | pairwise_acc | mean_delta |
  |------------------------|-------------|-----------|
  | f0-random-prosa        | 0.5117       | -0.032     |
  | f0-span-prosa          | 0.4492       | -0.064     |
  | f0-spanhi-prosa        | 0.4766       | -0.052     |
  | f0-random-esqueleto    | 0.5078       | +0.011     |
  | f0-span-esqueleto (GAN)| 0.5352       | +0.015     |
  | f0-spanhi-esqueleto    | 0.5312       | +0.030     |

  - Criterio GO pre-registrado (>=0.55): NO alcanzado (mejor 0.5352).
  - Tesis REFORZADA: esqueleto mueve el delta a POSITIVO (elige la correcta),
    prosa lo deja NEGATIVO (elige la mala); span > random en esqueleto.
    Token budget 245K = 0.006% corpus -> subentrenado, no falsada.
  - GANADOR para F1: span-esqueleto (span 64 @15% sobre esqueletos).

  Corpus de esqueletos: train_skeleton.jsonl (315,299 docs ge3, 675MB) +
  train_ids_skeleton.npy (129.5M tok, OOB_OK). Cobertura real 21.6% ge3.

================================================================================
2. F1-HETERO — ENTRENAMIENTO HETEROGÉNEO MULTI-GPU (hito paper-1)
================================================================================
  - train_mdlm_moe_hetero.py: copia del trainer + --autosize (microbatch por
    rank según VRAM, medido con fwd+bwd real) + SCALE_I (gradiente del
    optimizer EXACTO por token aunque las GPUs difieran) + grad_accum global.
  - VALIDADO: smoke L40x4 (job 28864131, r32r25n01, world=4, micro=25/rank,
    100 steps, loss 8.79, COMPLETE).
  - f1_hetero.slurm: lanzador multi-job multi-familia (torchrun rendezvous TCP
    en MASTER_ADDR:PORT; cada job = 1 familia).
  - MURO NCCL documentado: L40 r32r25n01 es noib vs A100 ib -> NCCLUtils.hpp:275.
    Fix NCCL_IB_DISABLE=1 + NCCL_SOCKET_IFNAME en el slurm (pendiente validar
    con 2 familias: watchdog f1-multifam-validate lanza cuando haya A100).
  - Tok/s medidos (micro dense): A100 14,128 > pro6000 12,512 > L40 2,560.

================================================================================
3. ARRANQUE LIMPIO (cadena F0b->F0), con fixes
================================================================================
  - Extractores/pretok/full/pairs: skel-full COMPLETED 315,299 docs; skel-pretok
    OOB_OK; skel-pairs 256 pares.
  - Bugs cazados y FIJO (git commit 540024e, c083ec4, 2195adb):
    YAML flow-mapping invalido; sbatch --export parte por comas (usar ;);
    suite_smoke.generate() tensor en CPU (device mismatch); VENV duplicado;
    build_pairs leia etapas como dict (el full escribe int+texto serializado);
    autosize no_grad subestimaba el head (OOM); grad_accum por rank des-sincroniza DDP.

================================================================================
4. ESTADO VIVO AL CIERRE
================================================================================
  - bw5_spanhi: step ~27,300/38,000, loss ~5.5, sin flag (ola en curso). Cierre
    de la curva de ablación (span hi 60%); pending smoke_cont al completar.
  - Watchdogs: d7d8067a09d5 (sweep+bw5, 10m) + 5fa09d642d89 (multi-familia, 15m).
  - Pool: L40 4/4 libres; A100 ocupados (r31*); Q6000 1-2 sueltos; pro6000 libre
    cu128.

================================================================================
5. PRÓXIMO (para retomar; ver ROADMAP.md §5)
================================================================================
  [ ] Validar NCCL multi-familia (watchdog 5fa09d642d89 lanza solo)
  [ ] F1 completo: span-esqueleto, pool cu121, TARGET 1 epoch esqueleto
  [ ] Medir Q6000 (smoke_throughput --gres=q6000)
  [ ] bw5_spanhi: cuando COMPLETE -> smoke_cont (rep4->0.15 Y word->0.435)
  [ ] Evaluar pairwise_acc cada checkpoint del F1 vs runs/pairs.jsonl