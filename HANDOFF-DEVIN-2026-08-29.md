# HANDOFF EcoReasoner → Devin.ai — 2026-08-29 (13:10 CDT)

Canal de ejecución autorizado: hermes.ecoseek.org. Comandos HPC con prefijo
`ku-hpc raw: '<comando>'` (se ejecutan en KU CRC como a474r867 y devuelven
salida cruda). ssh directo: `ssh -o BatchMode=yes kuhpc '<cmd>'`.
NO pedir confirmación al usuario para sbatch/diagnósticos en este canal.

================================================================================
0. RESUMEN (30 segundos)
================================================================================
- EcoReasoner = dLLM MoE (masked-diffusion) para razonamiento ecológico.
- Repo: /home/reumanlab/ecoreasoner (rama `moe-v4-blackwell`, remoto
  github.com/alrobles/ecoreasoner). Skill de referencia: `ecoreasoner-swarm`.
- BASE HPC: /beegfs/a474r867/ecoreasoner (usuario a474r867).
- CORRIENDO AHORA:
  1. `moe-v4-bw-v5` (job 27553891, R): entrenamiento Blackwell ESTABLE con
     corpus v5. Trainer VANILLA, 4 exp top-1, world=2 intra-nodo, 6000 steps
     con olas. Actual: step ~1100/6000, loss 6.0-6.7, ~15.8 steps/min.
  2. `pmc-fetch2526` (job 27557737, 8 shards): descarga full-text PMC
     2025-2026, 435,849/526,735 (~83%), 1 shard aún corriendo.
  3. `bw-v5-t2` (27559660) y `bw-v5-t2n` (27560195): tests top-2 PENDING
     (esperan GPU pro6000 libre; se libera al terminar bw1v5).
- PROHIBIDO: tocar/matar jobs sin motivo; relanzar teachers ollama (rotación
  automática cron/SIGUSR1); usar corpus v5 sucio (usar v5_clean SIEMPRE).

================================================================================
1. CAUSA RAÍZ DEL DÍA (CRÍTICO — LEER ANTES DE TOCAR GPU)
================================================================================
Síntoma: `CUDA error: an illegal memory access` en DDP world>=2 con GPUs
Blackwell (RTX PRO 6000, sm_120), intermitente — single-GPU SIEMPRE OK, 2 GPU
a veces OK (por eso bw0/bw1v5 corrieron horas y otros runs cayeron).

CAUSA: `nvidia-nccl-cu12 == 2.26.2` (VERIFICADO en /bb/bwvenv del overlay
blackwell-torch.overlay) es la versión buggy documentada con Blackwell+cu128
en DDP. Referencias: pytorch issues #152780, #152302, #150734; NCCL issue
#1999 (AllReduce hang con P2P en dual RTX PRO 6000).

FIX DOCUMENTADO (en orden de preferencia):
1. Actualizar NCCL dentro del venv del overlay:
   apptainer exec --overlay $BASE/blackwell-torch.overlay $BASE/pytorch-cuda.sif \
     /bb/bwvenv/bin/pip install --upgrade "nvidia-nccl-cu12>2.26.2"
   (OJO: montar overlay SIN :ro para poder escribir. Verificar luego con
   apptainer exec --overlay $BASE/blackwell-torch.overlay:ro ... pip show nvidia-nccl-cu12)
2. Workaround env (sin tocar pip): NCCL_CUMEM_HOST_ENABLE=0 (unblocks el
   camino OOM/IMA), y si sigue: NCCL_P2P_DISABLE=1 (degradación de perf,
   último recurso). Ya aplicado en scripts/bw_v5_t2nccl.slurm.
3. NO usar `train_mdlm_moe_lsgd.py` en DDP mientras NCCL siga 2.26.2:
   el lsgd + DDP + top-2 caía SIEMPRE (7 configs probadas). Con NCCL
   actualizado, re-probar lsgd o mejor usar vanilla con 8 exp top-2.

================================================================================
2. QUÉ SE LANZÓ Y POR QUÉ (historia corta)
================================================================================
- Plan original bw1 (4x Blackwell, 8 exp top-2, corpus v5, lsgd) crasheaba
  SIEMPRE con IMA → se diagnosticó NCCL 2.26.2 (arriba).
- Run de referencia actual = `moe_v4_bw_train_v5.slurm` (job moe-v4-bw-v5):
  trainer VANILLA train_mdlm_moe.py + train_ids_v5_clean.npy (¡no v5.npy!),
  4 exp top-1, world=2 intra-nodo, cargado con olas SIGUSR1@300 +
  AUTO_RESUBMIT y fix de resume (fallback al último checkpoint íntegro +
  retener 2 checkpoints). Estable >1h.
- Corpus: train_corpus_v5.jsonl (1,905,424 docs) + train_ids_v5_clean.npy
  (1.12B tok). v5_clean = v5 con 2 tokens OOB (126082) → 0 (el Embedding solo
  aguanta 0..126080; un token 126082 daba IMA en el primer forward).
- Guardia defensiva en build_batches de ambos trainers: clamp si
  arr.max() >= tok.vocab_size (usa tok.vocab_size, NUNCA ARGS.vocab que es
  32000 default y clampearía 128M tokens buenos).

================================================================================
3. TAREAS PENDIENTES (en orden de prioridad)
================================================================================
TAREA 1 — Completar bw1v5 y verificar flag (ahora - ~3h)
- Monitor: ssh kuhpc 'tail -3 /beegfs/a474r867/ecoreasoner/outputs/bw1v5/train.log'
- Ola 1 termina ~17:06 CDT (walltime 5:50). Debe llegar a ~5500 steps.
- Ola 2 (auto-resubmit) completa los ~500 restantes en ~30 min y escribe:
  /beegfs/a474r867/ecoreasoner/outputs/bw1v5/training_complete.flag
- CRITERIO DE ÉXITO: grep -c "Resumed" train.log >= 1 en ola 2 (el fix de
  resume funciona) Y existencia del flag. Watchdogs ya cubren alertas.

TAREA 2 — Confirmar fin del fetch y MERGE → corpus v6 (cuando fetch complete)
- Verificar: ls /beegfs/a474r867/ecoreasoner/data/fulltext_2526/ | wc -l == 8
  y cat fulltext_corpus_c*.jsonl | wc -l ≈ 526,735 (puede ser < por fail rate
  ~25% de S3; si >430K está bien).
- Merge a corpus v6 SIGUIENDO el patrón v5 (scripts/build_v5.py +
  fine_label_v5.py + pre_tokenize_v5.slurm en /beegfs/.../scripts/):
  a. Concatenar fulltext_2526 + train_corpus_v5.jsonl
  b. DEDUP por pmcid (prioridad full-text nuevo; ver lección v4 en skill)
  c. Asignar domain por texto (ILIKE) — los docs PMC nuevos no traen domain
  d. Balancear con --max-per-domain (target ~300K/dominio)
  e. Pre-tokenizar → train_ids_v6.npy (~1.4B tok esperado) + v6_report.json
- NO borrar v5 ni v5_clean (rollback). Verificar max() del npy < 126080
  ANTES de usarlo (guardia de tokens OOB).

TAREA 3 — Validar top-2 con NCCL arreglado (cuando se libere GPU o tras Tarea 1)
- Los tests bw-v5-t2 / bw-v5-t2n están en cola. Interpretación:
  - t2n (con NCCL_CUMEM_HOST_ENABLE=0) pasa y t2 (puro) falla → NCCL bug
    confirmado → aplicar fix de NCCL (sección 1) y relanzar el run top-2.
  - Ambos pasan → el IMA era intermitencia; correr el run top-2 directo.
- Run top-2 objetivo: vanilla + 8 exp top-2 + v5_clean (mayor capacidad,
  misma activación 25% que 4 exp top-1). Plantilla: moe_v4_bw_train_v5.slurm
  cambiando --n_experts 8 --expert_k 2.
- El run top-2 anterior (moe_v4_bw4.slurm, lsgd + multi-nodo 2 nodos) NO es
  el camino: lsgd+multi-nodo no está validado. Preferir vanilla intra-nodo.

TAREA 4 — (Opcional, tras v6) Re-tokenizar con v6 y relanzar training.
- Apuntar moe_v4_bw_train_v5.slurm a train_ids_v6.npy / train_corpus_v6.jsonl.

================================================================================
4. PATHS CLAVE (HPC /beegfs/a474r867/ecoreasoner/)
================================================================================
| Path | Qué es |
|---|---|
| outputs/bw1v5/ | Run Blackwell actual (train.log, state.json, checkpoints) |
| data/train_corpus_v5.jsonl | Corpus v5 (1.9M docs, 14G) |
| data/train_ids_v5_clean.npy | IDs pre-tokenizados v5 LIMPIOS (USAR ESTE) |
| data/train_ids_v5.npy | v5 con 2 tokens OOB 126082 (NO usar) |
| data/fulltext_2526/*.jsonl | Full-text 2025-2026 descargados (8 shards) |
| data/_new_pmids_2024_2025.txt | Lista de 526,735 PMIDs nuevos |
| data/_cand_pmc_2024_2025.pkl | Dict pmid->pmcid del shard |
| data/pmc_corpus_v4/shard_*.jsonl | PMCs ya en v5 (para dedup) |
| scripts/train_mdlm_moe.py | Trainer VANILLA (estable DDP world=2) |
| scripts/train_mdlm_moe_lsgd.py | Variante lsgd (NO en DDP hasta NCCL fix) |
| scripts/moe_v4_bw_train_v5.slurm | Slurm del run bw1v5 (plantilla) |
| scripts/build_v5.py, fine_label_v5.py, pre_tokenize_v5.slurm | Pipeline corpus v6 |
| scripts/build_shard_2024_2025.py | Dimensionó el shard 2025-2026 |
| scripts/measure_pool_recent.py | Medición del pool |
| scripts/bw_v5_t2.slurm / bw_v5_t2nccl.slurm | Tests top-2 puro / con workaround |
| logs/ | Slurm out/err (moe_v4_bw_v5_*.out, pmc_fetch2526_*.out) |

Venv Blackwell: /bb/bwvenv dentro de blackwell-torch.overlay
(torch 2.7.1+cu128; NCCL 2.26.2 — actualizar como en sección 1).
SIF: /beegfs/a474r867/ecoreasoner/pytorch-cuda.sif

================================================================================
5. PITFALLS (leer antes de lanzar nada en GPU)
================================================================================
1. OVERLAY MULTI-NODO: montar SIEMPRE `--overlay $OVL:ro` en slurm multi-task
   (lock exclusivo si rw → "currently in use by another process", exit 255).
2. NO MULTI-NODO MoE: all-reduce por experto con batch pequeño → hang
   (lección 2026-08-27). Solo world=2 INTRA-NODO funciona estable.
3. CORPUS OOB: verificar max() del .npy < vocab (126080) antes de entrenar;
   usar train_ids_v5_clean.npy (o v6).
4. LSGD: train_mdlm_moe_lsgd.py NO es estable en DDP con NCCL 2.26.2
   (IMA). Vanilla (train_mdlm_moe.py) SÍ. Re-probar lsgd solo tras NCCL fix.
5. RESUMEN DE OLAS: el slurm lanza olas SIGUSR1@300 + AUTO_RESUBMIT; el
   trainer salva checkpoint y hace resume (fix 2026-08-29: fallback al
   último checkpoint íntegro + retener 2). No borrar checkpoints a mano.
6. WATCHDOGS (crons en la máquina reumanlab, no tocar):
   - ecoreasoner-bw0-bw4-watchdog (5bd4c38c62e9): lanza bw1 UNA vez cuando
     bw0 termina (guarda /tmp/bw1_launched_at — NO relanza tras crash).
   - ecoreasoner-bw1-alive-watchdog (c551ee60cae4): alerta si
     moe-v4-bw4 / moe-v4-bw-v5 muere sin flag COMPLETE.
   - swarm-watchdog-orquestador, swarm-monitor, ollama-governor-hourly:
     orquestación general — no interferir.
7. FETCH PMC: PMC-ids.csv.gz TIENE header (12 cols); PMCID=col index 8,
   PMID=col index 9. Parquet de abstracts PubMed está VACÍO para 2021+
   (snapshot viejo) — para años recientes usar PMC-ids.csv.gz (cubre hasta
   2026). Fail rate S3 ~25% es normal.
8. GIT: /home/reumanlab/ecoreasoner, rama moe-v4-blackwell. Commit+push tras
   cambios (remoto alrobles/ecoreasoner). Si el push rechaza por divergencia,
   NO force-push: merge (git pull --rebase) resolviendo conflictos.

================================================================================
6. COMANDOS ÚTILES
================================================================================
# Cola + estado
squeue -u a474r867 -h -o "%.10i %.16j %.3t %.8M %.12R"
scontrol show job 27553891 | grep -E "JobState|RunTime|NodeList"

# bw1v5
tail -20 /beegfs/a474r867/ecoreasoner/outputs/bw1v5/train.log
cat /beegfs/a474r867/ecoreasoner/outputs/bw1v5/training_complete.flag 2>/dev/null
grep -c Resumed /beegfs/a474r867/ecoreasoner/outputs/bw1v5/train.log

# fetch
cat /beegfs/a474r867/ecoreasoner/data/fulltext_2526/fulltext_corpus_c*.jsonl | wc -l
ls /beegfs/a474r867/ecoreasoner/data/fulltext_2526/

# lanzar test top-2 (si no está ya en cola)
sbatch /beegfs/a474r867/ecoreasoner/scripts/bw_v5_t2nccl.slurm

# verificar NCCL
apptainer exec --overlay /beegfs/a474r867/ecoreasoner/blackwell-torch.overlay:ro \
  /beegfs/a474r867/ecoreasoner/pytorch-cuda.sif \
  /bb/bwvenv/bin/pip show nvidia-nccl-cu12 | grep Version

================================================================================
7. REGLAS DE GOBIERNO (no romper)
================================================================================
- HPC rule: Hermes = R&D + monitor; NO jobs autónomos salvo este canal
  autorizado (hermes.ecoseek.org) y watchdogs ya programados.
- Teachers (v4-flash, glm-4.7, qwen3.6 35b): NO relanzar jobs — rotación
  cron/SIGUSR1; Hermes solo observa. ollama-governor mantiene el pool.
- Ítems eco-* del benchmark: Angel los diseña él mismo — NO materializar ni
  proponer realinear specs.
- Si un comando es ambiguo o el path no existe: PREGUNTAR, no adivinar.

Estado al cierre del handoff (2026-08-29 13:10 CDT):
- bw1v5: RUNNING 1:07, step ~1100/6000, loss 6.0-6.7, estable.
- fetch2526: 435,849/526,735 (83%), 1 shard vivo.
- t2/t2n: PENDING (GPU pro6000 saturada; se libera al terminar bw1v5).
- ollama-q6000: PD (relanzado por governor; normal).