#!/usr/bin/env bash
# v4s_launch.sh — PORT de la receta campeon hinrcf10 al MoE v4.
#
# Pregunta (resultado v4, 2026-09-17): el retrain_v4 flat (mask_p=0.15) gano
# L0/L1 pero cayo a azar en L2/L3 (0.503/0.496). ¿La receta estructurada
# (random+hi+nr+cf10) recupera la frontera inferencial sobre el MoE+v4en
# conservando las ganancias superficiales?
#
# Brazo: continua outputs/retrain_v4/checkpoint-g11000 (sembrado en
# runs/retrain_v4s), +6K steps (17000) = ~74M tok, misma exposicion v4en
# que el brazo flat (3K x 24.6K tok). LR constante 1e-4 (recipe shift,
# sin cosine que lo apague al final de la ventana).
#
# WHOLE_STAGE=0 deliberado: con whole_stage=1 los docs SIN etiquetas de
# etapa (prosa UNAM ~16% de v4en) caen al fallback span64 (gen perdedor);
# con 0 reciben random-denso, fiel a hi. CF=1.0 sigue enmascarando la
# etapa candidata al 100% en docs esqueleto.
#
# Eval automatica (g0_run.slurm): battery dense+base sobre
# pairs_hard_v4_holdout_clean con eval-moe-v4.yaml — comparacion directa
# con retrain_v4 (L3 0.496) e hinrcf10 (L3 0.632).
#
# Uso: bash v4s_launch.sh   (desde el cluster, scripts/)
set -e
cd /beegfs/a474r867/ecoreasoner/scripts
BASE=/beegfs/a474r867/ecoreasoner
DATA=$BASE/data
RUNS=$BASE/runs

env TAG=retrain_v4s \
    DATA_CACHE=$DATA/train_ids_v4en.npz \
    VOCAB=126080 HIDDEN=768 LAYERS=12 HEADS=12 FF_MULT=4 \
    N_EXPERTS=8 EXPERT_K=1 SEQ_LEN=768 \
    MASK_TYPE=random MASK_SCHEDULE=uniform \
    MASK_SCHEDULE_ARGS='{"b_l":0.30,"b_h":0.99}' \
    CURRICULUM=1 WHOLE_STAGE=0 ROLE_MASK=0 CANDIDATE_FOCUS=1.0 \
    BATCH_SIZE=4 GRAD_ACCUM=4 MIN_VRAM_MIB=40000 \
    LR=1e-4 LR_DECAY=none TARGET_STEPS=17000 SEED=1 \
    PAIRS_DIR=$RUNS/pairs_hard_v4_holdout_clean \
    EVAL_CFG=$BASE/harness/configs/eval-moe-v4.yaml \
    sbatch --export=ALL --job-name=retrain_v4s g0_run.slurm
