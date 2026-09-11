#!/usr/bin/env bash
# launch_v3.2_role.sh — lanza V3.2 role-aware + curriculum sobre train_mdlm_moe_v2.py.
# Uso: ./scripts/launch_v3.2_role.sh
set -euo pipefail

BASE=/beegfs/a474r867/ecoreasoner
ENV=/tmp/v3.2_role_env.sh

cat > "$ENV" <<'EOF'
TAG=f0-span-v3-role
DATA_CACHE=/beegfs/a474r867/ecoreasoner/data/train_ids_skeleton_v2.npz
CONFIG=/beegfs/a474r867/ecoreasoner/harness/configs/f0-span-v3-role.yaml
TARGET_STEPS=10000
PAIRS_DIR=/beegfs/a474r867/ecoreasoner/runs/pairs_hard_v3
PAIRS=/beegfs/a474r867/ecoreasoner/runs/pairs_hard_v3/pairs_L0.jsonl
MASK_TYPE=span
MASK_SCHEDULE=uniform
MASK_SCHEDULE_ARGS={"b_l":0.05,"b_h":0.95}
SPAN_LEN=64
WHOLE_STAGE=1
CURRICULUM=1
CUR_STAGES=[[0,2000,16,0.30],[2000,5000,32,0.60],[5000,10000,64,0.95]]
ROLE_MASK=1
ROLE_CONFIG={"connectives":["because","despite","however","therefore","although","thus","since","while","whereas","yet","but","so","if"],"relation_verbs":["increases","reduces","decreases","inhibits","promotes","correlates","depends","drives","affects","influences","regulates","enhances","suppresses","causes","leads","results","associated","linked","modulates","triggers"],"species_keywords":["panthera","quercus","apis","daphnia","mytilus","giraffa","cervus","populus","saguaro","tigris","ilex","suber"],"number_weight":2.0,"connective_weight":3.0,"relation_weight":3.0,"entity_weight":2.0,"uppercase_bonus":0.5}
LR=2.0e-4
LR_DECAY=cosine
LR_MIN_RATIO=0.1
WARMUP=200
GRAD_CLIP=1.0
EMA_DECAY=0.0
WEIGHT_TYING=0
USE_ROPE=0
BATCH_SIZE=8
GRAD_ACCUM=2
VOCAB=126080
HIDDEN=512
LAYERS=8
HEADS=8
FF_MULT=4
N_EXPERTS=1
EXPERT_K=1
SEQ_LEN=768
EOF

cp "$ENV" "$BASE/scripts/.v3.2_role_env.sh"

JOB=$(sbatch --parsable --export=ALL,FILE="$BASE/scripts/.v3.2_role_env.sh" "$BASE/scripts/moe_v4_micro_v2.slurm")
echo "V3.2 role-aware lanzado: job $JOB"
echo "env guardado en $BASE/scripts/.v3.2_role_env.sh"
