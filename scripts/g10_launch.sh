#!/usr/bin/env bash
# g10_launch.sh — Generacion 10 del GA: hinge contrastivo sobre campeon G9.
#
# Diseno: docs/designs/EVOG9-V5-DESIGN.md §4 (post-audit 18-09).
# Backbone: runs/g9-ema-s1/checkpoint-g20000 — campeon G9 (dev L3 0.642,
# holdout L3 0.619/num 0.394/neg 0.613). Todos los brazos heredan
# EMA_DECAY=0.999 + EVAL_EMA=1 (el gen ganador queda ON).
#
# Clase mutable ya expandida (commit d5306c0): is_numx cubre piezas
# numericas (~98% cov number), flip-table 124 claves (direction->57%,
# causal_word->58%, temporal->77-80%, negation->100% medido holdout).
#
# Brazos (continuacion +3K -> 23000, LR 1e-4 flat):
#   g10-contr-s1  CONTR_W=0.3 CONTR_MARGIN=2.0
#   g10-contr-s2  CONTR_W=1.0 CONTR_MARGIN=2.0
#   g10-contr-s3  CONTR_W=0.3 CONTR_MARGIN=1.0
#   g10-contr-s4  CONTR_W=1.0 CONTR_MARGIN=1.0
#   g10-ep3-s1    control +steps (CONTR_W=0), aísla el efecto del hinge
#
# Fitness: 0.5*L3 + 0.5*min(num,neg) dev v3_eval; holdout SOLO al campeon.
# Sembrado: symlink a checkpoint-g20000 (retencion-2 no sigue symlinks).
#
# Uso: bash g10_launch.sh [--dry-run]
set -e
cd /beegfs/a474r867/ecoreasoner/scripts
BASE=/beegfs/a474r867/ecoreasoner
DATA=$BASE/data
RUNS=$BASE/runs
BACKBONE=$RUNS/g9-ema-s1/checkpoint-g20000
DRY=${1:-}

GEOM="VOCAB=126080 HIDDEN=768 LAYERS=12 HEADS=12 FF_MULT=4 \
      N_EXPERTS=8 EXPERT_K=1 SEQ_LEN=768"
RECETA="MASK_TYPE=random MASK_SCHEDULE=uniform \
      MASK_SCHEDULE_ARGS={\"b_l\":0.30,\"b_h\":0.99} \
      CURRICULUM=1 WHOLE_STAGE=0 ROLE_MASK=0 CANDIDATE_FOCUS=1.0 \
      BATCH_SIZE=4 GRAD_ACCUM=4 MIN_VRAM_MIB=40000 \
      DATA_CACHE=$DATA/train_ids_v4en.npz \
      EVAL_CFG=$BASE/harness/configs/eval-moe-v4.yaml"
CONT="LR=1e-4 LR_DECAY=none TARGET_STEPS=23000 \
      EMA_DECAY=0.999 EVAL_EMA=1"

seed_out() {  # $1=tag -> OUT sembrado con backbone en step 20000
    local out=$RUNS/$1
    mkdir -p "$out"
    ln -sfn "$BACKBONE" "$out/checkpoint-g20000"
    printf '{"step": 20000, "checkpoint": "checkpoint-g20000", "updated": %s}\n' \
        "$(date +%s)" > "$out/state.json"
    echo "[seed] $out -> step 20000"
}

run() {  # $1=tag $2=seed $3...=extra env
    local tag=$1 seed=$2; shift 2
    echo "[g10] $tag seed=$seed extra='$*'"
    [ "$DRY" = "--dry-run" ] && return 0
    env TAG="$tag" SEED="$seed" $GEOM $RECETA "$@" \
        sbatch --export=ALL --job-name="$tag" g0_run.slurm
}

for tag in g10-contr-s1 g10-contr-s2 g10-contr-s3 g10-contr-s4 g10-ep3-s1; do
    seed_out "$tag"
done
run g10-contr-s1 1 $CONT CONTR_W=0.3 CONTR_MARGIN=2.0
run g10-contr-s2 1 $CONT CONTR_W=1.0 CONTR_MARGIN=2.0
run g10-contr-s3 1 $CONT CONTR_W=0.3 CONTR_MARGIN=1.0
run g10-contr-s4 1 $CONT CONTR_W=1.0 CONTR_MARGIN=1.0
run g10-ep3-s1   1 $CONT

echo "G10 lanzada: 5 jobs (2x2 sweep contr_w x margin + control +steps)."
