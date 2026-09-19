#!/usr/bin/env bash
# g11_launch.sh — Generacion 11 del GA: hneg-data (ultimo gen de datos).
#
# Diseno: docs/designs/EVOG9-V5-DESIGN.md §4 — corpus += pares knn_edges
# verificados (pair-docs: dos esqueletos near-miss en una ventana; al
# enmascarar un valor, la otra claim queda visible -> binding). Si G11
# tampoco mueve el holdout -> cerrar GA como negativo-controlado.
#
# Backbone: runs/g10-ep3-s1/checkpoint-g23000 — campeon G10 (dev L3 0.650,
# holdout L3 0.620/num 0.394/neg 0.703). EMA_DECAY=0.999 + EVAL_EMA=1 ON.
#
# Brazos (continuacion +3K -> 26000, LR 1e-4 flat):
#   g11-hneg-s1  DATA_CACHE=train_ids_v4hneg.npz (corpus + pair-docs)
#   g11-hneg-s2  replica seed (decision de cierre GA -> 2 seeds)
#   g11-ep4-s1   control +steps sobre corpus base (DATA_CACHE=v4en)
#
# Fitness: 0.5*L3 + 0.5*min(num,neg) dev v3_eval; holdout SOLO al campeon.
# Sembrado: symlink a checkpoint-g23000 (retencion-2 no sigue symlinks).
#
# Uso: bash g11_launch.sh [--dry-run]
set -e
cd /beegfs/a474r867/ecoreasoner/scripts
BASE=/beegfs/a474r867/ecoreasoner
DATA=$BASE/data
RUNS=$BASE/runs
BACKBONE=$RUNS/g10-ep3-s1/checkpoint-g23000
DRY=${1:-}

GEOM="VOCAB=126080 HIDDEN=768 LAYERS=12 HEADS=12 FF_MULT=4 \
      N_EXPERTS=8 EXPERT_K=1 SEQ_LEN=768"
RECETA="MASK_TYPE=random MASK_SCHEDULE=uniform \
      MASK_SCHEDULE_ARGS={\"b_l\":0.30,\"b_h\":0.99} \
      CURRICULUM=1 WHOLE_STAGE=0 ROLE_MASK=0 CANDIDATE_FOCUS=1.0 \
      BATCH_SIZE=4 GRAD_ACCUM=4 MIN_VRAM_MIB=40000 \
      EVAL_CFG=$BASE/harness/configs/eval-moe-v4.yaml"
CONT="LR=1e-4 LR_DECAY=none TARGET_STEPS=26000 \
      EMA_DECAY=0.999 EVAL_EMA=1"

seed_out() {  # $1=tag -> OUT sembrado con backbone en step 23000
    local out=$RUNS/$1
    mkdir -p "$out"
    ln -sfn "$BACKBONE" "$out/checkpoint-g23000"
    printf '{"step": 23000, "checkpoint": "checkpoint-g23000", "updated": %s}\n' \
        "$(date +%s)" > "$out/state.json"
    echo "[seed] $out -> step 23000"
}

run() {  # $1=tag $2=seed $3=cache $4...=extra env
    local tag=$1 seed=$2 cache=$3; shift 3
    echo "[g11] $tag seed=$seed cache=$cache extra='$*'"
    [ "$DRY" = "--dry-run" ] && return 0
    env TAG="$tag" SEED="$seed" DATA_CACHE="$cache" $GEOM $RECETA $CONT "$@" \
        sbatch --export=ALL --job-name="$tag" g0_run.slurm
}

for tag in g11-hneg-s1 g11-hneg-s2 g11-ep4-s1; do
    seed_out "$tag"
done
run g11-hneg-s1 1 "$DATA/train_ids_v4hneg.npz"
run g11-hneg-s2 2 "$DATA/train_ids_v4hneg.npz"
run g11-ep4-s1  1 "$DATA/train_ids_v4en.npz"

echo "G11 lanzada: 3 jobs (hneg x2 seeds + control ep4 sobre corpus base)."
