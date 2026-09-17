#!/usr/bin/env bash
# g9_launch.sh — Generacion 9 del GA: mutantes sobre backbone retrain_v4s.
#
# Diseno: docs/designs/EVOG9-V5-DESIGN.md §3.
# Backbone: runs/retrain_v4s/checkpoint-g17000 (MoE 676M, L3 holdout 0.600).
#
# Brazos continuacion (+3K -> 20000, LR 1e-4 flat, receta v4s intacta):
#   g9-corrlo-s1/s2  CORRECTIVE_P=0.02   (2 seeds: p=0.10 fue bimodal)
#   g9-corrmd-s1     CORRECTIVE_P=0.05
#   g9-numwl-s1      LOSS_NUM_W=1.5 LOSS_NEG_W=1.5
#   g9-ep2-s1        control: +steps sin gen nuevo
#   g9-ema-s1        EMA_DECAY=0.999, eval sobre ema_model.pt (EVAL_EMA=1)
#
# Brazos fresh (10K desde cero, curriculum completo, LR 2e-4 cosine):
#   g9-scratch-s1/s2  linaje limpio — ¿techa el pre-historial flat de v4s?
#
# Fitness: media-seeds 0.5*L3 + 0.5*min(num,neg) en dev v3_eval (PAIRS_DIR
# default de g0_run.slurm). holdout_clean SOLO al campeon de la generacion.
#
# Sembrado: cada OUT de continuacion recibe state.json(step=17000) +
# symlink a checkpoint-g17000 (rmtree de retencion-2 no sigue symlinks:
# falla silencioso, el backbone queda intacto).
#
# Uso: bash g9_launch.sh            (desde el cluster, scripts/)
#      bash g9_launch.sh --dry-run  (imprime sin lanzar)
set -e
cd /beegfs/a474r867/ecoreasoner/scripts
BASE=/beegfs/a474r867/ecoreasoner
DATA=$BASE/data
RUNS=$BASE/runs
BACKBONE=$RUNS/retrain_v4s/checkpoint-g17000
DRY=${1:-}

GEOM="VOCAB=126080 HIDDEN=768 LAYERS=12 HEADS=12 FF_MULT=4 \
      N_EXPERTS=8 EXPERT_K=1 SEQ_LEN=768"
RECETA="MASK_TYPE=random MASK_SCHEDULE=uniform \
      MASK_SCHEDULE_ARGS={\"b_l\":0.30,\"b_h\":0.99} \
      CURRICULUM=1 WHOLE_STAGE=0 ROLE_MASK=0 CANDIDATE_FOCUS=1.0 \
      BATCH_SIZE=4 GRAD_ACCUM=4 MIN_VRAM_MIB=40000 \
      DATA_CACHE=$DATA/train_ids_v4en.npz \
      EVAL_CFG=$BASE/harness/configs/eval-moe-v4.yaml"
CONT="LR=1e-4 LR_DECAY=none TARGET_STEPS=20000"
FRESH="LR=2e-4 LR_DECAY=cosine TARGET_STEPS=10000"

seed_out() {  # $1=tag -> OUT sembrado con backbone en step 17000
    local out=$RUNS/$1
    mkdir -p "$out"
    ln -sfn "$BACKBONE" "$out/checkpoint-g17000"
    printf '{"step": 17000, "checkpoint": "checkpoint-g17000", "updated": %s}\n' \
        "$(date +%s)" > "$out/state.json"
    echo "[seed] $out -> step 17000"
}

run() {  # $1=tag $2=seed $3...=extra env
    local tag=$1 seed=$2; shift 2
    echo "[g9] $tag seed=$seed extra='$*'"
    [ "$DRY" = "--dry-run" ] && return 0
    env TAG="$tag" SEED="$seed" $GEOM $RECETA "$@" \
        sbatch --export=ALL --job-name="$tag" g0_run.slurm
}

# --- continuacion (backbone g17000) ---
for tag in g9-corrlo-s1 g9-corrlo-s2 g9-corrmd-s1 g9-numwl-s1 g9-ep2-s1 g9-ema-s1; do
    seed_out "$tag"
done
run g9-corrlo-s1 1 $CONT CORRECTIVE_P=0.02
run g9-corrlo-s2 2 $CONT CORRECTIVE_P=0.02
run g9-corrmd-s1 1 $CONT CORRECTIVE_P=0.05
run g9-numwl-s1  1 $CONT LOSS_NUM_W=1.5 LOSS_NEG_W=1.5
run g9-ep2-s1    1 $CONT
run g9-ema-s1    1 $CONT EMA_DECAY=0.999 EVAL_EMA=1

# --- fresh (sin semilla: OUT limpio, step 0) ---
run g9-scratch-s1 1 $FRESH
run g9-scratch-s2 2 $FRESH

echo "G9 lanzada: 8 jobs / 7 brazos (corrlo x2 por bimodalidad de corrective)."
