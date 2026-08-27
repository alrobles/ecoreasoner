#!/bin/bash
#SBATCH --job-name=mdlm-retrain-v3
#SBATCH --partition=sixhour
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:q6000:4
#SBATCH --mem=150G
#SBATCH --time=05:50:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/mdlm_retrain_v3_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/mdlm_retrain_v3_%j.err

# =============================================================================
# FASE A — Refinar base dLLM-MoE sobre train_corpus_v3 (continua desde retrain1)
#   Punto de partida: checkpoint-g5000 de outputs/retrain1 (base ya entrenado
#   sobre train_corpus_v2 ~1.7B tok, 676M/279M activos MoE top-1 8 exp).
#   Este job: copia el base a retrain_v3/ y continua sobre train_corpus_v3.jsonl
#   (v2 + 1724 docs EcoEvoRxiv). 2 nodos x 4 Q6000 DDP = 8 ranks.
#   SIGUSR1 300s = checkpoint + exit 42 = relanza.
# Uso: sbatch --export=TARGET_STEPS=8000 mdlm_retrain_v3.slurm
# =============================================================================
set -euo pipefail

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
SCRIPTS=$BASE/scripts
DATA_DIR=$BASE/data
DATA="$DATA_DIR/train_corpus_v3.jsonl"
BASE_OUT="$BASE/outputs/retrain1"     # base ya entrenado
WAVE_ID="${WAVE_ID:-retrain_v3}"
OUT="${WAVE_OUT:-$BASE/outputs/$WAVE_ID}"
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
TARGET_STEPS="${TARGET_STEPS:-8000}"

mkdir -p "$OUT" "$BASE/logs"

# --- clonar base si el OUT todavia no tiene checkpoint (arranque desde retrain1 g5000)
if [ ! -f "$OUT/state.json" ]; then
    echo "[$(date)] Clonando base desde $BASE_OUT/..." 
    src_ck=$(cat "$BASE_OUT/state.json" | python3 -c "import sys,json;print(json.load(sys.stdin)['checkpoint'])")
    echo "  base checkpoint: $src_ck (step $(cat "$BASE_OUT/state.json" | python3 -c "import sys,json;print(json.load(sys.stdin)['step'])"))"
    cp "$BASE_OUT/state.json" "$OUT/state.json"
    cp -r "$BASE_OUT/$src_ck" "$OUT/$src_ck"
    echo "  clonado -> $OUT/$src_ck"
else
    echo "  $OUT ya tiene state.json (resume propio/ola previa)"
fi

echo "[$(date)] FASE A dLLM-MoE $WAVE_ID (8x Q6000 DDP, DATA=v3, TARGET_STEPS=$TARGET_STEPS)"

FINALIZED=0
finalize() {
    [ "$FINALIZED" = "1" ] && return
    FINALIZED=1
    if [ -f "$FLAG" ]; then echo "complete"; cat "$FLAG"; exit 0; fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        echo "relanzando ola $WAVE_ID..."
        sbatch --export="AUTO_RESUBMIT=1,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT,TARGET_STEPS=$TARGET_STEPS" \
            "$BASE/scripts/mdlm_retrain_v3.slurm" 2>/dev/null
    fi
}
handle_signal() {
    echo "=== SIGUSR1 $(date -u) — guardando ola ==="
    if [ -n "${SRUN_PID:-}" ] && kill -0 "$SRUN_PID" 2>/dev/null; then
        kill -USR1 "$SRUN_PID" 2>/dev/null || true
        set +e; wait "$SRUN_PID"; tx=$?; set -e
    fi
    finalize
}
trap handle_signal USR1
trap 'echo SIGTERM; exit 1' TERM

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29514
export OMPI_MCA_btl=^openib

# run 8 ranks (2 nodos x 4). SLURM_PROCID/LOCALID set by srun.
# --export=ALL propaga MASTER_ADDR/MASTER_PORT a TODOS los nodos (crítico DDP multi-nodo)
srun --export=ALL -n 8 /usr/bin/apptainer exec --nv \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$SIF" python3 "$SCRIPTS/train_mdlm_moe.py" \
        --data $DATA --data_cache "$DATA_DIR/train_ids_v3.npy" --output "$OUT" \
        --hidden 768 --layers 12 --heads 12 \
        --n_experts 8 --expert_k 1 --ff_mult 4 \
        --seq_len 768 --batch_size 2 --grad_accum 2 \
        --lr 2e-4 --warmup 200 --mask_p 0.15 \
        --max_steps "$TARGET_STEPS" &
SRUN_PID=$!
set +e; wait "$SRUN_PID"; TRAIN_X=$?; set -e
echo "=== srun exit $TRAIN_X ==="
case "$TRAIN_X" in
    0) echo "COMPLETE" > "$FLAG"; exit 0;;    # alcanzo max_steps
    42) finalize;;                            # ola (SIGUSR1)
    *) if [ "${SIGNAL_RECEIVED:-0}" = "1" ]; then finalize; else exit "$TRAIN_X"; fi;;
esac