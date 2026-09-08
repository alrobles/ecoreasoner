#!/usr/bin/env bash
#SBATCH --job-name=eval-curve-watch-f2
#SBATCH --partition=sixhour
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/eval_curve_watch_f2_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/eval_curve_watch_f2_%j.err
# =============================================================================
# eval_curve_watch_f2.sh — orquestador curva del puente f2-spanes.
# Igual que eval_curve_watch.sh pero apuntando a runs/f2-spanes. El micro
# trainer SÍ escribe training_complete.flag → detección de fin por flag.
# =============================================================================
set -uo pipefail

BASE=/beegfs/a474r867/ecoreasoner
RUN=$BASE/runs/f2-spanes
MARK=$RUN/.eval_curve_last_step
FLAG=$RUN/training_complete.flag
EVAL_SLURM=$BASE/scripts/eval_curve.slurm
EVAL_CMD="sbatch --parsable --gres=gpu:pro6000:1 --time=00:30:00 \
  --export=RUN_DIR=$RUN,PAIRS=$BASE/runs/pairs.jsonl,CONFIG=$BASE/harness/configs/f0-span-esqueleto.yaml,EVERY=1,MIN_STEP=500,EXTRA=--no-gen \
  $EVAL_SLURM"

last=0
[ -f "$MARK" ] && last=$(cat "$MARK" 2>/dev/null | tr -dc '0-9')
echo "[watch-f2] start last=$last $(date -u)"

while true; do
    cur=$(ls -d "$RUN"/checkpoint-g* 2>/dev/null | grep -oE 'g[0-9]+$' | tr -d g | sort -n | tail -1)
    cur=${cur:-0}

    if [ "$cur" -gt "$last" ]; then
        echo "[watch-f2] ckpt nuevo g$cur (last=$last) -> eval $(date -u)"
        $EVAL_CMD
        last=$cur
        echo "$last" > "$MARK"
    fi

    if [ -f "$FLAG" ] && grep -q COMPLETE "$FLAG" 2>/dev/null; then
        echo "[watch-f2] COMPLETE -> eval final $(date -u)"
        $EVAL_CMD
        echo "[watch-f2] done $(date -u)"
        exit 0
    fi

    sleep 600
done