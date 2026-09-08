#!/usr/bin/env bash
#SBATCH --job-name=eval-curve-watch
#SBATCH --partition=sixhour
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/eval_curve_watch_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/eval_curve_watch_%j.err
# =============================================================================
# eval_curve_watch.sh — orquestador de la curva de discriminación del F1.
# Cada vez que aparece un checkpoint-g<N> nuevo en el run-dir lanza un
# eval_curve corto (pro6000, --no-gen, ~1 min/ckpt); al detectar el flag
# COMPLETE lanza el eval final (cubre los últimos ckpts) y termina.
# Los rows se APPENDEN a <run-dir>/eval_curve.jsonl (dups por step: dedupe
# al final, quedarse con el último).
# =============================================================================
set -uo pipefail

BASE=/beegfs/a474r867/ecoreasoner
RUN=$BASE/runs/f1-hetero
MARK=$RUN/.eval_curve_last_step
FLAG=$RUN/training_complete.flag
EVAL_SLURM=$BASE/scripts/eval_curve.slurm
# SIN comillas en el --export (word-splitting de $EVAL_CMD no reinterpreta
# comillas; los valores no tienen espacios).
EVAL_CMD="sbatch --parsable --gres=gpu:pro6000:1 --time=00:30:00 \
  --export=RUN_DIR=$RUN,PAIRS=$BASE/runs/pairs.jsonl,CONFIG=$BASE/harness/configs/f0-span-esqueleto.yaml,EVERY=1,MIN_STEP=500,EXTRA=--no-gen \
  $EVAL_SLURM"

last=0
[ -f "$MARK" ] && last=$(cat "$MARK" 2>/dev/null | tr -dc '0-9')
echo "[watch] start last=$last $(date -u)"

while true; do
    # step máximo de los checkpoints en disco (retention-2: solo 2 últimos)
    cur=$(ls -d "$RUN"/checkpoint-g* 2>/dev/null | grep -oE 'g[0-9]+$' | tr -d g | sort -n | tail -1)
    cur=${cur:-0}

    if [ "$cur" -gt "$last" ]; then
        echo "[watch] ckpt nuevo g$cur (last=$last) -> eval $(date -u)"
        $EVAL_CMD
        last=$cur
        echo "$last" > "$MARK"
    fi

    if [ -f "$FLAG" ] && grep -q COMPLETE "$FLAG" 2>/dev/null; then
        echo "[watch] COMPLETE -> eval final $(date -u)"
        $EVAL_CMD
        echo "[watch] done $(date -u)"
        exit 0
    fi

    sleep 600
done