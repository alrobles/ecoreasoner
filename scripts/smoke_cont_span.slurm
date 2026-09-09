#!/usr/bin/env bash
#SBATCH --job-name=smoke-cont-span
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/smoke_cont_span_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/smoke_cont_span_%j.err

# Smoke de CONTINUACION contra el checkpoint MAS NUEVO de bw4_span (experimento
# span-masking 2026-09-02). El cron lanza este slurm con CKPT_STEP; si el ckpt
# elegido fue purgado (trainer retiene 2), cae al mas nuevo con model.pt.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
OVL="$BASE/blackwell-torch.overlay"
SIF="$BASE/pytorch-cuda.sif"
RUN=bw4_span

CKPT_STEP="${CKPT_STEP:-latest}"
resolve_latest() {
    for d in $(ls -d "$BASE/outputs/$RUN/checkpoint-g"* 2>/dev/null | sort -V); do
        [ -f "$d/model.pt" ] && basename "$d" | sed 's/checkpoint-g//'
    done | tail -1
}
if [ "$CKPT_STEP" = "latest" ] || [ ! -f "$BASE/outputs/$RUN/checkpoint-g$CKPT_STEP/model.pt" ]; then
    NEW=$(resolve_latest)
    [ -n "$NEW" ] || { echo "sin ckpt valido con model.pt"; exit 1; }
    echo "==> ckpt g$CKPT_STEP no disponible, usando g$NEW"
    CKPT_STEP="$NEW"
fi
echo "=== smoke-cont-span contra checkpoint-g$CKPT_STEP $(date -u) ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python \
  "$BASE/scripts/smoke_cont_bw3.py" \
  --ckpt "$BASE/outputs/$RUN/checkpoint-g$CKPT_STEP" \
  --corpus "$BASE/data/curate/train_corpus_v7_curated.jsonl" \
  --n-docs 6 --ctx-lens 0,64,200,400 --n-gen 180 --temps 0.7,0.9 --steps 64,128 \
  --out "$BASE/data/l1/smoke_cont_span_g${CKPT_STEP}.json"
echo "smoke-cont-span done rc=$?"