#!/usr/bin/env bash
#SBATCH --job-name=smoke-bw3
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/smoke_bw3_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/smoke_bw3_%j.err

# Smoke L1 contra el checkpoint MAS NUEVO de bw3 (training base v7 en curso).
# Usa la GPU libre (slurm asigna: si el training ocupa 2 GPUs de un nodo,
# cae en el otro nodo Blackwell o en r23r09n01). Parametrizado por CKPT_STEP.
# El trainer conserva 2 checkpoints (cada 50 steps) -> el paso lo elige el cron.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
OVL="$BASE/blackwell-torch.overlay"
SIF="$BASE/pytorch-cuda.sif"
# Resolver el ckpt a usar: si no se paso CKPT_STEP, o el indicado ya NO
# tiene model.pt (el trainer retiene solo 2 -> un ckpt elegido por el cron
# puede ser purgado mientras el job espera en cola saturada), cae al mas
# nuevo con model.pt INTEGRO.
CKPT_STEP="${CKPT_STEP:-latest}"
resolve_latest() {
    for d in $(ls -d "$BASE/outputs/bw3/checkpoint-g"* 2>/dev/null | sort -V); do
        [ -f "$d/model.pt" ] && basename "$d" | sed 's/checkpoint-g//'
    done | tail -1
}
if [ "$CKPT_STEP" = "latest" ] || [ ! -f "$BASE/outputs/bw3/checkpoint-g$CKPT_STEP/model.pt" ]; then
    NEW=$(resolve_latest)
    [ -n "$NEW" ] || { echo "sin ckpt valido con model.pt"; exit 1; }
    echo "==> ckpt g$CKPT_STEP no disponible (purgado?), usando g$NEW"
    CKPT_STEP="$NEW"
fi
echo "=== smoke-bw3 contra checkpoint-g$CKPT_STEP $(date -u) ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python \
  "$BASE/scripts/smoke_l1.py" \
  --ckpt "$BASE/outputs/bw3/checkpoint-g$CKPT_STEP" \
  --corpus "$BASE/data/l1/train_corpus_l1.jsonl" \
  --n 40 --n-gen 120 --steps 64 \
  --out "$BASE/data/l1/smoke_bw3_g${CKPT_STEP}.json"
echo "smoke-bw3 done rc=$?"