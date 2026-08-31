#!/usr/bin/env bash
#SBATCH --job-name=hybrid-test
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:q6000:2
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/hybrid_test_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/hybrid_test_%j.err
# HITO b — TEST: 2 ranks en 1 nodo (2x Q6000). Valida autosize + grad_accum
# adaptativo + all-reduce manual de LoRA. (El multi-nodo es refinamiento de red;
# este valida el algoritmo core.)
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
export PYTHONPATH="$BASE/venv-extra:${PYTHONPATH:-}"
apptainer exec --nv "$BASE/pytorch-cuda.sif" env PYTHONPATH="$BASE/scripts:$BASE/venv-extra:${PYTHONPATH:-}" \
  python3 -m torch.distributed.run --nproc_per_node=2 --master_port=29515 \
  "$BASE/scripts/train_hybrid.py" \
  --pairs "$BASE/data/l1/sft_v3_pairs.jsonl" \
  --epochs 1 --global-batch 96 --checkpointing --max-len 1024 \
  --out "$BASE/outputs/hybrid_test"
echo "hybrid test done rc=$?"