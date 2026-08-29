#!/bin/bash
#SBATCH --job-name=v100-smoke
#SBATCH --partition=sixhour
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/v100_smoke_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/v100_smoke_%j.err
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OUT=$BASE/outputs/v100_smoke
mkdir -p "$OUT"
export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29516
export NCCL_DEBUG=INFO
echo "=== v100 smoke: nodes=$SLURM_JOB_NODELIST tasks=$SLURM_NTASKS master=$MASTER_ADDR ==="
srun --export=ALL --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
  apptainer exec --nv "$SIF" python3 "$BASE/scripts/train_mdlm_moe_lsgd.py" \
  --data "$BASE/data/sci_v2_B1_consolidated.jsonl" \
  --data_cache "$BASE/data/train_ids_v5.npy" \
  --hidden 768 --layers 12 --heads 12 --n_experts 4 --expert_k 1 --ff_mult 4 \
  --seq_len 512 --batch_size 4 --grad_accum 4 \
  --sync_every 4 --clip_grad 1.0 \
  --max_steps 40 --output "$OUT"
rc=$?
echo "=== smoke rc=$rc ==="
exit $rc