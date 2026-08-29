#!/usr/bin/env bash
#SBATCH --job-name=bw1-top1
#SBATCH --partition=sixhour
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/bw1_t1_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/bw1_t1_%j.err

# AISLA el top-2: DDP world=2 + 4 exp top-1 (igual config a bw0 pero con lsgd + v5_clean).
# Si pasa -> el IMA es especifico de top-2 en DDP. Si falla -> es DDP+lsgd general.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF="$BASE/pytorch-cuda.sif"
OVL="$BASE/blackwell-torch.overlay"
VENV=/bb/bwvenv
cd "$BASE"
rm -rf outputs/bw1_t1
apptainer exec --nv --overlay "$OVL":ro "$SIF" env PYTHONDONTWRITEBYTECODE=1 \
  "$VENV/bin/python" -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_id=bw1-t1 \
  "$BASE/scripts/train_mdlm_moe_lsgd.py" \
  --data_cache "$BASE/data/train_ids_v5_clean.npy" \
  --data "$BASE/data/train_corpus_v5.jsonl" \
  --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
  --n_experts 4 --expert_k 1 --seq_len 768 \
  --batch_size 4 --grad_accum 4 --mask_p 0.15 \
  --sync_every 1 --clip_grad 1.0 \
  --lr 2e-4 --warmup 5 --max_steps 20 \
  --output "$BASE/outputs/bw1_t1"
echo "bw1-t1 done rc=$?"
