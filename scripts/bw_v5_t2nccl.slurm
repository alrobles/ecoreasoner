#!/usr/bin/env bash
#SBATCH --job-name=bw-v5-t2n
#SBATCH --partition=sixhour
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=80G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/bw_v5_t2n_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/bw_v5_t2n_%j.err

# TEST: vanilla 8 exp top-2 + v5_clean con WORKAROUND NCCL (nvidia-nccl-cu12 2.26.2
# buggy en Blackwell DDP -> IMA intermitente; fix documentado pytorch #152302:
# NCCL_CUMEM_HOST_ENABLE=0 unblocks; #152780: actualizar nccl a >2.26.2).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
cd "$BASE"
rm -rf outputs/bw_v5_t2n
apptainer exec --nv --overlay "$OVL" "$SIF" env \
  NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN \
  "$VENV/bin/python" -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29512 \
  "$BASE/scripts/train_mdlm_moe.py" \
  --data_cache "$BASE/data/train_ids_v5_clean.npy" \
  --data "$BASE/data/train_corpus_v5.jsonl" \
  --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
  --n_experts 8 --expert_k 2 --seq_len 768 \
  --batch_size 4 --grad_accum 4 --mask_p 0.15 \
  --lr 2e-4 --warmup 5 --max_steps 60 \
  --output "$BASE/outputs/bw_v5_t2n"
echo "bw-v5-t2n done rc=$?"