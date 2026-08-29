#!/usr/bin/env bash
#SBATCH --job-name=bw-v5-t2
#SBATCH --partition=sixhour
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=80G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/bw_v5_t2_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/bw_v5_t2_%j.err

# TEST: trainer VANILLA (el unico estable en DDP world=2) con 8 exp top-2 + v5_clean.
# bw1v5 (4 exp top-1) corre estable. Si vanilla top-2 world=2 pasa -> run de mayor
# capacidad estable. Todos los intentos top-2 previos fueron con lsgd (IMA en DDP).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
cd "$BASE"
rm -rf outputs/bw_v5_t2
apptainer exec --nv --overlay "$OVL":ro "$SIF" \
    "$VENV/bin/python" -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29512 \
    "$BASE/scripts/train_mdlm_moe.py" \
    --data_cache "$BASE/data/train_ids_v5_clean.npy" \
    --data "$BASE/data/train_corpus_v5.jsonl" \
    --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
    --n_experts 8 --expert_k 2 --seq_len 768 \
    --batch_size 4 --grad_accum 4 --mask_p 0.15 \
    --lr 2e-4 --warmup 5 --max_steps 30 \
    --output "$BASE/outputs/bw_v5_t2"
echo "bw-v5-t2 done rc=$?"