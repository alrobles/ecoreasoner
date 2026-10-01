#!/bin/bash
#SBATCH --job-name=smoke-blk
#SBATCH --partition=sixhour
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40|a40|q8000|pro6000"
#SBATCH --mem=80G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/smoke_blk_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/smoke_blk_%j.err
# SMOKE del brazo E-arq block-diffusion (BD3-LM): ~300 steps 1-GPU sobre el
# stream real + init_from EMA g495501. Verifica: build 16E + block_size=32,
# warm-start integro (mismos tensores), mascara block-causal en forward,
# masking de UN bloque/ej, loss razonable y descendente, save ckpt.
# Gate de diseno: loss < init esperado + decay visible antes de los 30K.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=/beegfs/a474r867/ecoreasoner
OUT=/tmp/blkdiff_smoke_$SLURM_JOB_ID
BLK=${BLK:-32}
STEPS=${STEPS:-300}
apptainer exec --nv "$BASE/pytorch-cuda.sif" python3 \
    "$BASE/scripts/train_mdlm_moe_v2.py" \
    --data dummy --data_cache "$BASE/data/train_ids_v5pdb.npz" \
    --output "$OUT" \
    --hidden 768 --layers 12 --heads 12 \
    --n_experts 16 --ff_mult 4 --expert_k 1 \
    --block_size "$BLK" \
    --init_from "$BASE/runs/v5-1b/battery_curve/ckpts/g495501/ema_model.pt" \
    --seq_len 768 --batch_size 1 --grad_accum 1 \
    --lr 2e-4 --warmup 20 --lr_decay none \
    --mask_type random --mask_schedule uniform \
    --mask_schedule_args '{"b_l":0.30,"b_h":0.99}' \
    --ema_decay 0.0 --ckpt_every 100 \
    --max_steps "$STEPS"
echo "=== SMOKE DONE ==="
ls -la "$OUT"/checkpoint-g"$STEPS"/ 2>/dev/null || true
