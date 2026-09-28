#!/bin/bash
#SBATCH --job-name=smoke-fg
#SBATCH --partition=sixhour
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40|a40|q8000|pro6000"
#SBATCH --mem=80G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/smoke_fg_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/smoke_fg_%j.err
# SMOKE del brazo fgmode: 20 steps 1-GPU sobre el stream real + init_from
# desde el EMA g316001 pineado. Verifica: build 32E-finos+shared, warm-start
# parcial, forward/backward, balance_loss, save ckpt. Sin DDP, sin EMA.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=/beegfs/a474r867/ecoreasoner
OUT=/tmp/fgmode_smoke_$SLURM_JOB_ID
apptainer exec --nv "$BASE/pytorch-cuda.sif" python3 \
    "$BASE/scripts/train_mdlm_moe_v2.py" \
    --data dummy --data_cache "$BASE/data/train_ids_v5pdb.npz" \
    --output "$OUT" \
    --hidden 768 --layers 12 --heads 12 \
    --n_experts 32 --ff_mult 4 \
    --expert_k 2 --fine_ff_div 2 --n_shared 1 --shared_ff_div 4 \
    --init_from "$BASE/runs/v5-1b/battery_curve/ckpts/g316001/ema_model.pt" \
    --seq_len 768 --batch_size 1 --grad_accum 1 \
    --lr 2e-4 --warmup 5 --lr_decay none \
    --mask_type random --mask_schedule uniform \
    --mask_schedule_args '{"b_l":0.30,"b_h":0.99}' \
    --ema_decay 0.0 --ckpt_every 20 \
    --max_steps 20
echo "=== SMOKE DONE ==="
ls -la "$OUT"/checkpoint-g20/ 2>/dev/null || true
