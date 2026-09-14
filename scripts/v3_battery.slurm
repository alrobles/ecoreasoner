#!/bin/bash
#SBATCH --job-name=v3-battery
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/v3_battery_eval.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/v3_battery_eval.err

BASE=/beegfs/a474r867/ecoreasoner
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay:ro" "$BASE/pytorch-cuda.sif" \
    /bb/bwvenv/bin/python "$BASE/harness/suite_smoke_v2.py" \
    --ckpt "$BASE/runs/f0-span-v3-contrastive/checkpoint-g-final/model.pt" \
    --config "$BASE/harness/configs/f0-span-v3-contrastive.yaml" \
    --pairs-dir "$BASE/runs/pairs_hard_v3_eval" \
    --out-dir "$BASE/runs/f0-span-v3-contrastive/battery" \
    --device cuda
