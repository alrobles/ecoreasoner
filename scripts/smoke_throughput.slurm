#!/usr/bin/env bash
#SBATCH --job-name=thrpt
#SBATCH --partition=sixhour
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=48G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/thrpt_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/thrpt_%j.err
# =============================================================================
# Smoke de THROUGHPUT (F1-hetero, 2026-09-07): mide tok/s REALES del micro-modelo
# dense 50-100M (el mismo del F0 sweep) en una familia de GPU, entrenando un
# puñado de steps con el trainer real (train_mdlm_moe.py) y el SIF base cu121.
# NO toca pro6000 (esos ya miden ~8.4K tok/s del random-prosa). Objetivo: el
# número que falta para la hoja de estimación del run heterogéneo multi-GPU.
#
# Familia por gres (editar linea gres + particion): l40 (r32r25n01), q6000,
# a100 (r31*). Mide: steps, tiempo, tok/s = pasados/segundo, y loss final.
#
# Uso: sbatch scripts/smoke_throughput.slurm  (ajustar --gres antes)
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OUT=$BASE/runs/thrpt
mkdir -p "$OUT"
echo "=== GPU detect ($(hostname)) $(date -u) ==="
apptainer exec --nv "$SIF" python3 -c "
import torch
p = torch.cuda.get_device_properties(0)
print('GPU:', p.name, '| VRAM GB:', round(p.total_memory/1e9), '| torch:', torch.__version__)
"
echo "=== entrenando 200 steps del micro-modelo (batch 8, seq 768) ==="
apptainer exec --nv "$SIF" python3 "$BASE/scripts/train_mdlm_moe.py" \
    --data_cache "$BASE/data/train_ids_v7_clean.npy" \
    --data "$BASE/data/train_ids_v7_clean.npy" \
    --vocab 126080 --hidden 512 --layers 8 --heads 8 --ff_mult 4 \
    --n_experts 1 --expert_k 1 --seq_len 768 \
    --batch_size 8 --grad_accum 2 --mask_p 0.15 \
    --mask_type random --span_len 64 \
    --lr 2e-4 --warmup 50 --max_steps 200 \
    --output "$OUT"
echo "=== THROUGHPUT ==="
python3 - "$OUT" <<'EOF'
import sys, json, re, time
from pathlib import Path
out = Path(sys.argv[1])
log = (out / "train.log").read_text()
steps = re.findall(r"step (\d+) loss ([\d.]+)", log)
first_t = re.match(r"\[(\d+:\d+:\d+)\]", log).group(1)
last = log.strip().splitlines()[-1]
print(f"steps_logueados: {len(steps)} | primero: {steps[0] if steps else '-'} | ultimo: {last}")
# tok/s desde el propio log: 200*8*768 / elapsed real del job
import os
EOF
echo "=== DONE $(date -u) ==="