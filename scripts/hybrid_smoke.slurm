#!/usr/bin/env bash
#SBATCH --job-name=hybrid-smoke
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:q6000:2
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/hybrid_smoke_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/hybrid_smoke_%j.err
# SMOKE del trainer heterogeneo: 2 ranks Q6000, dataset MINI (4 pairs = 1 batch)
# para validar el fix use_reentrant=False (loss con grad_fn + backward ok).
# ENTORNO: overlay+bwvenv (torch 2.7.1) + venv-extra — el SIF base (torch 2.4.1)
# ya no sirve: venv-extra trae torchvision 0.22.1+cu128 (para el SFT) que es
# incompatible con torch 2.4.1 ("operator torchvision::nms does not exist").
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
export PYTHONPATH="$BASE/venv-extra:${PYTHONPATH:-}"
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay":ro "$BASE/pytorch-cuda.sif" \
  env PYTHONPATH="$BASE/venv-extra:$BASE/scripts:${PYTHONPATH:-}" \
  /bb/bwvenv/bin/python -m torch.distributed.run --nproc_per_node=2 --master_port=29516 \
  "$BASE/scripts/train_hybrid.py" \
  --pairs "$BASE/data/l1/sft_v3_smoke.jsonl" \
  --epochs 1 --global-batch 96 --checkpointing --max-len 1024 \
  --out "$BASE/outputs/hybrid_smoke"
echo "hybrid smoke done rc=$?"