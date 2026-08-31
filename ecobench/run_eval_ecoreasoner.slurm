#!/usr/bin/env bash
#SBATCH --job-name=eval-ecoreasoner
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/eval_ecoreasoner_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/eval_ecoreasoner_%j.err
# EcoBench-EVAL con EcoReasoner local (LLaDA-MoE + LoRA SFT v2) como agente.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE/ecobench"
export PYTHONPATH="$BASE/venv-extra:${PYTHONPATH:-}"
export PYTHONPATH="$BASE/ecobench:${PYTHONPATH:-}"
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay":ro "$BASE/pytorch-cuda.sif" \
  /bb/bwvenv/bin/python "$BASE/ecobench/run_eval_ecoreasoner.py" \
  --lora "$BASE/outputs/sft_moe_v2/lora-final/lora.pt" \
  --model-dir "$BASE/models/LLaDA-MoE-7B-A1B-Instruct" \
  --gen-length 512 --timeout-per-item 300
echo "eval-ecoreasoner done rc=$?"