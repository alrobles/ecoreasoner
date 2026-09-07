#!/usr/bin/env bash
#SBATCH --job-name=eval-eco-v3
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/eval_ecoreasoner_v3_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/eval_ecoreasoner_v3_%j.err
# EcoBench-EVAL con EcoReasoner local (LLaDA-MoE + LoRA SFT v3) como agente.
# v3 = SFT a CODIGO Python ejecutable (86 golds SAB safe + 14,178 ecocode L1).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE/ecobench"
export PYTHONPATH="$BASE/venv-extra:${PYTHONPATH:-}"
export PYTHONPATH="$BASE/ecobench:${PYTHONPATH:-}"
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay":ro "$BASE/pytorch-cuda.sif" \
  /bb/bwvenv/bin/python "$BASE/ecobench/run_eval_ecoreasoner.py" \
  --lora "$BASE/outputs/sft_moe_v3/lora-final/lora.pt" \
  --model-dir "$BASE/models/LLaDA-MoE-7B-A1B-Instruct" \
  --gen-length 512 --timeout-per-item 300
echo "eval-ecoreasoner-v3 done rc=$?"