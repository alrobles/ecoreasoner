#!/usr/bin/env bash
# DEPRECATED 2026-09-20: dependia de un modelo de origen chino vetado
# (Qwen / Tencent HY-MT / LLaDA weights / DeepSeek). Los pesos fueron
# eliminados del cluster; este script ya no corre. El tokenizer LLaDA
# (vocab 126080) sigue vivo SOLO como tokenizer de nuestro modelo.
#SBATCH --job-name=sft-moe-v2
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=100G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/sft_moe_v2_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/sft_moe_v2_%j.err
# SFT V2 del LLaDA-MoE: SOLO kinds JSON (gen+repairs, sin "final" de codigo),
# LR 1e-5 warmup 500, 2 epochs — enfocado en tool-call JSON robusto.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
export PYTHONPATH="$BASE/venv-extra:${PYTHONPATH:-}"
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay":ro "$BASE/pytorch-cuda.sif" \
  /bb/bwvenv/bin/python "$BASE/scripts/sft_lladamoe.py" \
  --batch 8 --max-len 1536 --rank 64 --epochs 2 \
  --lr 1e-5 --warmup 500 \
  --pairs "$BASE/data/l1/sft_moe_pairs_v2.jsonl" \
  --out "$BASE/outputs/sft_moe_v2"
echo "sft v2 done rc=$?"