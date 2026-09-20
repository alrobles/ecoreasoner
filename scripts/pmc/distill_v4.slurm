#!/usr/bin/env bash
# DEPRECATED 2026-09-20: dependia de un modelo de origen chino vetado
# (Qwen / Tencent HY-MT / LLaDA weights / DeepSeek). Los pesos fueron
# eliminados del cluster; este script ya no corre. El tokenizer LLaDA
# (vocab 126080) sigue vivo SOLO como tokenizer de nuestro modelo.
#SBATCH --job-name=distill-v4
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/distill_v4_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/distill_v4_%j.err

# Destilación con teacher DeepSeek-V4-Flash (nodo actual) — CPU, sin GPU
cd /beegfs/a474r867/ecoreasoner
TEACHER_ENDPOINT="${TEACHER_ENDPOINT:-http://r30r08n01:43919/v1/chat/completions}"
echo "[$(date)] teacher: $TEACHER_ENDPOINT"
PYTHONUNBUFFERED=1 python3 -u scripts/gen_distill_data.py \
  --endpoint "$TEACHER_ENDPOINT" \
  --variants 6 \
  --max_rounds 8 \
  --out data/distill_v4_round3.jsonl
echo "[$(date)] done rc=$?"