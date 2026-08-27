#!/bin/bash
#SBATCH --job-name=gen-distill
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/gen_distill_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/gen_distill_%j.err

# Genera datos de destilacion: teacher v4-flash resuelve los prompts canonicos
# (tool-calls cerrados) y guarda trayectorias entrenables.
# Endpoint: teacher v4-flash en r30r24n01:39815 (red interna HPC, sin tunel)
set -u
BASE=/beegfs/a474r867/ecoreasoner
echo "[$(date)] generando datos de destilacion..."
python3 "$BASE/scripts/gen_distill_data.py" \
    --endpoint "http://r30r24n01:39815/v1/chat/completions" \
    --prompts "$BASE/data/prompts_toolcall_canonical.jsonl" \
    --out "$BASE/data/distill_data.jsonl" \
    --variants 3 --max_rounds 3
echo "[$(date)] GEN-DISTILL DONE exit=$?"