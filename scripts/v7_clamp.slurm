#!/usr/bin/env bash
#SBATCH --job-name=v7-clamp
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/v7_clamp_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/v7_clamp_%j.err

# Clamp OOB de train_ids_v7.npy: 2 tokens == 126082 >= vocab(126080) -> 0.
# Regla de la skill: verificar max < vocab ANTES de entrenar (el trainer ya
# clampea, pero trabajamos con archivo limpio para no depender de la guardia).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
apptainer exec "$SIF" python3 - <<'EOF'
import numpy as np, time
src = "/beegfs/a474r867/ecoreasoner/data/train_ids_v7.npy"
dst = "/beegfs/a474r867/ecoreasoner/data/train_ids_v7_clean.npy"
t0 = time.time()
a = np.load(src)
n_bad = int((a >= 126080).sum())
a = np.where(a >= 126080, 0, a).astype(np.int32)
np.save(dst, a)
print(f"clamp {n_bad} tokens >= 126080 -> 0 | max={int(a.max())} min={int(a.min())} n={a.size} | {time.time()-t0:.1f}s")
print("OOB_OK" if int(a.max()) < 126080 else "OOB_AUN_PROBLEMA")
EOF
echo "[$(date)] CLAMP DONE exit=$?"