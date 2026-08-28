#!/usr/bin/env bash
#SBATCH --job-name=bw-torch-setup
#SBATCH --partition=sixhour
#SBATCH --time=01:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=20G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/bw_torch_setup_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/bw_torch_setup_%j.err

# Prepara entorno Blackwell (sm_120): dentro del overlay, crea un VENV LIMPIO e
# instala torch 2.7.1 cu128 DENTRO del venv (--ignore-installed) para NO tocar la
# base inmutable del SIF (ahí fallaba desinstalar triton-3.0.0 -> OSError dir not empty).
# Valida sm_120. Los slurm de training montaran --overlay y apuntaran a este venv.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF="$BASE/pytorch-cuda.sif"
OVL="$BASE/blackwell-torch.overlay"
SZ=20000
VENV="/bb/bwvenv"

cd "$BASE"
echo "== 1. crear overlay (si falta) =="
if [ ! -f "$OVL" ]; then
  apptainer overlay create --size $SZ "$OVL"
  echo "overlay creado: $OVL"
else
  echo "overlay ya existe"
fi

echo "== 2. crear venv limpio dentro del overlay =="
# /app no existe en el SIF -> crea el venv limpio (no pisa /opt/conda base)
apptainer exec --overlay "$OVL" "$SIF" \
  /opt/conda/bin/python -m venv --system-site-packages "$VENV"

echo "== 3. instalar torch 2.7.1 cu128 en el venv (ignore-installed) =="
apptainer exec --nv --overlay "$OVL" "$SIF" \
  "$VENV/bin/pip" install --no-cache-dir --ignore-installed \
    torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

echo "== 4. validar sm_120 desde el venv =="
apptainer exec --nv --overlay "$OVL" "$SIF" \
  "$VENV/bin/python" -c \
    "import torch; print('torch', torch.__version__); print('cap', torch.cuda.get_device_capability(0)); print('arch', torch.cuda.get_arch_list()); print('OK sm120' if (12,0)==torch.cuda.get_device_capability(0) else 'FAIL')"

echo "SETUP DONE rc=$?"