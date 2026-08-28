#!/usr/bin/env bash
#SBATCH --job-name=bw-torch-install
#SBATCH --partition=sixhour
#SBATCH --time=01:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=20G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/bw_torch_install_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/bw_torch_install_%j.err

# Prepara el entorno Blackwell: crea un overlay ext3 persistente sobre pytorch-cuda.sif
# e instala torch>=2.6 (cu128) que soporta sm_120. Valida con torch.cuda.get_device_capability.
# Reusable: los slurm de training montaran --overlay.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF="$BASE/pytorch-cuda.sif"
OVL="$BASE/blackwell-torch.overlay"
SZ=20000   # 20GB overlay

cd "$BASE"
echo "== 1. crear overlay =="
if [ ! -f "$OVL" ]; then
  apptainer overlay create --size $SZ "$OVL"
  echo "overlay creado: $OVL"
else
  echo "overlay ya existe"
fi

echo "== 2. instalar torch cu128 (sm_120) dentro del overlay =="
apptainer exec --nv --overlay "$OVL" "$SIF" \
  /opt/conda/bin/pip install --no-cache-dir \
    torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

echo "== 3. validar sm_120 =="
apptainer exec --nv --overlay "$OVL" "$SIF" \
  /opt/conda/bin/python -c \
    "import torch; print('torch', torch.__version__); print('cap', torch.cuda.get_device_capability(0)); print('arch', torch.cuda.get_arch_list()); print('OK sm120' if (12,0)==torch.cuda.get_device_capability(0) else 'FAIL')"

echo "PREP DONE rc=$?"