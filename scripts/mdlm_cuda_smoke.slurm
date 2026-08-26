#!/bin/bash
#SBATCH --job-name=mdlm-cuda-smoke
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/mdlm_cuda_smoke_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/mdlm_cuda_smoke_%j.err

set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif

echo "=== CUDA smoke: $(date) ==="
nvidia-smi | head -12 || echo "no nvidia-smi"
apptainer exec --nv \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$SIF" python3 -c '
import torch, sys
print("torch", torch.__version__)
print("cuda avail:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("check return:", torch.zeros(2).cuda())

# Build 0.5B MoE and a masked-diffusion step
sys.path.insert(0, "/beegfs/a474r867/ecoreasoner/scripts")
from train_mdlm_moe import MdLMMoE, ARGS

m = MdLMMoE(32000, 768, 12, 12, 4, 768, 8, 1).cuda()
n = sum(p.numel() for p in m.parameters())
print(f"model params: {n/1e6:.1f}M")
import torch.nn.functional as F
torch.manual_seed(0)
x = torch.randint(0, 30000, (1, 768)).cuda()
mp = torch.randperm(768)[:120].cuda()
xm = x.clone(); xm[:, mp] = 32000
out = m(xm)
print("forward out:", out.shape)
loss = F.cross_entropy(out[:, mp].reshape(-1, 32000), x[:, mp].reshape(-1))
print("loss:", loss.item())
loss.backward()
g = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
print(f"backward OK: {g}/{len(list(m.parameters()))} grads")
import time
t0=time.time()
for _ in range(5):
    out = m(xm)
    F.cross_entropy(out[:, mp].reshape(-1,32000), x[:,mp].reshape(-1)).backward()
print(f"5 steps in {time.time()-t0:.1f}s")
print("CUDA SMOKE PASS")
'