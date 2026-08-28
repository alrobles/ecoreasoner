#!/usr/bin/env bash
#SBATCH --job-name=v4memtest3
#SBATCH --partition=sixhour
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=30G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/v4memtest3_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/v4memtest3_%j.err

# Test de memoria MoE v4 con 2 y 3 expertos, batch 1 + grad_accum 2.
# Busca config con margen holgado (<~19GB) para aguantar overhead DDP en 24GB.
cd /beegfs/a474r867/ecoreasoner
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
apptainer exec --nv pytorch-cuda.sif \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 - <<'PY'
import torch
from scripts.moe_exp_E2_router_timestep import MdLMMoE

for ne in [2, 3]:
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    m = MdLMMoE(126080, 1024, 16, 16, 4, 768, n_experts=ne, k=1, tdim=64, n_steps=100, shared=True)
    m = m.cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
    ids = torch.randint(0, 126080, (1, 768)).cuda()
    tm = torch.zeros((1, 768), dtype=torch.long).cuda(); tm[:, :100] = 1
    try:
        for _ in range(2):
            out = m(ids, timestep=tm)
            loss = out.flatten().sum() + sum(b.mlp.balance_loss(0.01) for b in m.blocks)
            (loss/2).backward()
        opt.step()
        peak = torch.cuda.max_memory_allocated()/1e9
        print(f"n_experts={ne}: total={m.n_params()/1e6:.0f}M, PEAK={peak:.2f}GB (margen {23.5-peak:.1f}GB)", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"n_experts={ne}: OOM", flush=True)
    del m, opt
    try: torch.cuda.empty_cache()
    except: pass
PY
echo "memtest3 done rc=$?"