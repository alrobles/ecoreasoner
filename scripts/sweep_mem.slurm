#!/usr/bin/env bash
#SBATCH --job-name=sweep-mem
#SBATCH --partition=sixhour
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=30G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/sweep_mem_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/sweep_mem_%j.err

# Barrido de configs que caben en Q6000 24GB (single GPU = por réplica DDP).
# Modelo expE2 (timestep + shared). Mide peak VRAM con forward+backward+AdamW
# y grad_accum 2. Marca cuáles caben con margen >2GB (para tolerar overhead DDP).
cd /beegfs/a474r867/ecoreasoner
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
apptainer exec --nv pytorch-cuda.sif \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 - <<'PY'
import torch
from scripts.moe_exp_E2_router_timestep import MdLMMoE

# barrido: (hidden, layers, n_experts)
configs = [
    (768,  12, 8),   # e2-v3 (baseline conocido que corre)
    (768,  12, 12),  # v3 + mas expertos
    (896,  12, 8),   # hidden mas grande
    (1024, 12, 8),   # hidden 1024, 12 layers
    (1024, 14, 8),
    (1024, 16, 4),
    (1024, 16, 3),
    (1024, 16, 2),
]
for hidden, layers, ne in configs:
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    try:
        m = MdLMMoE(126080, hidden, layers, hidden//64, 4, 768, n_experts=ne, k=1,
                    tdim=64, n_steps=100, shared=True).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
        ids = torch.randint(0, 126080, (1, 768)).cuda()
        tm = torch.zeros((1, 768), dtype=torch.long).cuda(); tm[:, :100] = 1
        for _ in range(2):
            out = m(ids, timestep=tm)
            loss = out.flatten().sum() + sum(b.mlp.balance_loss(0.01) for b in m.blocks)
            (loss/2).backward()
        opt.step()
        peak = torch.cuda.max_memory_allocated()/1e9
        total = m.n_params()
        allexp = sum(p.numel() for b in m.blocks for e in b.mlp.experts for p in e.parameters())
        shp = sum(p.numel() for b in m.blocks if b.mlp.shared for p in b.mlp.shared.parameters())
        act = (total-allexp-shp) + shp + allexp/ne
        ok = peak < 21.0  # margen >2GB para overhead DDP
        print(f"h={hidden} l={layers} ne={ne}: total={total/1e6:.0f}M activos={act/1e6:.0f}M peak={peak:.1f}GB {'OK' if ok else 'MARGEN-INSUF'}", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"h={hidden} l={layers} ne={ne}: OOM", flush=True)
    del m, opt
    try: torch.cuda.empty_cache()
    except: pass
PY
echo "sweep-mem done rc=$?"