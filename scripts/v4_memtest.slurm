#!/usr/bin/env bash
#SBATCH --job-name=v4memtest
#SBATCH --partition=sixhour
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=30G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/v4memtest_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/v4memtest_%j.err

# Test de memoria para el modelo v4 (hidden 1024, layers 16, 12 exp) en Q6000 24GB.
# Instancia el modelo, hace forward/backward con batch variable, mide VRAM.
cd /beegfs/a474r867/ecoreasoner
apptainer exec --nv pytorch-cuda.sif python3 - <<'PY'
import torch, os
from scripts.moe_exp_E2_router_timestep import MdLMMoE

os.makedirs("outputs/memtest_v4", exist_ok=True)
torch.manual_seed(0)

cfg = dict(vocab=126080, hidden=1024, layers=16, heads=16, ff_mult=4, seq_len=768,
           n_experts=12, k=1, tdim=64, n_steps=100, shared=True)
m = MdLMMoE(**cfg)
print(f"params total: {m.n_params()/1e6:.1f}M", flush=True)
# conteo activos
all_exp = sum(p.numel() for b in m.blocks for e in b.mlp.experts for p in e.parameters())
shared = sum(p.numel() for b in m.blocks if b.mlp.shared is not None for p in b.mlp.shared.parameters())
dense = m.n_params() - all_exp - shared
active = dense + shared + all_exp/12
print(f"activos ≈ {active/1e6:.1f}M (dense {dense/1e6:.1f}M + shared {shared/1e6:.1f}M + exp {all_exp/12/1e6:.1f}M)", flush=True)

for bs in [1, 2, 4]:
    torch.cuda.reset_peak_memory_stats()
    m = m.cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
    ids = torch.randint(0, 126080, (bs, 768)).cuda()
    tm = torch.zeros((bs, 768), dtype=torch.long).cuda()
    tm[:, :100] = 1
    out = m(ids, timestep=tm)
    loss = out.flatten(0,1).sum() + sum(b.mlp.balance_loss(0.01) for b in m.blocks)
    loss.backward(); opt.step()
    peak = torch.cuda.max_memory_allocated()/1e9
    print(f"batch {bs}: forward+backward OK, peak VRAM {peak:.2f} GB", flush=True)
    del out, loss
PY
echo "memtest done rc=$?"