"""smoke_mi210_ddp.py — spike ROCm fase 2: DDP intra-nodo (RCCL) de MdLMMoE v5-1b.

3 ranks en un nodo mi210 via torchrun. Verifica: (a) init_process_group
nccl->rccl, (b) DDP fwd/bwd con all-reduce real, (c) tok/s agregado.
Gate estrategia-escala: si pasa -> clonar v5_1b_ddp.slurm a mi210.

Uso (en nodo mi210): torchrun --nproc_per_node=3 smoke_mi210_ddp.py [--steps 20 --bs 4]
"""
import argparse, importlib.util, os, sys, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

BASE = Path("/beegfs/a474r867/ecoreasoner")
ARCH = dict(vocab=126080, hidden=768, layers=12, heads=12, ff_mult=4,
            seq_len=768, n_experts=16, k=1, use_rope=False,
            weight_tying=False, n_shared=0, shared_ff_div=4, fine_ff_div=1)

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--bs", type=int, default=4)
a = ap.parse_args()

p = BASE/"scripts"/"train_mdlm_moe_v2.py"
spec = importlib.util.spec_from_file_location("tmm2", p)
mod = importlib.util.module_from_spec(spec)
sys.argv = [p.name, "--data", "dummy", "--output", "/tmp/smoke_mi210_ddp"]
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass  # el trainer parsea args; el modulo ya quedo ejecutado
except Exception as e:
    print(f"[smoke] WARN exec_module: {e}", flush=True)
MdLMMoE = mod.MdLMMoE

local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dist.init_process_group("nccl")  # torch+rocm: "nccl" aliasa a RCCL
rank, world = dist.get_rank(), dist.get_world_size()
dev = torch.device("cuda", local)
if rank == 0:
    print(f"[smoke-ddp] world={world} gpu={torch.cuda.get_device_name(local)} "
          f"bs={a.bs} dtype=bf16", flush=True)

model = MdLMMoE(**ARCH).to(dev)
ddp = DDP(model, device_ids=[local], find_unused_parameters=False)
opt = torch.optim.AdamW(ddp.parameters(), lr=1e-5)
ddp.train()

def step():
    x = torch.randint(0, ARCH["vocab"], (a.bs, ARCH["seq_len"]), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = ddp(x)
    # loss fiel al trainer: balance_loss toca TODOS los expertos (top-1
    # deja la mayoria sin grad -> aux obligatorio con find_unused=False)
    aux = sum(b.mlp.balance_loss(0.01) for b in model.blocks)
    loss = logits.float().mean() + aux
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

for _ in range(3):
    step()
torch.cuda.synchronize(); dist.barrier()

t0 = time.time()
for _ in range(a.steps):
    step()
torch.cuda.synchronize(); dist.barrier()
el = time.time() - t0
tps = a.steps * a.bs * ARCH["seq_len"] * world / el
mem = torch.cuda.max_memory_allocated()/2**30
if rank == 0:
    print(f"[smoke-ddp] {a.steps} steps x {world} ranks: {el:.1f}s = "
          f"{el/a.steps:.3f}s/step, {tps:.0f} tok/s agg, peak_mem={mem:.1f}GiB",
          flush=True)

t = torch.tensor([float(rank + 1)], device=dev)
dist.all_reduce(t)
if rank == 0:
    print(f"[smoke-ddp] all_reduce OK sum={t.item()} "
          f"(esperado={world*(world+1)/2})", flush=True)
dist.destroy_process_group()
