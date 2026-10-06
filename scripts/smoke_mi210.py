"""smoke_mi210.py — spike ROCm: corre el MdLMMoE v5-1b en una MI210.

Mide fwd/bwd sintetico (bs x seq768, dtype del trainer) y reporta tok/s.
Sin DDP: solo prueba que el modelo corre en gfx90a con venv_rocm.

Uso (en nodo mi210): venv_rocm/bin/python smoke_mi210.py [--steps 20 --bs 4]
"""
import argparse, importlib.util, sys, time
from pathlib import Path
import torch

BASE = Path("/beegfs/a474r867/ecoreasoner")
ARCH = dict(vocab=126080, hidden=768, layers=12, heads=12, ff_mult=4,
            seq_len=768, n_experts=16, k=1, use_rope=False,
            weight_tying=False, n_shared=0, shared_ff_div=4, fine_ff_div=1)
MASK_ID = ARCH["vocab"]

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--bf16", type=int, default=1)
a = ap.parse_args()

p = BASE/"scripts"/"train_mdlm_moe_v2.py"
spec = importlib.util.spec_from_file_location("tmm2", p)
mod = importlib.util.module_from_spec(spec)
sys.argv = [p.name, "--data", "dummy", "--output", "/tmp/smoke_mi210"]
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass  # el trainer parsea args; el modulo ya quedo ejecutado
except Exception as e:
    print(f"[smoke] WARN exec_module: {e}")
MdLMMoE = mod.MdLMMoE

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[smoke] torch={torch.__version__} hip={torch.version.hip} "
      f"avail={torch.cuda.is_available()} dev={dev}")
if dev.type == "cuda":
    print(f"[smoke] gpu={torch.cuda.get_device_name(0)} "
          f"cap={torch.cuda.get_device_capability(0)} "
          f"mem={torch.cuda.get_device_properties(0).total_memory/2**30:.0f}GiB")

model = MdLMMoE(**ARCH).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()
dt = torch.bfloat16 if a.bf16 else torch.float32
print(f"[smoke] params={sum(p.numel() for p in model.parameters())/1e9:.2f}B "
      f"dtype={dt} bs={a.bs}")

# warmup
x = torch.randint(0, ARCH["vocab"], (a.bs, ARCH["seq_len"]), device=dev)
for _ in range(3):
    with torch.autocast("cuda", dtype=dt, enabled=(dev.type == "cuda")):
        logits = model(x)
    loss = logits.float().mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
if dev.type == "cuda":
    torch.cuda.synchronize()

t0 = time.time()
for i in range(a.steps):
    x = torch.randint(0, ARCH["vocab"], (a.bs, ARCH["seq_len"]), device=dev)
    with torch.autocast("cuda", dtype=dt, enabled=(dev.type == "cuda")):
        logits = model(x)
    loss = logits.float().mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
if dev.type == "cuda":
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated()/2**30
else:
    mem = 0
el = time.time() - t0
tps = a.steps * a.bs * ARCH["seq_len"] / el
print(f"[smoke] {a.steps} steps en {el:.1f}s = {el/a.steps:.3f}s/step, "
      f"{tps:.0f} tok/s, peak_mem={mem:.1f}GiB")
