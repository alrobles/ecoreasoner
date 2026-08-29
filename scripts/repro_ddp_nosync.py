#!/usr/bin/env python
"""Repro DDP world=2 con no_sync() (path del lsgd con sync_every>1).
Corrido via srun -n 2 (sin torchrun/elastic) para que una IMA se propague
como traceback Python con la linea exacta, no que la trague el watchdog NCCL."""
import os, sys
import torch
import torch.nn.functional as F
import torch.distributed as dist

sys.argv = ["repro", "--data", "x", "--output", "/tmp/repro_out",
            "--hidden", "1024", "--layers", "16", "--heads", "16",
            "--n_experts", "8", "--expert_k", "2", "--seq_len", "768",
            "--batch_size", "4", "--grad_accum", "4", "--sync_every", "4"]
sys.path.insert(0, "/beegfs/a474r867/ecoreasoner/scripts")
import train_mdlm_moe_lsgd as T

rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29560")
dist.init_process_group("nccl", rank=rank, world_size=world)
torch.cuda.set_device(local_rank)
dev = torch.device("cuda", local_rank)
print(f"[{rank}] DDP world={world} dev={dev}", flush=True)

V = 126080
T.ARGS.vocab = V
model = T.build_model().to(dev)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                  find_unused_parameters=False)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
model.zero_grad(set_to_none=True)

seq, bsz, acc = 768, 4, 16
torch.manual_seed(rank)

def one_step(step):
    xb = torch.randint(0, V-1, (bsz, seq), device=dev)
    xm = xb.clone(); head = seq//2
    n_masked = max(1, int(head*0.15))
    mp = torch.randperm(seq, device=dev)[:n_masked]
    xm[:, mp] = V
    out = model(xm)
    loss = F.cross_entropy(out[:, mp].reshape(-1, V), xb[:, mp].reshape(-1))
    raw = model.module
    aux = sum(b.mlp.balance_loss(0.01) for b in raw.blocks)
    is_sync = ((step+1) % acc == 0)
    if T.ARGS.sync_every > 1 and not is_sync:
        with model.no_sync():
            (loss/acc + aux).backward()
        print(f"[{rank}] step {step} no_sync backward OK", flush=True)
    else:
        (loss/acc + aux).backward()
        if is_sync:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            print(f"[{rank}] step {step} SYNC backward+step OK", flush=True)
    dist.barrier()

print(f"[{rank}] arrancando pasos...", flush=True)
for s in range(16):
    one_step(s)
dist.destroy_process_group()
print(f"[{rank}] REPRO_DDP_PASS", flush=True)
