#!/usr/bin/env python3
"""Mini-test all-reduce real en 2 GPUs (validar NCCL 2.26.5 en Blackwell)."""
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
t = torch.ones(1 << 24, dtype=torch.bfloat16, device=f"cuda:{rank}") * (rank + 1)
for i in range(5):
    dist.all_reduce(t)
    torch.cuda.synchronize()
dist.barrier()
print(f"rank {rank}: nccl={torch.cuda.nccl.version()} allreduce OK sum={t[0].item():.0f}", flush=True)
dist.destroy_process_group()