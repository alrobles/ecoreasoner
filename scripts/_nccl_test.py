"""Test nccl inter-nodo. MASTER_ADDR viene del slurm (exportado en host)."""
import os, socket, time, torch, torch.distributed as dist
rank = int(os.environ.get("SLURM_PROCID", "0")); world = int(os.environ.get("SLURM_NTASKS", "1"))
master = os.environ.get("MASTER_ADDR") or socket.gethostname()   # YA viene del slurm
os.environ["MASTER_ADDR"] = master; os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT","29513")
os.environ["RANK"] = str(rank); os.environ["WORLD_SIZE"] = str(world)
print(f"[{socket.gethostname()}] rank={rank} master={master} world={world}", flush=True)
dist.init_process_group("nccl", rank=rank, world_size=world)
t = torch.ones(5_000_000, dtype=torch.bfloat16, device="cuda") * rank  # 10MB
dist.barrier()
t0 = time.time()
for _ in range(5):
    dist.all_reduce(t)
torch.cuda.synchronize()
print(f"[{socket.gethostname()}] rank={rank} all_reduce 10MB x5 = {time.time()-t0:.1f}s OK sum_check={t[0].item():.0f}", flush=True)
dist.destroy_process_group()
