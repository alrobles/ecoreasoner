#!/usr/bin/env python3
"""bench_tflops.py — mide TFLOPS REALES (FP16/BF16 matmul) de la GPU local.

Ranking por RFLOPS: medimos en el cluster, no spec de datasheet.
Uso (dentro de un job slurm con la GPU):
  srun apptainer exec --nv pytorch-cuda.sif python3 bench_tflops.py
"""
import torch, time

print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"capability: {torch.cuda.get_device_capability(0)}", flush=True)
print(f"torch {torch.__version__} | cuda {torch.version.cuda}", flush=True)

def bench(dtype, n=8192, iters=10, label=""):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3):
        a @ b  # warmup (JIT/clock ramp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    flops = 2 * n ** 3
    tflops = flops / dt / 1e12
    print(f"  {label:6s} matmul {n}^3 x{iters}: {tflops:8.1f} TFLOPS  ({dt*1e3:.2f} ms/op)", flush=True)
    return tflops

bench(torch.float32, label="FP32")
if torch.cuda.get_device_capability(0)[0] >= 7:
    bench(torch.float16, label="FP16")
if torch.cuda.get_device_capability(0)[0] >= 8:
    bench(torch.bfloat16, label="BF16")
print("DONE", flush=True)