#!/usr/bin/env python
"""Repro minimo: 1 forward + 1 backward del MoE 8 exp top-2 en 1 GPU, single rank.
Con CUDA_LAUNCH_BLOCKING=1 + TORCH_USE_CUDA_DSA=1 => un illegal memory access
se reporta como traceback Python con la linea exacta (no solo watchdog NCCL).
"""
import os, sys, json
import torch
import torch.nn.functional as F

# el trainer ejecuta parse() al importarse -> darle args dummy validos
sys.argv = ["repro", "--data", "x", "--output", "/tmp/repro_out",
            "--hidden", "1024", "--layers", "16", "--heads", "16",
            "--n_experts", "8", "--expert_k", "2", "--seq_len", "768",
            "--batch_size", "4", "--grad_accum", "4", "--sync_every", "4"]
sys.path.insert(0, "/beegfs/a474r867/ecoreasoner/scripts")
import train_mdlm_moe_lsgd as T

torch.manual_seed(0)
dev = torch.device("cuda")

# Parametros identicos a bw1
hid, layers, heads, ff = 1024, 16, 16, 4
n_exp, k = 8, 2
seq, bsz = 768, 4

# Vocab real de LLaDA
V = 126080  # LLaDA vocab_size (el trainer lo setea en main() desde el tokenizer)
T.ARGS.vocab = V  # build_model() lee ARGS.vocab
print("vocab", V, "device", torch.cuda.get_device_name(0))

# IMPORTANTE: usar build_model() identico al trainer real -> MdLMMoE(ARGS.vocab,..)
# con head=vocab y tok_emb=vocab+1 (MASK=vocab cae dentro). No pasar V+1 manual.
model = T.build_model().to(dev)
print("model", model.n_params()/1e6, "M")
N = torch.cuda.mem_get_info(0)
print("VRAM libre GB:", N[0]/1e9, "uso MB:", N[1]/1e6)

opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

# batch aleatorio de tokens validos (< V)
xb = torch.randint(0, V-1, (bsz, seq), device=dev)
xm = xb.clone()
head = seq//2
n_masked = max(1, int(head*0.15))
mp = torch.randperm(seq, device=dev)[:n_masked]
xm[:, mp] = V  # MASK token = vocab (dentro de Embedding vocab+1)

print("forward+backward...")
out = model(xm)
loss = F.cross_entropy(out[:, mp].reshape(-1, V), xb[:, mp].reshape(-1))
raw = model
aux = sum(b.mlp.balance_loss(0.01) for b in raw.blocks)
(loss + aux).backward()
print("OK loss", loss.item(), "aux", aux.item(), "grad_norm",
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
opt.step()
print("REPRO_PASS")
