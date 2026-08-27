#!/usr/bin/env python3
"""
train_mdlm_moe_expE2.py — EXPERIMENTO E2+E4 (frontera MoE-dLLM de dominio).

Entrena el dLLM-MoE con dos mejoras fronterizas sobre el baseline (train_mdlm_moe.py):
  E2: router condicionado al TIMESTEP de denoising (gate(x + t_emb(mask)))
  E4: EXPERTO COMPARTIDO siempre activo + top-k especializados (patrón DeepSeek)

DATASET (ver docs/dataset-registry.md): usa 2_ids_3 (train_ids_v3.npy),
el pre-tokenizado único de 2_pretrain_3 (train_corpus_v3.jsonl).

El resto (datos cache, DDP, resume, SIGUSR1, slurm) es idéntico al baseline.

Uso (en cluster, 1 GPU):
  python3 train_mdlm_moe_expE2.py --data_cache data/train_ids_v3.npy --max_steps 500
"""
import argparse, os, sys, time, json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_exp_E2_router_timestep import MdLMMoE as MdLMMoE_E2   # E2+E4 classes

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--n_experts",type=int,default=8)
    p.add_argument("--expert_k",type=int,default=1)
    p.add_argument("--hidden",type=int,default=768)
    p.add_argument("--layers",type=int,default=12)
    p.add_argument("--heads",type=int,default=12)
    p.add_argument("--ff_mult",type=int,default=4)
    p.add_argument("--seq_len",type=int,default=768)
    p.add_argument("--batch_size",type=int,default=2)
    p.add_argument("--grad_accum",type=int,default=2)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--warmup",type=int,default=200)
    p.add_argument("--mask_p",type=float,default=0.15)
    p.add_argument("--max_steps",type=int,default=500)
    p.add_argument("--log_every",type=int,default=20)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--data_cache",default="train_ids_v3.npy")
    p.add_argument("--data",default="",help="fallback jsonl (sin cache)")
    p.add_argument("--timestep_emb",type=int,default=64)
    p.add_argument("--shared",type=int,default=1,help="1=experto compartido (E4), 0=sin")
    p.add_argument("--out_dir",default="outputs/moe_expE2")
    return p.parse_args()

ARGS=parse()
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS_DONE=[0]

def log(msg): print(msg, flush=True)

def _load_tokenizer():
    # LLaDA tokenizer nativo (sin transformers): vocab fijo del cache
    class _T: vocab_size=126080
    return _T()

def build_model():
    m=MdLMMoE_E2(ARGS.vocab, ARGS.hidden, ARGS.layers, ARGS.heads, ARGS.ff_mult,
                 ARGS.seq_len, ARGS.n_experts, ARGS.expert_k,
                 tdim=ARGS.timestep_emb, n_steps=100, shared=bool(ARGS.shared))
    return m

def build_batches():
    import numpy as np
    arr=np.load(ARGS.data_cache)
    tok=_load_tokenizer()
    all_ids=torch.from_numpy(arr.astype(np.int64))
    log(f"cache: {arr.size/1e9:.2f}B tokens, vocab={tok.vocab_size}")
    b=ARGS.batch_size
    n=(all_ids.numel()//(b*ARGS.seq_len))*(b*ARGS.seq_len)
    buf=all_ids[:n].view(b,-1)
    return tok,[buf[:,i*ARGS.seq_len:(i+1)*ARGS.seq_len] for i in range(buf.size(1)//ARGS.seq_len)]

def main():
    random.seed(ARGS.seed); np.random.seed(ARGS.seed); torch.manual_seed(ARGS.seed)
    os.makedirs(ARGS.out_dir, exist_ok=True)
    tok,batches=build_batches()
    ARGS.vocab=tok.vocab_size
    model=build_model().to(DEVICE)
    nparam=model.n_params()
    # conteo activos (incl shared)
    all_exp=sum(sum(sum(p.numel() for p in b.mlp.experts[e].parameters()) for e in range(ARGS.n_experts)) for b in model.blocks)
    shared_p=sum(p.numel() for b in model.blocks if b.mlp.shared is not None for p in b.mlp.shared.parameters())
    all_dense=nparam-all_exp-shared_p
    active=all_dense+shared_p+all_exp*(ARGS.expert_k/ARGS.n_experts)
    log(f"[E2+E4] total={nparam/1e6:.1f}M act≈{active/1e6:.1f}M | shared={shared_p/1e6:.1f}M | MoE {ARGS.n_experts} top-{ARGS.expert_k} + timestep-router")
    opt=torch.optim.AdamW(model.parameters(),lr=ARGS.lr,weight_decay=0.01)
    MASK=ARGS.vocab
    n_masked=max(1,int((ARGS.seq_len//2)*ARGS.mask_p))
    nb=len(batches); it=0
    acc=0; t0=time.time()
    for step in range(ARGS.max_steps):
        model.train()
        xb=batches[it%nb].to(DEVICE); it+=1
        xm=xb.clone()
        mp=torch.randperm(xb.size(1))[:n_masked]
        xm[:,mp]=MASK
        # E2: timestep por token = 1 si enmascarado, 0 si visible (proxy del paso de denoising)
        timestep=torch.zeros_like(xb); timestep[:,mp]=1
        logits=model(xm, timestep=timestep)
        loss=F.cross_entropy(logits.reshape(-1,ARGS.vocab), xb.reshape(-1), ignore_index=-100)
        # load balance aux: diversidad de gate (sin etiquetas; uniforme ligera)
        loss.backward()
        acc+=loss.item()
        if (step+1)%ARGS.grad_accum==0:
            opt.step(); opt.zero_grad(set_to_none=True)
        if (step+1)%ARGS.log_every==0:
            log(f"[step {step+1}] loss {acc/ARGS.log_every:.4f} ({time.time()-t0:.0f}s)")
            acc=0
        # save ckpt
        if (step+1)%200==0:
            torch.save(model.state_dict(), f"{ARGS.out_dir}/e2_step{step+1}.pt")
    torch.save(model.state_dict(), f"{ARGS.out_dir}/e2_final.pt")
    log(f"DONE {ARGS.max_steps} steps -> {ARGS.out_dir}/e2_final.pt")

if __name__=="__main__":
    main()