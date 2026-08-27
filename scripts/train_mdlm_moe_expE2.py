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
    p.add_argument("--aux_coeff",type=float,default=0.0,help="coef del loss aux de balance de carga (0=off; recomienda 0.01)")
    p.add_argument("--ent_beta",type=float,default=0.0,help="coef de regularización por entropía del gate (Plan B; 0=off; probar 0.05-0.2)")
    p.add_argument("--lr_decay",type=str,default="none",choices=["none","cosine"],help="decay de LR tras warmup: none o cosine")
    p.add_argument("--lr_min_ratio",type=float,default=0.1,help="LR final como fracción del LR inicial (cosine)")
    p.add_argument("--ckpt_every",type=int,default=200,help="guardar checkpoint cada N steps")
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
    # ---- LR schedule: warmup lineal + (opcional) cosine decay ----
    total = max(ARGS.max_steps, 1)
    warm = ARGS.warmup
    def _lr(step):
        # step = número de optim steps (ya en grad_accum)
        if step < warm:
            return step / max(warm, 1)
        if ARGS.lr_decay == "cosine":
            p = (step - warm) / max(total - warm, 1)
            cos = 0.5 * (1 + math.cos(math.pi * p))
            return ARGS.lr_min_ratio + (1 - ARGS.lr_min_ratio) * cos
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr)
    MASK=ARGS.vocab
    n_masked=max(1,int((ARGS.seq_len//2)*ARGS.mask_p))
    nb=len(batches); it=0
    acc=0; acc_aux=0; acc_ent=0; t0=time.time(); opt_steps=0
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
        # loss aux de balance de carga (Switch-Transformer), si activo
        aux = sum(b.mlp.balance_loss(alpha=ARGS.aux_coeff) for b in model.blocks) if ARGS.aux_coeff > 0 else torch.zeros((), device=DEVICE)
        # regularización por entropía del gate (Plan B), si activo
        ent = sum(b.mlp.entropy_reg(beta=ARGS.ent_beta) for b in model.blocks) if ARGS.ent_beta > 0 else torch.zeros((), device=DEVICE)
        loss_total = loss + (ARGS.aux_coeff * aux if ARGS.aux_coeff > 0 else 0) + ent
        loss_total.backward()
        acc+=loss.item(); acc_aux+= aux.item() if ARGS.aux_coeff > 0 else 0.0; acc_ent+= ent.item() if ARGS.ent_beta > 0 else 0.0
        if (step+1)%ARGS.grad_accum==0:
            opt.step(); opt.zero_grad(set_to_none=True); opt_steps+=1
            sched.step()
        if (step+1)%ARGS.log_every==0:
            lr=opt.param_groups[0]["lr"]
            # métricas de router (media sobre capas)
            st=[b.mlp.router_stats() for b in model.blocks if hasattr(b.mlp,"router_stats")]
            ents=[s["entropy"] for s in st if s]
            eff=[s["eff_n"] for s in st if s]
            ent = sum(ents)/len(ents) if ents else float("nan")
            effn = sum(eff)/len(eff) if eff else float("nan")
            log(f"[step {step+1}] loss {acc/ARGS.log_every:.4f} aux {acc_aux/ARGS.log_every:.4f} ent {acc_ent/ARGS.log_every:.4f} lr {lr:.2e} entH {ent:.3f} effN {effn:.2f} ({time.time()-t0:.0f}s)")
            acc=0; acc_aux=0; acc_ent=0
        # save ckpt
        if (step+1)%ARGS.ckpt_every==0:
            torch.save(model.state_dict(), f"{ARGS.out_dir}/e2_step{step+1}.pt")
    torch.save(model.state_dict(), f"{ARGS.out_dir}/e2_final.pt")
    log(f"DONE {ARGS.max_steps} steps -> {ARGS.out_dir}/e2_final.pt")

if __name__=="__main__":
    main()