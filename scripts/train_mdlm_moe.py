#!/usr/bin/env python3
"""MdLMMoE — masked-diffusion LLM with MoE FFN (EcoReasoner dLLM PoC).

Trains a small masked-diffusion language model (D3PM absorb-mask objective, the
same denoising objective behind LLaDA/MDLM) with a sparse MoE FFN, on the sci
corpus, using the EcoReasoner wave pattern:
  - SIGUSR1 -> save checkpoint + exit 42 (ola boundary, auto-resume)
  - resume from state.json (global step) + newest checkpoint-gN
  - write progress.json / state.json for the swarm watchdog
Run via apptainer SIF (ROCm), 1-2 GPUs. Single-GPU friendly for PoC.
"""
import argparse, json, os, signal, sys, time, shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- CLI ----------------
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--expert_k", type=int, default=1)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=768)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--mask_p", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--tokenizer", default="/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
    p.add_argument("--output", required=True)
    return p.parse_args()
ARGS = parse()

OUT = Path(ARGS.output); OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "train.log"
def log(msg):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(s, flush=True)
    try:
        with open(LOG, "a") as f: f.write(s + "\n")
    except Exception: pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- model ----------------
def _default_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

class MoEMLP(nn.Module):
    """Sparse MLP FFN with top-k router over n_experts."""
    def __init__(self, dim, ff, n_experts, k):
        super().__init__()
        self.n, self.k = n_experts, k
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))
            for _ in range(n_experts)])
    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        g = torch.softmax(self.gate(flat).float(), dim=-1)
        gv, gi = g.topk(self.k, dim=-1)
        out = torch.zeros_like(flat)
        for rank in range(self.k):
            ids = gi[:, rank]; w = gv[:, rank]
            for e in range(self.n):
                sel = (ids == e)
                if sel.any():
                    out[sel] += w[sel, None] * self.experts[e](flat[sel])
        return out.reshape(B, T, D)

class Block(nn.Module):
    def __init__(self, dim, ff, heads, n_experts, k):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MoEMLP(dim, ff, n_experts, k)
    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))

class MdLMMoE(nn.Module):
    def __init__(self, vocab, hidden, layers, heads, ff_mult, seq_len, n_experts, k):
        super().__init__()
        self.vocab = vocab
        self.tok_emb = nn.Embedding(vocab + 1, hidden)   # +1 for MASK token
        self.pos = nn.Embedding(seq_len, hidden)
        self.blocks = nn.ModuleList([
            Block(hidden, hidden*ff_mult, heads, n_experts, k) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab)
        self.apply(_default_init)
    def forward(self, ids):
        B, T = ids.shape
        h = self.tok_emb(ids) + self.pos(torch.arange(T, device=ids.device))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h))
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

def build_model():
    return MdLMMoE(ARGS.vocab, ARGS.hidden, ARGS.layers, ARGS.heads,
                   ARGS.ff_mult, ARGS.seq_len, ARGS.n_experts, ARGS.expert_k)

# ---------------- data ----------------
def load_corpus(paths):
    rows = []
    for p in paths:
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if not ln: continue
            try:
                o = json.loads(ln)
                txt = (str(o.get("prompt","")) + "\n" + str(o.get("answer") or o.get("title") or "")).strip()
            except Exception:
                txt = ln
            rows.append(txt)
    return rows

def build_batches():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ARGS.tokenizer, trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    corp = load_corpus(ARGS.data)
    seqs = []
    for t in corp:
        ids = tok.encode(t)[:ARGS.seq_len-2]
        if len(ids) >= 4:
            seqs.append(torch.tensor(ids, dtype=torch.long))
    log(f"corpus docs: {len(corp)}, usable: {len(seqs)}")
    log(f"tokenizer vocab_size: {tok.vocab_size}")
    all_ids = torch.cat(seqs) if seqs else torch.tensor([], dtype=torch.long)
    b = ARGS.batch_size
    n = (all_ids.numel() // (b * ARGS.seq_len)) * (b * ARGS.seq_len)
    if n == 0: raise RuntimeError("corpus too small for a single batch")
    buf = all_ids[:n].view(b, -1)
    return tok, [buf[:, i*ARGS.seq_len:(i+1)*ARGS.seq_len]
                 for i in range(buf.size(1)//ARGS.seq_len)]

# ---------------- checkpoint / resume ----------------
glob_model = None
glob_opt = None
STEPS_DONE = [0]
LAST_LOSS = [0.0]

def _save_checkpoint(tag):
    g = STEPS_DONE[0]
    ckpt = OUT / f"checkpoint-g{g}"
    ckpt.mkdir(parents=True, exist_ok=True)
    # if DDP-wrapped, save the inner module weights (no "module." prefix)
    sd = glob_model.state_dict()
    if isinstance(glob_model, torch.nn.parallel.DistributedDataParallel):
        sd = glob_model.module.state_dict()
    torch.save({"model": sd}, ckpt/"model.pt")
    torch.save({"optimizer": glob_opt.state_dict()}, ckpt/"optimizer.pt")
    (OUT/"state.json").write_text(json.dumps(
        {"step": g, "checkpoint": f"checkpoint-g{g}", "updated": time.time()}))
    (OUT/"progress.json").write_text(json.dumps(
        {"step": g, "loss": LAST_LOSS[0], "updated": time.time()}))
    for f in sorted(OUT.glob("checkpoint-g*")):
        if f.name != ckpt.name: shutil.rmtree(f, ignore_errors=True)
    log(f"  checkpoint g{g} guardado")

def resume():
    sf = OUT/"state.json"
    if not sf.exists(): return
    try: state = json.loads(sf.read_text())
    except Exception: return
    ck = OUT/state.get("checkpoint","")
    if ck.exists() and (ck/"model.pt").exists():
        glob_model.load_state_dict(torch.load(ck/"model.pt", map_location="cpu")["model"])
        glob_opt.load_state_dict(torch.load(ck/"optimizer.pt", map_location="cpu")["optimizer"])
        STEPS_DONE[0] = state.get("step",0)
        log(f"Resumed {state.get('checkpoint')} (step {STEPS_DONE[0]})")

def _handle_sig(sig, frm):
    log("SIGUSR1 — guardando ola y saliendo 42")
    _save_checkpoint("sigusr1")
    raise SystemExit(42)

signal.signal(signal.SIGUSR1, _handle_sig)

# ---------------- train ----------------
def main():
    global glob_model, glob_opt, DEVICE
    # ---- DDP init (multi-GPU via slurm) ----
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    world_size = world
    ddp = world_size > 1
    if ddp:
        import socket
        # ensure rendezvous env (srun/apptainer may not forward these)
        os.environ.setdefault("MASTER_ADDR", socket.gethostname())
        os.environ.setdefault("MASTER_PORT", "29512")
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        # srun+gres may expose one GPU/task (CUDA_VISIBLE_DEVICES) -> clamp to visible set
        nvis = torch.cuda.device_count()
        dev_idx = min(local_rank, max(0, nvis-1))
        torch.cuda.set_device(dev_idx)
        DEVICE = torch.device("cuda", dev_idx)
        log(f"DDP: rank={rank} local={local_rank} world={world_size} visible={nvis} dev={dev_idx}")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok, batches_all = build_batches()
    # distribute batches across ranks (each rank trains on a distinct slice)
    if ddp:
        nb = len(batches_all)
        per = nb // world_size
        batches = batches_all[rank*per : (rank+1)*per] if rank < world_size-1 else batches_all[rank*per:]
        if len(batches)==0: batches = batches_all[:1]
    else:
        batches = batches_all
    # derive vocab from the tokenizer (LLaDA = 126080); +1 slot for MASK
    ARGS.vocab = tok.vocab_size
    if rank==0: log(f"using vocab_size={ARGS.vocab} (from tokenizer)")
    glob_model = build_model().to(DEVICE)
    nparam = glob_model.n_params()
    # active params ~ dense (attn/gate/emb) + activated expert weights (k/n_experts of MoE)
    if ARGS.n_experts > 1:
        # Dense = everything outside the MoE experts. Active = dense + (k/n)·experts.
        all_exp = sum(
            sum(sum(p.numel() for p in b.mlp.experts[e].parameters())
                for e in range(ARGS.n_experts))
            for b in glob_model.blocks)
        all_dense = nparam - all_exp
        active_n = all_dense + all_exp * (ARGS.expert_k / ARGS.n_experts)
        log(f"model: total={nparam/1e6:.1f}M, MoE {ARGS.n_experts} top-{ARGS.expert_k}")
        log(f"model: active params ~ {active_n/1e6:.1f}M ({active_n/nparam:.0%} of total) "
            f"[dense {all_dense/1e6:.0f}M + active experts {all_exp*(ARGS.expert_k/ARGS.n_experts)/1e6:.0f}M]")
    else:
        active_n = nparam
        log(f"model: total={nparam/1e6:.1f}M (dense)")
    glob_opt = torch.optim.AdamW(glob_model.parameters(), lr=ARGS.lr, weight_decay=0.01)
    resume()
    # wrap in DDP after resume so state stays on raw module.
    # MoE: per-iteration unused experts -> need find_unused_parameters=True
    if ddp:
        glob_model = torch.nn.parallel.DistributedDataParallel(
            glob_model, device_ids=[dev_idx], find_unused_parameters=True)
    glob_model.zero_grad(set_to_none=True)
    MASK = ARGS.vocab
    n_masked = max(1, int((ARGS.seq_len//2) * ARGS.mask_p))
    nb = len(batches); it = 0
    for step in range(STEPS_DONE[0], ARGS.max_steps):
        xb = batches[it % nb].to(DEVICE); it += 1
        xm = xb.clone(); head = xb.size(1)//2
        mp = torch.randperm(xb.size(1))[:n_masked]
        # mask in second half (like diffusion delete region) — simplest: mask per half
        xm[:, mp] = ARGS.vocab
        out = glob_model(xm)
        loss = F.cross_entropy(out[:, mp].reshape(-1, ARGS.vocab),
                               xb[:, mp].reshape(-1))
        (loss/ARGS.grad_accum).backward()
        if (step+1) % ARGS.grad_accum == 0:
            glob_opt.step(); glob_opt.zero_grad(set_to_none=True)
        LAST_LOSS[0] = loss.item(); STEPS_DONE[0] = step+1
        if ddp: torch.distributed.barrier()
        if (not ddp or rank==0) and step % 10 == 0: log(f"step {step} loss {loss.item():.4f}")
        if (not ddp or rank==0) and step % 50 == 0: _save_checkpoint("step")
    if not ddp or rank==0: _save_checkpoint("final")
    if ddp: torch.distributed.destroy_process_group()
    if not ddp or rank==0: log("COMPLETE")

if __name__ == "__main__":
    main()