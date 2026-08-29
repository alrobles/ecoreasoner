#!/usr/bin/env python3
"""MdLMMoE multinodo robusto (variante q6000 4xN).

Copia del trainer base (train_mdlm_moe.py) con 2 adiciones para romper el
techo de comunicacion multi-nodo sin cambiar la semantica por defecto:

1. --sync_every K   (post-localSGD de GRADIENTES vía DDP no_sync())
   Con DDP vanilla, CADA backward dispara un all-reduce de gradientes
   (3.4GB fp32 para el modelo 850M) -> a ~200MB/s de red Q6000 son ~17s/step.
   Con sync_every=K, los K-1 backwards van en contexto no_sync() (grad local)
   y solo el K-esimo all-reducea -> comunicacion / K. Equivale a un
   grad_accum global = grad_accum * K. Default 1 = comportamiento EXACTO
   del trainer base (DDP vanilla).

2. --clip_grad C    (grad norm clipping, default 0 = off)
   Robustez multi-nodo ante gradientes explosivos con LR alta.

NO toca el archivo original train_mdlm_moe.py (lo usa bw0 en Blackwell).
Este archivo es la variante para Q6000 multi-nodo; misma onda SIGUSR1/resume.
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
    p.add_argument("--sync_every", type=int, default=1,
                   help="Post-localSGD: all-reduce de gradientes cada K pasos (1 = DDP vanilla). "
                        "Comunicacion / K; lr escala ~ sqrt(K).")
    p.add_argument("--clip_grad", type=float, default=0.0,
                   help="Grad norm clip (0 = off). Robustez multi-nodo.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--mask_p", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--data_cache", default=None,
                   help="Path a un .npy de IDs pre-tokenizados (evita re-tokenizar el corpus en cada slurm). "
                        "Si se da, build_batches carga los IDs de disco en vez de tokenizar.")
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

# ---------------- model (identico al base) ----------------
def _default_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

class MoEMLP(nn.Module):
    """Sparse MLP FFN with top-k router over n_experts (con balance_loss + probe)."""
    def __init__(self, dim, ff, n_experts, k):
        super().__init__()
        self.n, self.k = n_experts, k
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))
            for _ in range(n_experts)])
        self.register_buffer("_fcount", torch.zeros(n_experts), persistent=False)
        self._gate_probs = None
        self._probe_x = None
        self._tokens = 0
    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        g = torch.softmax(self.gate(flat).float(), dim=-1)
        gv, gi = g.topk(self.k, dim=-1)
        if self.training:
            self._gate_probs = g
            self._fcount.zero_()
            self._fcount.scatter_add_(0, gi.reshape(-1), torch.ones(gi.numel(), device=gi.device))
            self._tokens = gi.numel()
            s = min(self.n, flat.shape[0])
            self._probe_x = flat[torch.arange(s, device=flat.device)].detach()
        out = torch.zeros_like(flat)
        for rank in range(self.k):
            ids = gi[:, rank]; w = gv[:, rank]
            for e in range(self.n):
                sel = (ids == e)
                if sel.any():
                    out[sel] += w[sel, None] * self.experts[e](flat[sel])
        return out.reshape(B, T, D)
    def balance_loss(self, alpha=0.01, probe_alpha=0.01):
        """Aux con grad REAL a todo el bloque MoE: router aux + probe por experto."""
        if self._gate_probs is None or self._tokens == 0 or self._probe_x is None:
            return torch.zeros((), device=self.gate.weight.device)
        P = self._gate_probs.mean(0)
        f = self._fcount.to(P.dtype) / max(self._tokens, 1)
        router_aux = alpha * self.n * (f * P).sum()
        probe = torch.zeros((), device=P.device)
        n = self.n
        for e in range(n):
            ye = self.experts[e](self._probe_x)
            probe = probe + (ye ** 2).mean()
        probe = probe / n
        return router_aux + probe_alpha * probe

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
        self.tok_emb = nn.Embedding(vocab + 1, hidden)
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

# ---------------- data (identico) ----------------
def load_corpus(paths):
    rows = []
    for p in paths:
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if not ln: continue
            try:
                o = json.loads(ln)
                txt = (str(o.get("prompt","")) + "\n" + str(o.get("answer") or o.get("title") or o.get("text") or "")).strip()
            except Exception:
                txt = ln
            rows.append(txt)
    return rows

def build_batches():
    if ARGS.data_cache and os.path.exists(ARGS.data_cache):
        import numpy as np
        t0 = time.time()
        arr = np.load(ARGS.data_cache)
        tok = _load_tokenizer()
        all_ids = torch.from_numpy(arr.astype(np.int64))
        log(f"cache: cargado {arr.size/1e9:.2f}B tokens desde {ARGS.data_cache} "
            f"({time.time()-t0:.1f}s). tokenizer vocab={tok.vocab_size}")
        b = ARGS.batch_size
        n = (all_ids.numel() // (b * ARGS.seq_len)) * (b * ARGS.seq_len)
        if n == 0:
            raise RuntimeError("cache too small for a single batch")
        buf = all_ids[:n].view(b, -1)
        return tok, [buf[:, i*ARGS.seq_len:(i+1)*ARGS.seq_len]
                     for i in range(buf.size(1)//ARGS.seq_len)]
    tok = _load_tokenizer()
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
    if n == 0:
        raise RuntimeError("corpus too small for a single batch")
    buf = all_ids[:n].view(b, -1)
    return tok, [buf[:, i*ARGS.seq_len:(i+1)*ARGS.seq_len]
                 for i in range(buf.size(1)//ARGS.seq_len)]

def _load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ARGS.tokenizer, trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

# ---------------- checkpoint / resume (identico) ----------------
glob_model = None
glob_opt = None
STEPS_DONE = [0]
LAST_LOSS = [0.0]

def _save_checkpoint(tag):
    g = STEPS_DONE[0]
    ckpt = OUT / f"checkpoint-g{g}"
    ckpt.mkdir(parents=True, exist_ok=True)
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
    # FIX multi-nodo torchrun (2026-08-29): con srun + torchrun los workers heredan
    # SLURM_PROCID del srun task (0,1 por nodo) PERO RANK de torchrun es el GLOBAL
    # (0..world-1). RANK debe ganar SIEMPRE que torchrun lo defina; SLURM_PROCID solo
    # aplica en el modo vanilla (srun directo sin torchrun, donde RANK no existe).
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    world_size = world
    ddp = world_size > 1
    if ddp:
        import socket
        master_addr = socket.gethostname()
        try:
            nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
            if nodelist:
                import subprocess
                first = subprocess.run(
                    ["scontrol", "show", "hostname", nodelist],
                    capture_output=True, text=True, timeout=10).stdout.splitlines()[0].strip()
                if first:
                    master_addr = first
        except Exception:
            pass
        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", "29512")
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        nvis = torch.cuda.device_count()
        dev_idx = min(local_rank, max(0, nvis-1))
        torch.cuda.set_device(dev_idx)
        DEVICE = torch.device("cuda", dev_idx)
        log(f"DDP: rank={rank} local={local_rank} world={world_size} visible={nvis} dev={dev_idx} "
            f"sync_every={ARGS.sync_every} clip={ARGS.clip_grad or 'off'}")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok, batches_all = build_batches()
    if ddp:
        nb = len(batches_all)
        per = nb // world_size
        batches = batches_all[rank*per : (rank+1)*per] if rank < world_size-1 else batches_all[rank*per:]
        if len(batches)==0: batches = batches_all[:1]
    else:
        batches = batches_all
    ARGS.vocab = tok.vocab_size
    if rank==0: log(f"using vocab_size={ARGS.vocab} (from tokenizer)")
    glob_model = build_model().to(DEVICE)
    nparam = glob_model.n_params()
    if ARGS.n_experts > 1:
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
    # DDP: con el aux de balance tocando TODOS los expertos cada iteracion no hay
    # params 'unused' -> find_unused_parameters=False (sin deadlock).
    if ddp:
        glob_model = torch.nn.parallel.DistributedDataParallel(
            glob_model, device_ids=[dev_idx], find_unused_parameters=False)
    glob_model.zero_grad(set_to_none=True)
    MASK = ARGS.vocab
    n_masked = max(1, int((ARGS.seq_len//2) * ARGS.mask_p))
    nb = len(batches); it = 0
    # ---- post-localSGD: ciclo de acumulacion = grad_accum * sync_every ----
    acc = max(1, ARGS.grad_accum) * max(1, ARGS.sync_every)
    for step in range(STEPS_DONE[0], ARGS.max_steps):
        xb = batches[it % nb].to(DEVICE); it += 1
        xm = xb.clone(); head = xb.size(1)//2
        mp = torch.randperm(xb.size(1))[:n_masked]
        xm[:, mp] = ARGS.vocab
        out = glob_model(xm)
        loss = F.cross_entropy(out[:, mp].reshape(-1, ARGS.vocab),
                               xb[:, mp].reshape(-1))
        raw = glob_model.module if ddp else glob_model
        aux = sum(b.mlp.balance_loss(0.01) for b in raw.blocks)
        is_sync_step = ((step+1) % acc == 0)
        if ddp and ARGS.sync_every > 1 and not is_sync_step:
            # grad LOCAL sin all-reduce (DDP no_sync) -> comunicacion / sync_every
            with glob_model.no_sync():
                (loss/acc + aux).backward()
        else:
            (loss/acc + aux).backward()
            if is_sync_step:
                if ARGS.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(glob_model.parameters(), ARGS.clip_grad)
                glob_opt.step(); glob_opt.zero_grad(set_to_none=True)
        LAST_LOSS[0] = loss.item(); STEPS_DONE[0] = step+1
        if ddp: torch.distributed.barrier()
        if (not ddp or rank==0) and step % 10 == 0:
            log(f"step {step} loss {loss.item():.4f}" +
                (f" (sync)" if is_sync_step else ""))
        if (not ddp or rank==0) and step % 50 == 0: _save_checkpoint("step")
    if not ddp or rank==0: _save_checkpoint("final")
    if ddp: torch.distributed.destroy_process_group()
    if not ddp or rank==0: log("COMPLETE")

if __name__ == "__main__":
    main()