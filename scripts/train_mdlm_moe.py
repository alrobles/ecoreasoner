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
        self.register_buffer("_fcount", torch.zeros(n_experts), persistent=False)
        self._gate_probs = None
        self._probe_x = None
        self._tokens = 0
    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        g = torch.softmax(self.gate(flat).float(), dim=-1)
        gv, gi = g.topk(self.k, dim=-1)
        # registrar métricas aux. _gate_probs se conserva DIFERENCIABLE para que el
        # balance_loss pase grad real al router (load balancing). _fcount es stop-grad.
        if self.training:
            self._gate_probs = g
            self._fcount.zero_()
            self._fcount.scatter_add_(0, gi.reshape(-1), torch.ones(gi.numel(), device=gi.device))
            self._tokens = gi.numel()
            # Azada probe: elegimos hasta n_experts filas (detach) y en el aux las pasamos
            # por CADA experto -> unidad de peso de cada experto recibe grad siempre.
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
        """Aux cargada en grad REAL a TODO el bloque MoE, cada iteración:

        router_aux  = alpha*n*sum(f_e * P_e):  f_e = fracción ocupada de tokens (stop-grad),
                                              P_e = prob media de gate (DIFERENCIABLE).
                                              Equilibra el router -> evita colapso de carga.
        probe_aux   = probe_alpha * (1/n) * sum_e mean(experts_e(probe_x)^2):
                                              pasa un token por TODOS los expertos =>
                                              cada experto recibe grad real SIEMPRE.
                                              -> find_unused_parameters=False no da deadlock
                                                 y ningún experto queda 'no usado' en DDP.
        """
        if self._gate_probs is None or self._tokens == 0 or self._probe_x is None:
            return torch.zeros((), device=self.gate.weight.device)
        P = self._gate_probs.mean(0)                       # differentiable
        f = self._fcount.to(P.dtype) / max(self._tokens, 1)  # stop-grad (int scatter)
        router_aux = alpha * self.n * (f * P).sum()
        # ---- probe: gradito real a cada experto (anti 'unused' DDP) ----
        probe = torch.zeros((), device=P.device)
        n = self.n
        for e in range(n):
            ye = self.experts[e](self._probe_x)            # differentiable en w_e
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
                txt = (str(o.get("prompt","")) + "\n" + str(o.get("answer") or o.get("title") or o.get("text") or "")).strip()
            except Exception:
                txt = ln
            rows.append(txt)
    return rows

def build_batches():
    # ---------- carga desde cache (evita re-tokenizar) ----------
    if ARGS.data_cache and os.path.exists(ARGS.data_cache):
        import numpy as np
        t0 = time.time()
        arr = np.load(ARGS.data_cache)   # int32 plano: tokens concatenados
        tok = _load_tokenizer()
        # GUARDIA 2026-08-29: corpus v5 tenia 2 tokens 126082 (fuera de rango,
        # Embedding solo aguanta 0..vocab). Out-of-bounds en el Embedding -> CUDA
        # illegal memory access en el primer forward. Clip defensivo aqui.
        # OJO: usar tok.vocab_size (NO ARGS.vocab, default 32000, aqui sin actualizar).
        VB = tok.vocab_size
        if int(arr.max()) >= VB:
            nbad = int((arr >= VB).sum())
            log(f"GUARDIA: {nbad} tokens >= vocab({VB}) -> clamp a 0")
            arr = np.where(arr >= VB, 0, arr)
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

    # ---------- tokenizar en vivo (solo si NO hay cache) ----------
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
    # ESCRITURA ATOMICA (2026-08-29): el SIGUSR1 llega a los DOS ranks; con
    # torch.save directo al mismo path el archivo quedaba intercalado (corrupto).
    # tmp con PID unico (cada rank escribe su tmp; el rename es atomico) y
    # SOLO rank 0 guarda en SIGUSR1 (_handle_sig) -> un solo escritor por dir.
    pid = os.getpid()
    tmp_m = ckpt/f"model.pt.tmp.{pid}"; tmp_o = ckpt/f"optimizer.pt.tmp.{pid}"
    torch.save({"model": sd}, tmp_m)
    torch.save({"optimizer": glob_opt.state_dict()}, tmp_o)
    os.replace(tmp_m, ckpt/"model.pt")
    os.replace(tmp_o, ckpt/"optimizer.pt")
    (OUT/"state.json").write_text(json.dumps(
        {"step": g, "checkpoint": f"checkpoint-g{g}", "updated": time.time()}))
    (OUT/"progress.json").write_text(json.dumps(
        {"step": g, "loss": LAST_LOSS[0], "updated": time.time()}))
    ckpts = sorted(OUT.glob("checkpoint-g*"), key=lambda d: int(d.name.split("-g")[1]))
    for f in ckpts[:-2]:  # conserva 2: el actual + el de la ola previa (resume seguro)
        shutil.rmtree(f, ignore_errors=True)
    log(f"  checkpoint g{g} guardado")

def _try_load(ck, step):
    """Carga un checkpoint; devuelve True si OK. El checkpoint puede estar
    CORRUPTO (race SIGUSR1 de 2 ranks guardando al mismo dir, 2026-08-29):
    torch.load lanza -> saltar al siguiente integro."""
    try:
        glob_model.load_state_dict(torch.load(ck/"model.pt", map_location="cpu")["model"])
        glob_opt.load_state_dict(torch.load(ck/"optimizer.pt", map_location="cpu")["optimizer"])
        STEPS_DONE[0] = step
        log(f"Resumed {ck.name} (step {STEPS_DONE[0]})")
        return True
    except Exception as e:
        log(f"  checkpoint {ck.name} corrupto/incompleto ({type(e).__name__}) -> intentar previo")
        return False

def resume():
    """Prueba candidatos de MAS NUEVO a MAS ANTIGUO: primero el de state.json,
    luego el resto de checkpoint-g* por step desc. Salta los corruptos."""
    sf = OUT/"state.json"
    cands = []  # (ckpt_dir, step)
    st_cand = None; st_step = 0
    if sf.exists():
        try:
            st = json.loads(sf.read_text())
            c = OUT/st.get("checkpoint","")
            if c.exists() and (c/"model.pt").exists() and (c/"optimizer.pt").exists():
                st_cand, st_step = c, st.get("step",0)
        except Exception:
            pass
    for d in sorted(OUT.glob("checkpoint-g*"),
                    key=lambda x: int(x.name.split("-g")[1]), reverse=True):
        if (d/"model.pt").exists() and (d/"optimizer.pt").exists():
            cands.append((d, int(d.name.split("-g")[1])))
    if st_cand is not None:
        # estado.json primero (mas fiable); el loop cubre el resto
        if _try_load(st_cand, st_step):
            return
    for ck, step in cands:
        if st_cand is not None and ck == st_cand:
            continue  # ya probado
        if _try_load(ck, step):
            return

def _handle_sig(sig, frm):
    log("SIGUSR1 — guardando ola y saliendo 42")
    # SOLO rank 0 guarda (world>1): si los 2 ranks escribieran al mismo dir,
    # race de escritura -> checkpoint corrupto o FileNotFoundError (2026-08-29).
    r = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    w = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    if w <= 1 or r == 0:
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
        # MULTI-NODO: el master es el nodo del rank 0 (SLURM_JOB_NODELIST), NO
        # socket.gethostname() de cada rank (eso solo funciona single-nodo y
        # cuelga en multi-nodo -> TCPStore timeout). SLURM puede darnos el primer
        # hostname; si no, usamos el propio (single-nodo).
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
    # MoE: con el loss aux de balance tocando TODOS los expertos cada iteración,
    # no hay params 'unused' -> find_unused_parameters=False (evita deadlock).
    if ddp:
        glob_model = torch.nn.parallel.DistributedDataParallel(
            glob_model, device_ids=[dev_idx], find_unused_parameters=False)
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
        # aux loss de balance (Switch): toca TODOS los expertos cada iteración
        # -> evita params 'unused' en DDP (deadlock). Solo si n_experts>1.
        raw = glob_model.module if ddp else glob_model
        aux = sum(b.mlp.balance_loss(0.01) for b in raw.blocks)
        (loss/ARGS.grad_accum + aux).backward()
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
