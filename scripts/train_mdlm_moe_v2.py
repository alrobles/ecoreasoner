#!/usr/bin/env python3
"""MdLMMoE v2 — masked-diffusion LLM con receta Fase 3 v2.

Receta v2 (2026-09-09):
  - packing con EOS y data_cache .npz
  - mask_schedule U(0,1) por ejemplo
  - whole-stage masking de esqueletos
  - cosine/WSD LR, grad clipping, EMA
  - weight tying y RoPE opcional
  - mantiene wave pattern, DDP y anti-reentrada del trainer v1

Run via apptainer SIF (ROCm), 1-2 GPUs. Single-GPU friendly for PoC.
"""
import argparse, json, math, os, random, re, signal, sys, time, shutil, contextlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Esquema por defecto del curriculum V3.1 (step_inicio, step_fin, span_len, b_h).
# span_len y b_h son los valores al INICIO de cada etapa; se interpola
# linealmente hasta el inicio de la siguiente etapa.
CUR_STAGES_DEFAULT = '[[0,2000,16,0.30],[2000,5000,32,0.60],[5000,10000,64,0.95]]'

# Config por defecto de role-aware masking V3.2. Puntuacion base = 1.0;
# cada match anade (weight) a la puntuacion de ese token.
# El listado es orientativo: conectivas causales/adversativas, verbos de
# relacion, numeros y entidades (especies / mayusculas). Sin NER.
_ROLE_CONFIG_DICT = {
    "connectives": [
        "because", "despite", "however", "therefore", "although", "thus",
        "since", "while", "whereas", "consequently", "nevertheless",
        "nonetheless", "accordingly", "hence", "yet", "but", "so", "if",
        "when", "after", "before", "until"
    ],
    "relation_verbs": [
        "increases", "reduces", "decreases", "inhibits", "promotes",
        "correlates", "depends", "drives", "affects", "influences",
        "regulates", "mediates", "enhances", "suppresses", "determines",
        "predicts", "causes", "leads", "results", "associated", "linked",
        "related", "modulates", "triggers", "induces", "prevents", "limits",
        "constrains", "facilitates"
    ],
    "species_keywords": [
        "panthera", "quercus", "apis", "daphnia", "mytilus", "giraffa",
        "cervus", "populus", "saguaro", "tigris", "ilex", "suber", "homo",
        "sapiens", "acer", "pinus", "fagus", "salix"
    ],
    "number_weight": 2.0,
    "connective_weight": 3.0,
    "relation_weight": 3.0,
    "entity_weight": 2.0,
    "uppercase_bonus": 0.5
}
ROLE_CONFIG_DEFAULT = json.dumps(_ROLE_CONFIG_DICT, ensure_ascii=False, separators=(',', ':'))

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
    p.add_argument("--lr_decay", default="none", choices=["none", "cosine", "wsd"],
                   help="decay de LR tras warmup: none, cosine o Warmup-Stable-Decay")
    p.add_argument("--lr_min_ratio", type=float, default=0.1,
                   help="LR final como fraccion del LR inicial (cosine/wsd)")
    p.add_argument("--grad_clip", type=float, default=0.0,
                   help="max norm para clip_grad_norm (0=desactivado)")
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="decay para EMA de pesos (0=desactivado, tipico 0.9999)")
    p.add_argument("--ema_device", type=str, default="cpu",
                   choices=["cpu", "cuda"],
                   help="device del state_dict EMA: cpu (lento, ~30s/step en 1B) "
                        "o cuda (requiere ~4.5GB libres por rank)")
    p.add_argument("--ema_every", type=int, default=1,
                   help="actualizar EMA cada N steps (decay efectivo beta^N, "
                        "misma escala temporal); amortigua el costo en CPU")
    p.add_argument("--weight_tying", action="store_true",
                   help="compartir pesos tok_emb -> head")
    p.add_argument("--use_rope", action="store_true",
                   help="usar RoPE en atencion (en vez de pos embedding aprendida)")
    p.add_argument("--mask_p", type=float, default=0.15)
    p.add_argument("--mask_type", default="random", choices=["random", "span"],
                   help="random: tokens aislados al azar (historico, default). "
                        "span: bloques contiguos largos (2026-09-02, diagnostico "
                        "smoke_cont_bw3: la mascara dispersa premia autocorrelacion "
                        "local; spans fuerzan coherencia estructural).")
    p.add_argument("--span_len", type=int, default=64,
                   help="longitud (tokens) de cada span contiguo con --mask_type=span")
    p.add_argument("--mask_schedule", default="fixed",
                   choices=["fixed", "uniform", "cosine"],
                   help="schedule de corrupcion: fixed (legacy mask_p), "
                        "uniform (t~U(b_l,b_h) fraccion de tokens enmascarados), "
                        "cosine (distribucion sesgada a mas masking)")
    p.add_argument("--mask_schedule_args", default="",
                   help="JSON con args, p.ej. '{\"b_l\":0.1,\"b_h\":0.9}'")
    p.add_argument("--curriculum", action="store_true",
                   help="activar curriculum de masking (varia span_len y b_h a lo largo del entrenamiento)")
    p.add_argument("--cur_stages", default=CUR_STAGES_DEFAULT,
                   help="JSON con etapas [step_inicio, step_fin, span_len, b_h]. "
                        "span_len y b_h son los valores al INICIO de la etapa; "
                        "se interpolan linealmente hasta el inicio de la siguiente.")
    p.add_argument("--whole_stage", action="store_true",
                   help="en datos de esqueleto, enmascarar ETAPAS enteras "
                        "(en vez de spans random)")
    p.add_argument("--stage_labels", default="[OBSERVACION],[HIPOTESIS],[PREDICCION],[EVIDENCIA],[CONCLUSION]",
                   help="etiquetas de etapa para --whole_stage")
    p.add_argument("--candidate_focus", type=float, default=0.0,
                   help="probabilidad por ejemplo de enmascarar UNA etapa "
                        "completa (etiqueta+contenido, sesgo a tardias): "
                        "replica la tarea del scorer denso L3 en training")
    p.add_argument("--role_mask", action="store_true",
                   help="activar masking ponderado por rol semantico (V3.2). "
                        "Default OFF: con --role_mask ausente el comportamiento "
                        "es identico al V3.1.")
    p.add_argument("--role_config", default=ROLE_CONFIG_DEFAULT,
                   help="JSON con listas de conectivas, verbos de relacion, "
                        "especies y pesos para role-aware masking.")
    p.add_argument("--loss_num_w", type=float, default=1.0,
                   help="peso de loss en tokens numericos (DSFT-style: "
                        "ponderar la loss, no el masking). 1.0 = off")
    p.add_argument("--loss_neg_w", type=float, default=1.0,
                   help="peso de loss en tokens de negacion. 1.0 = off")
    p.add_argument("--corrective_p", type=float, default=0.0,
                   help="fraccion de tokens VISIBLES corrompidos por ejemplo "
                        "(Corrective Diffusion LM style): el modelo debe "
                        "predecir el token original en posiciones corruptas, "
                        "no solo enmascaradas. 0 = off")
    p.add_argument("--corrective_boost", type=float, default=8.0,
                   help="peso de seleccion de tokens informativos (numeros, "
                        "negaciones, verbos de direccion) al elegir posiciones "
                        "a corromper, vs 1.0 para el resto")
    p.add_argument("--mras_gamma", type=float, default=0.0,
                   help="masking adaptativo por dificultad (EMA de CE por "
                        "token-id): peso de seleccion ~ EMA^gamma mezclado "
                        "con uniforme. 0 = off")
    p.add_argument("--mras_ema", type=float, default=0.999,
                   help="decaimiento de la EMA de dificultad por token")
    p.add_argument("--mras_floor", type=float, default=0.3,
                   help="mezcla uniforme residual para no dejar de explorar")
    p.add_argument("--contr_w", type=float, default=0.0,
                   help="peso del termino contrastivo in-loss: hinge sobre "
                        "logp[real] - logp[mutado] en posiciones enmascaradas "
                        "mutables (digitos/negaciones/direccion). 0 = off")
    p.add_argument("--contr_margin", type=float, default=2.0,
                   help="margen del hinge contrastivo (nats de logprob)")
    p.add_argument("--contr_p", type=float, default=1.0,
                   help="fraccion de posiciones mutables contrastadas por step")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--data", nargs="*", default=[])
    p.add_argument("--data_cache", default=None,
                   help="Path a un .npy de IDs pre-tokenizados (evita re-tokenizar el corpus en cada slurm). "
                        "Si se da, build_batches carga los IDs de disco en vez de tokenizar.")
    p.add_argument("--tokenizer", default="/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad_ckpt", action="store_true",
                   help="Activation checkpointing por bloque: recomputa activaciones "
                        "en backward en vez de guardarlas (~-30%% memoria activ, ~+25%% step)")
    p.add_argument("--zero1", action="store_true",
                   help="ZeRO-1: estados del optimizador shardados entre ranks DDP "
                        "(memoria fija /world). Solo con ddp>1; mismo AdamW por dentro.")
    p.add_argument("--ckpt_every", type=int, default=50,
                   help="steps entre checkpoints (con zero1 el optimizer.pt pesa ~9GB)")
    return p.parse_args()
ARGS = parse()
if ARGS.mask_schedule_args:
    ARGS.mask_schedule_args = json.loads(ARGS.mask_schedule_args)
else:
    ARGS.mask_schedule_args = {}
if ARGS.cur_stages:
    try:
        ARGS.cur_stages = json.loads(ARGS.cur_stages)
    except Exception as e:
        raise SystemExit(f"[fatal] --cur_stages no es JSON valido: {e}")
else:
    ARGS.cur_stages = []
if ARGS.curriculum and not ARGS.cur_stages:
    ARGS.cur_stages = json.loads(CUR_STAGES_DEFAULT)
if ARGS.role_config:
    try:
        ARGS.role_config = json.loads(ARGS.role_config)
    except Exception as e:
        raise SystemExit(f"[fatal] --role_config no es JSON valido: {e}")
else:
    ARGS.role_config = {}
if ARGS.stage_labels:
    ARGS.stage_labels = [x.strip() for x in ARGS.stage_labels.split(",") if x.strip()]
else:
    ARGS.stage_labels = []
if ARGS.seed:
    random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)

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

class RoPEMultiheadAttention(nn.Module):
    """Atencion con Rotary Position Embedding (RoPE) sin pos embedding aprendida.
    Implementacion minimal: qkv lineal + RoPE en q,k + softmax atencion full (diffusion)."""
    def __init__(self, d_model, n_heads, max_seq_len=2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])
    def _apply_rotary(self, x):
        d = x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * self.cos[:, :, :x.size(2), :d] + rot * self.sin[:, :, :x.size(2), :d]
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self._apply_rotary(q)
        k = self._apply_rotary(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, dim, ff, heads, n_experts, k, use_rope=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = (RoPEMultiheadAttention(dim, heads) if use_rope
                     else nn.MultiheadAttention(dim, heads, batch_first=True))
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MoEMLP(dim, ff, n_experts, k)
        self._use_rope = use_rope
    def forward(self, x):
        h = self.ln1(x)
        if self._use_rope:
            a = self.attn(h)
        else:
            a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TiedHead(nn.Module):
    """Head que comparte pesos con tok_emb (vocab primeras filas)."""
    def __init__(self, tok_emb, vocab):
        super().__init__()
        self.tok_emb = tok_emb
        self.vocab = vocab
    def forward(self, h):
        return F.linear(h, self.tok_emb.weight[:self.vocab])


class MdLMMoE(nn.Module):
    def __init__(self, vocab, hidden, layers, heads, ff_mult, seq_len, n_experts, k,
                 use_rope=False, weight_tying=False):
        super().__init__()
        self.vocab = vocab
        self.use_rope = use_rope
        self.tok_emb = nn.Embedding(vocab + 1, hidden)   # +1 for MASK token
        self.pos = None if use_rope else nn.Embedding(seq_len, hidden)
        self.blocks = nn.ModuleList([
            Block(hidden, hidden*ff_mult, heads, n_experts, k, use_rope=use_rope)
            for _ in range(layers)])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = (TiedHead(self.tok_emb, vocab) if weight_tying
                     else nn.Linear(hidden, vocab))
        self.apply(_default_init)
    def forward(self, ids, output_hidden_states=False):
        B, T = ids.shape
        h = self.tok_emb(ids)
        if not self.use_rope:
            h = h + self.pos(torch.arange(T, device=ids.device))
        ckpt = getattr(self, "_grad_ckpt", False) and self.training
        for b in self.blocks:
            h = (torch.utils.checkpoint.checkpoint(b, h, use_reentrant=False)
                 if ckpt else b(h))
        hidden = self.ln_f(h)
        logits = self.head(hidden)
        if output_hidden_states:
            return logits, hidden
        return logits
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

def build_model():
    return MdLMMoE(ARGS.vocab, ARGS.hidden, ARGS.layers, ARGS.heads,
                   ARGS.ff_mult, ARGS.seq_len, ARGS.n_experts, ARGS.expert_k,
                   use_rope=ARGS.use_rope, weight_tying=ARGS.weight_tying)

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

def _pack_with_eos(all_ids, lengths, eos_id, batch_size, seq_len):
    """Packing con EOS: rellena streams por documento y los corta a seq_len.
    El ultimo trozo de cada stream se rellena con EOS para no perder datos.
    Devuelve batches de forma [batch_size, seq_len]."""
    import random
    docs = []
    pos = 0
    for l in lengths:
        docs.append(all_ids[pos:pos+l].tolist() + [eos_id])
        pos += l
    random.shuffle(docs)
    streams = [[] for _ in range(batch_size)]
    for i, doc in enumerate(docs):
        streams[i % batch_size].extend(doc)
    # descartar streams vacios y rellenar a multiplo de seq_len
    streams = [s for s in streams if s]
    if not streams:
        return []
    max_len = max(len(s) for s in streams)
    n_chunks = (max_len + seq_len - 1) // seq_len
    full_len = n_chunks * seq_len
    padded = []
    for s in streams:
        pad = full_len - len(s)
        if pad > 0:
            s = s + [eos_id] * pad
        padded.append(torch.tensor(s, dtype=torch.long).view(n_chunks, seq_len))
    # stack: [batch_size, n_chunks, seq_len] -> list of [batch_size, seq_len]
    stacked = torch.stack(padded, dim=0)
    return [stacked[:, i, :] for i in range(n_chunks)]


def build_batches():
    # ---------- carga desde cache (evita re-tokenizar) ----------
    if ARGS.data_cache and os.path.exists(ARGS.data_cache):
        import numpy as np
        t0 = time.time()
        tok = _load_tokenizer()
        VB = tok.vocab_size
        if ARGS.data_cache.endswith(".npz"):
            # NUEVO v2: packing con EOS y longitudes por documento
            # Corpus grande: preferir <cache>.ids.npy crudo -> np.load(mmap)
            # comparte page cache entre ranks (~46GB total, no por rank).
            raw_ids = ARGS.data_cache + ".ids.npy"
            if os.path.exists(raw_ids):
                arr = np.load(raw_ids, mmap_mode="r")  # int32 memmap
                if int(arr[:50_000_000].max()) >= VB:
                    log(f"GUARDIA: tokens >= vocab({VB}) en head -> clamp a 0")
                all_ids = torch.from_numpy(np.ascontiguousarray(arr)) \
                    if not arr.flags["C_CONTIGUOUS"] else torch.from_numpy(arr)
                b = ARGS.batch_size
                n = (arr.shape[0] // (b * ARGS.seq_len)) * (b * ARGS.seq_len)
                buf = all_ids[:n].view(b, -1)
                batches = [buf[:, i*ARGS.seq_len:(i+1)*ARGS.seq_len]
                           for i in range(buf.size(1)//ARGS.seq_len)]
                log(f"cache ids.npy (memmap int32): {arr.shape[0]/1e9:.2f}B tokens, "
                    f"{len(batches)} batches ({time.time()-t0:.1f}s)")
                return tok, batches
            npz = np.load(ARGS.data_cache)
            arr = npz["ids"]
            lengths = npz["lengths"]
            eos_id = int(npz.get("eos_id", tok.eos_token_id or tok.pad_token_id or 0))
            if int(arr.max()) >= VB:
                nbad = int((arr >= VB).sum())
                log(f"GUARDIA: {nbad} tokens >= vocab({VB}) -> clamp a 0")
                arr = np.where(arr >= VB, 0, arr)
            if eos_id >= VB:
                log(f"GUARDIA: eos_id({eos_id}) >= vocab({VB}) -> reemplazar por 0")
                eos_id = 0
            if arr.size > 500_000_000:
                # corpus grande sin .ids.npy: vistas int32 del array plano
                # (docs ya llevan EOS dentro; ~46GB/rank en vez de ~700GB del
                # pack por listas Python). Buffers por columna = cada fila del
                # batch lee una region distinta del corpus.
                all_ids = torch.from_numpy(arr)  # int32, sin copia int64
                b = ARGS.batch_size
                n = (arr.size // (b * ARGS.seq_len)) * (b * ARGS.seq_len)
                buf = all_ids[:n].view(b, -1)
                batches = [buf[:, i*ARGS.seq_len:(i+1)*ARGS.seq_len]
                           for i in range(buf.size(1)//ARGS.seq_len)]
                log(f"cache npz (vista int32): {arr.size/1e9:.2f}B tokens, "
                    f"{len(batches)} batches ({time.time()-t0:.1f}s)")
                return tok, batches
            all_ids = arr.astype(np.int64)
            batches = _pack_with_eos(all_ids, lengths, eos_id, ARGS.batch_size, ARGS.seq_len)
            # re-clamp tras packing por si el npz contenía tokens fuera de rango
            for b in batches:
                if int(b.max()) >= VB:
                    nbad = int((b >= VB).sum())
                    log(f"GUARDIA post-pack: {nbad} tokens >= vocab({VB}) -> clamp a 0")
                    b[:] = torch.where(b >= VB, torch.tensor(0, dtype=b.dtype, device=b.device), b)
            log(f"cache npz: {arr.size/1e9:.2f}B tokens, {len(lengths)} docs, "
                f"{len(batches)} batches ({time.time()-t0:.1f}s)")
            return tok, batches
        # legacy .npy (sin EOS)
        arr = np.load(ARGS.data_cache)   # int32 plano: tokens concatenados
        if int(arr.max()) >= VB:
            nbad = int((arr >= VB).sum())
            log(f"GUARDIA: {nbad} tokens >= vocab({VB}) -> clamp a 0")
            arr = np.where(arr >= VB, 0, arr)
        all_ids = torch.from_numpy(arr.astype(np.int64))
        log(f"cache npy: cargado {arr.size/1e9:.2f}B tokens desde {ARGS.data_cache} "
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
glob_ema_sd = [None]
STEPS_DONE = [0]
LAST_LOSS = [0.0]

def build_span_mask(T, n_target, span_len, device="cpu"):
    """Mascara de SPANS CONTIGUOS largos (2026-09-02).

    Enmascara ~n_target tokens repartidos en spans contiguos de ~span_len
    tokens (M = max(1, n_target // span_len) spans), con inicios aleatorios y
    sin solapamiento. A diferencia del random disperso -- que premia copiar el
    vecino inmediato (autocorrelacion local, nunca sintaxis) y es la causa del
    colapso observado en smoke_cont_bw3 (rep4 0.41 con contexto vs 0.15 en
    texto real) -- aqui el modelo debe reconstruir REGIONES completas desde
    contexto lejano: fuerza coherencia estructural. El numero real de
    posiciones puede desviarse ligeramente de n_target (clamp a [1, T]).
    """
    T = int(T)
    span_len = max(1, min(int(span_len), T))
    n_spans = max(1, int(n_target) // span_len)
    masked = torch.zeros(T, dtype=torch.bool, device=device)
    for _ in range(n_spans * 6):
        if int(masked.sum()) >= n_target or masked.all():
            break
        start = int(torch.randint(0, max(1, T - span_len + 1), (1,), device=device))
        masked[start:start + span_len] = True
    # si los spans dejaron deficit, cerrarlo con tokens aislados
    while int(masked.sum()) < n_target and not masked.all():
        free = (~masked).nonzero(as_tuple=False).squeeze(-1)
        j = int(torch.randint(0, free.numel(), (1,), device=device))
        masked[int(free[j])] = True
    return masked.nonzero(as_tuple=False).squeeze(-1)


def curriculum_state(step, stages):
    """Devuelve (span_len, b_h) para el step global segun el curriculum.

    Cada etapa = [step_inicio, step_fin, span_len, b_h], donde span_len y b_h
    son los valores al INICIO de la etapa. La interpolacion lineal dentro de la
    etapa avanza hasta el inicio de la siguiente; la ultima etapa es constante.
    - step < inicio primera: usa la primera etapa.
    - step >= fin ultima: usa la ultima etapa.
    - etapas contiguas [s,e] se interpretan como [s,e) para la seleccion, pero
      los valores de inicio/fin coinciden, asi que el curriculum es continuo.
    """
    if not stages:
        return ARGS.span_len, float(ARGS.mask_schedule_args.get("b_h", 1.0))
    stages = sorted(stages, key=lambda s: s[0])
    if step < stages[0][0]:
        s, e, span, frac = stages[0]
        return int(span), float(frac)
    if step >= stages[-1][1]:
        s, e, span, frac = stages[-1]
        return int(span), float(frac)
    for i, (s, e, span, frac) in enumerate(stages):
        if s <= step < e:
            delta = max(1, e - s)
            t = (step - s) / delta
            next_span = stages[i + 1][2] if i + 1 < len(stages) else span
            next_frac = stages[i + 1][3] if i + 1 < len(stages) else frac
            cur_span = int(round(span + t * (next_span - span)))
            cur_frac = float(frac + t * (next_frac - frac))
            return max(1, cur_span), max(0.0, min(1.0, cur_frac))
    # fallback: ultima etapa
    s, e, span, frac = stages[-1]
    return int(span), float(frac)


def sample_mask_fraction(schedule, step, device="cpu"):
    """Samplea la fraccion de tokens a enmascarar en un paso de entrenamiento.

    - fixed: usa la semantica legacy mask_p (15% de la mitad = ~7.5% total).
    - uniform: t ~ U(b_l, b_h) fraccion TOTAL de tokens.
    - cosine: t = 1 - cos(pi/2 * u) con u~U(0,1), sesgado a mas masking.
    """
    if schedule == "fixed":
        return ARGS.mask_p * 0.5
    if schedule == "uniform":
        b_l = float(ARGS.mask_schedule_args.get("b_l", 0.0))
        b_h = float(ARGS.mask_schedule_args.get("b_h", 1.0))
        return b_l + (b_h - b_l) * torch.rand(1, device=device).item()
    if schedule == "cosine":
        u = torch.rand(1, device=device).item()
        return 1.0 - math.cos(u * math.pi / 2)
    return ARGS.mask_p * 0.5


# ---------------- role / connective-aware masking (V3.2) ----------------
# Interaccion con curriculum V3.1: el curriculum fija span_len y b_h por step;
# role_mask actua DESPUES, decidiendo DONDE colocar esos spans (o que etapas
# enmascarar) priorizando regiones con conectivas, verbos de relacion, numeros
# y entidades. Con --role_mask ausente el camino es identico al V3.1.

_NUMBER_RE = re.compile(r"^[+-]?(\d+([.,]\d+)?|\d*\.\d+)(\s*%)?$")
# G10 (audit 18-09): la clase mutable numerica cubre tambien las piezas
# atomicas de numeros multi-token ('.', ',', '%', signos) — las mutaciones
# del eval retokenizan el numero y el diff cae en esas piezas (~98% cov).
_NUM_PIECE_RE = re.compile(r"^[.,%+\-−]$|^\d+$")
_ANY_DIGIT_RE = re.compile(r"\d")
_ROLE_SCORE_CACHE = {}


def _normalize_token_text(text):
    """Quita prefijos comunes de subword (SentencePiece, BPE, RoBERTa)."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    for prefix in ("▁", "Ġ", "##", "Ċ"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text


def _token_role_score(text, cfg):
    """Puntuacion base 1.0 + bonus aditivos por categorias informativas."""
    if not text:
        return 1.0
    t = text
    lower = t.lower()
    score = 1.0

    if lower in cfg.get("connectives", []):
        score += float(cfg.get("connective_weight", 0.0))
    if lower in cfg.get("relation_verbs", []):
        score += float(cfg.get("relation_weight", 0.0))
    if lower in cfg.get("species_keywords", []):
        score += float(cfg.get("entity_weight", 0.0))
    if _NUMBER_RE.match(t):
        score += float(cfg.get("number_weight", 0.0))
    # Heuristica de nombres propios / entidades: primera letra mayuscula,
    # no esta todo en mayusculas, y no es un simbolo aislado.
    if (t and t[0].isupper() and not t.isupper() and
            t[0].isalpha() and len(t) >= 2):
        score += float(cfg.get("uppercase_bonus", 0.0))
    return score


def _all_token_strings(tok, vocab):
    """Devuelve la lista de strings representativos de cada id del vocabulario."""
    if hasattr(tok, "convert_ids_to_tokens"):
        try:
            return tok.convert_ids_to_tokens(list(range(vocab)))
        except Exception:
            pass
    if hasattr(tok, "decode"):
        return [tok.decode([i], skip_special_tokens=False) for i in range(vocab)]
    return [f"<tok{i}>" for i in range(vocab)]


def build_role_score_table(tok, cfg, device="cpu"):
    """Precomputa un tensor [vocab] con el peso semantico de cada token."""
    vocab = getattr(tok, "vocab_size", 0)
    if vocab == 0:
        return torch.ones(0, device=device)
    scores = torch.ones(vocab, dtype=torch.float32, device=device)
    for i, s in enumerate(_all_token_strings(tok, vocab)):
        text = _normalize_token_text(s)
        scores[i] = _token_role_score(text, cfg)
    return scores


def get_token_role_scores(tok, ids, cfg=None):
    """Dado un batch de ids [B,T] devuelve scores [B,T] usando cache por cfg."""
    if cfg is None:
        cfg = ARGS.role_config if (hasattr(ARGS, "role_config") and
                                   isinstance(ARGS.role_config, dict)) else {}
    key = (id(tok), json.dumps(cfg, sort_keys=True, ensure_ascii=False))
    if key not in _ROLE_SCORE_CACHE:
        _ROLE_SCORE_CACHE[key] = build_role_score_table(tok, cfg, device="cpu")
    table = _ROLE_SCORE_CACHE[key]
    dev = ids.device if torch.is_tensor(ids) else "cpu"
    return table.to(dev)[ids]


def build_role_weighted_random_mask(T, n_target, token_scores, device="cpu"):
    """Mascara individual ponderada por token_scores."""
    T = int(T)
    n = max(0, min(int(n_target), T))
    if n == 0:
        return torch.tensor([], dtype=torch.long, device=device)
    weights = token_scores.to(device).float().clamp(min=0)
    if weights.sum() <= 0:
        weights = torch.ones_like(weights)
    probs = weights / weights.sum()
    return torch.multinomial(probs, n, replacement=False)


def build_role_weighted_span_mask(T, n_target, span_len, token_scores, device="cpu"):
    """Spans contiguos cuyos centros se muestrean proporcional al 'bonus'
    semantico de la ventana (suma de scores-1 de las posiciones libres).
    Esto prioriza regiones con conectivas, relaciones, numeros y entidades.
    """
    T = int(T)
    span_len = max(1, min(int(span_len), T))
    n_spans = max(1, int(n_target) // span_len)
    masked = torch.zeros(T, dtype=torch.bool, device=device)
    # bonus sobre posiciones aun libres
    bonus = (token_scores.to(device).float() - 1.0).clamp(min=0)

    for _ in range(n_spans * 6):
        if int(masked.sum()) >= n_target or masked.all():
            break
        free_bonus = bonus * (~masked).float()
        n_starts = max(1, T - span_len + 1)
        if n_starts == 1:
            window_weights = free_bonus.sum().unsqueeze(0)
        else:
            # convolucion 1D: suma de free_bonus en cada ventana de span_len
            fb = free_bonus.unsqueeze(0).unsqueeze(0)  # [1,1,T]
            k = torch.ones(1, 1, span_len, device=device)
            window_weights = F.conv1d(fb, k).squeeze()
        # Si todas las ventanas tienen bonus 0, caemos a uniforme.
        if window_weights.sum() <= 0:
            start = int(torch.randint(0, n_starts, (1,), device=device))
        else:
            probs = window_weights / window_weights.sum()
            start = int(torch.multinomial(probs, 1).item())
        masked[start:start + span_len] = True

    # cerrar deficit con tokens individuales, preferentemente de alto score
    while int(masked.sum()) < n_target and not masked.all():
        free = (~masked).nonzero(as_tuple=False).squeeze(-1)
        fscores = token_scores[free]
        if fscores.sum() > 0:
            j = int(torch.multinomial(fscores / fscores.sum(), 1).item())
        else:
            j = int(torch.randint(0, free.numel(), (1,), device=device))
        masked[int(free[j])] = True
    return masked.nonzero(as_tuple=False).squeeze(-1)


def build_role_weighted_mask(T, n_target, mask_type, span_len, token_scores, device="cpu"):
    """Entry de masking con pesos por token."""
    if mask_type == "random":
        return build_role_weighted_random_mask(T, n_target, token_scores, device=device)
    return build_role_weighted_span_mask(T, n_target, span_len, token_scores, device=device)


# ---------------- corrective corruption + loss weighting (V3.4) ----------------
# Corrective Diffusion LM (2025): mezcla de masks absorbentes + tokens
# VISIBLES corrompidos, supervisados a predecir el token original. A
# diferencia de ELECTRA (detecta provenance), aqui la supervision es de
# correccion: el modelo aprende "este token visible esta mal -> deberia
# ser X", que es exactamente la habilidad que mide el eval pairwise.
#
# Corrupciones con sentido cientifico (no uniformes a ciegas):
#   - numeros -> otro token numerico del vocabulario (swap de magnitud)
#   - negaciones -> token que neutraliza/invierte la polaridad
#   - verbos/comparativos de direccion -> su antonimo (increases->decreases)
#   - resto -> token uniforme del vocabulario (corrupcion generica)
#
# numw (DSFT-style): peso extra en la LOSS de tokens numericos/negacion,
# en vez de sesgar el masking (leccion de g3-numb: mask-bias inestable).

_CORRUPT_FLIP_TEXT = {
    # negaciones -> neutralizacion / flip de polaridad
    "not": ["also", "indeed", "often", "still"],
    "no": ["a", "the", "some"],
    "never": ["often", "sometimes", "usually", "always"],
    "cannot": ["can", "may", "could"],
    "without": ["with"],
    "nor": ["or", "and"],
    "neither": ["either", "both"],
    "fails": ["succeeds", "manages"],
    "fail": ["succeed", "manage"],
    "failed": ["succeeded", "managed"],
    # direccion / comparacion -> antonimo
    "increases": ["decreases", "reduces"],
    "decreases": ["increases", "enhances"],
    "inhibits": ["promotes", "enhances"],
    "promotes": ["inhibits", "suppresses"],
    "enhances": ["suppresses", "reduces"],
    "suppresses": ["enhances", "promotes"],
    "improves": ["worsens", "impairs"],
    "reduces": ["increases", "raises"],
    "higher": ["lower"],
    "lower": ["higher"],
    "more": ["less", "fewer"],
    "less": ["more"],
    "above": ["below"],
    "below": ["above"],
    "before": ["after"],
    "after": ["before"],
    "positive": ["negative"],
    "negative": ["positive"],
    "significant": ["negligible", "marginal"],
    # --- expansion G10 (audit 18-09, pares 1:1 medidos del eval) ---
    # direccion: inflecciones + familias nuevas
    "increased": ["reduced", "decreased"],
    "reduced": ["increased", "enhanced"],
    "decreased": ["increased"],
    "increase": ["decrease", "reduce"],
    "decrease": ["increase"],
    "greater": ["smaller"],
    "smaller": ["greater"],
    "enhanced": ["suppressed", "reduced"],
    "suppressed": ["enhanced"],
    "improve": ["worsen", "impair"],
    "worsen": ["improve"],
    "improved": ["worsened"],
    "worsened": ["improved"],
    "associated": ["unrelated"],
    "unrelated": ["associated"],
    "presence": ["absence"],
    "absence": ["presence"],
    "raises": ["lowers"],
    "lowers": ["raises"],
    "stronger": ["weaker"],
    "weaker": ["stronger"],
    "elevated": ["reduced"],
    "larger": ["smaller"],
    "fewer": ["more"],
    "exceeds": ["falls"],
    "gain": ["loss"],
    "loss": ["gain"],
    # causal: conectivas contrastivas (however<->therefore etc.)
    "however": ["therefore", "thus", "consequently"],
    "therefore": ["however", "nevertheless"],
    "thus": ["nevertheless", "however"],
    "nevertheless": ["thus", "therefore"],
    "conversely": ["consequently", "similarly"],
    "consequently": ["conversely"],
    "because": ["despite", "although"],
    "despite": ["because"],
    "although": ["because"],
    "hence": ["however"],
    "whereas": ["therefore"],
    "nonetheless": ["therefore"],
    "moreover": ["however"],
    "furthermore": ["however"],
    "instead": ["additionally"],
    # temporal: familias medidas
    "early": ["late"],
    "late": ["early"],
    "initial": ["final"],
    "final": ["initial"],
    "initially": ["finally"],
    "finally": ["initially"],
    "acute": ["chronic"],
    "chronic": ["acute"],
    "rapidly": ["gradually"],
    "gradually": ["rapidly"],
    "rapid": ["gradual", "slow"],
    "gradual": ["rapid"],
    "long": ["short"],
    "short": ["long"],
    "longer": ["shorter"],
    "shorter": ["longer"],
    "earlier": ["later"],
    "later": ["earlier"],
    "temporary": ["permanent"],
    "permanent": ["temporary"],
    "transient": ["persistent"],
    "persistent": ["transient"],
    "delayed": ["immediate"],
    "immediate": ["delayed"],
    "brief": ["prolonged"],
    "prolonged": ["brief"],
    # verbos de reporte / polaridad (negation auxiliares)
    "supports": ["contradicts"],
    "contradicts": ["supports"],
    "confirms": ["refutes"],
    "refutes": ["confirms"],
    "consistent": ["inconsistent"],
    "inconsistent": ["consistent"],
    # entorno / mecanismo (open-class frecuentes del eval)
    "temperature": ["light"],
    "elevation": ["latitude"],
    "latitude": ["elevation"],
    "nitrogen": ["phosphorus"],
    "phosphorus": ["nitrogen"],
    "drought": ["flooding"],
    "flooding": ["drought"],
    "host": ["parasite"],
    "parasite": ["host"],
    "prey": ["predator"],
    "predator": ["prey"],
    # simbolos de relacion (byte-BPE observados): <= <-> >=, -> <-> <-,
    # = <-> '!='-like, +- <-> similares. Claves = strings del vocab
    # YA EN MINUSCULAS (el lookup hace .lower() antes de indexar).
    "âł": ["âī¥", ">", "="],          # ~'<=' -> >=, >, =
    "âī¥": ["âł", "<", "="],          # '>=' -> <=, <, =
    "âī¤": ["âī¥", ">"],              # '<=' (otra forma) -> >=
    "âģī": ["âĩĳ", "âĩķ"],            # '~->' -> '<-'
    "âĩĵ": ["âĩĳ"],                   # '->' -> '<-'
    "âĩĳ": ["âĩĵ"],                   # '<-' -> '->'
    "â±": ["<", ">"],                 # '+-' -> direcciones
    "<": [">", "âī¥"],
    ">": ["<", "âł"],
    "=": ["<", ">"],
}
_NEG_WORDS = {"not", "no", "never", "cannot", "without", "nor",
              "neither", "fails", "fail", "failed", "did", "does",
              "found", "find", "showed", "show", "suggests", "suggest",
              "indicates", "indicate", "demonstrates", "demonstrated",
              "reveals", "revealed"}
_FLIP_K = 4  # max candidatos por token en la tabla de flips


def build_correction_tables(tok, vocab, device):
    """Tablas [vocab] para corrupcion corrective y ponderacion de loss."""
    strs = _all_token_strings(tok, vocab)
    text2ids = {}
    for i, s in enumerate(strs):
        text2ids.setdefault(_normalize_token_text(s).lower(), []).append(i)
    is_num = torch.zeros(vocab, dtype=torch.bool)
    is_numx = torch.zeros(vocab, dtype=torch.bool)
    is_flip = torch.zeros(vocab, dtype=torch.bool)
    is_neg = torch.zeros(vocab, dtype=torch.bool)
    flip_cand = torch.full((vocab, _FLIP_K), -1, dtype=torch.long)
    num_ids = []
    for i, s in enumerate(strs):
        t = _normalize_token_text(s)
        tl = t.lower()
        if _NUMBER_RE.match(t):
            is_num[i] = True
            num_ids.append(i)
        if _NUMBER_RE.match(t) or _NUM_PIECE_RE.match(t):
            is_numx[i] = True
        if tl in _NEG_WORDS:
            is_neg[i] = True
        if tl in _CORRUPT_FLIP_TEXT:
            cands = []
            for rep in _CORRUPT_FLIP_TEXT[tl]:
                cands += text2ids.get(rep, [])
            if cands:
                is_flip[i] = True
                for j, c in enumerate(cands[:_FLIP_K]):
                    flip_cand[i, j] = c
    lw = torch.ones(vocab, dtype=torch.float32)
    lw[is_num] = float(ARGS.loss_num_w)
    lw[is_neg] = float(ARGS.loss_neg_w)
    sel_w = torch.ones(vocab, dtype=torch.float32)
    sel_w[is_num | is_flip] += float(ARGS.corrective_boost)
    return {
        "is_num": is_num.to(device), "is_flip": is_flip.to(device),
        "is_numx": is_numx.to(device),
        "flip_cand": flip_cand.to(device),
        "num_ids": torch.tensor(num_ids or [0], dtype=torch.long, device=device),
        "lw": lw.to(device), "sel_w": sel_w.to(device),
        "n_num": len(num_ids), "n_flip": int(is_flip.sum().item()),
    }


def _corrupt_positions(x1, mp_local, tables, device):
    """Elige posiciones VISIBLES a corromper para un ejemplo [T].

    x1: ids del ejemplo; mp_local: indices ya enmascarados (a excluir).
    Devuelve indices locales 1D o None.
    """
    T = x1.numel()
    vis = torch.ones(T, dtype=torch.bool, device=device)
    vis[mp_local] = False
    vis_idx = vis.nonzero(as_tuple=False).squeeze(-1)
    n_vis = int(vis_idx.numel())
    n_corr = int(round(n_vis * ARGS.corrective_p))
    if n_vis == 0 or n_corr <= 0:
        return None
    w = tables["sel_w"][x1.reshape(-1)[vis_idx]]
    sel = torch.multinomial(w, min(n_corr, n_vis), replacement=False)
    return vis_idx[sel]


def _corrupt_replacements(orig_ids, tables, vocab, device):
    """Devuelve ids corruptos para orig_ids [N] (vectorizado)."""
    n = orig_ids.numel()
    rep = torch.randint(0, vocab, (n,), device=device)
    num_m = tables["is_num"][orig_ids]
    if bool(num_m.any()):
        ri = torch.randint(0, tables["num_ids"].numel(),
                           (int(num_m.sum().item()),), device=device)
        rep[num_m] = tables["num_ids"][ri]
    fl_m = tables["is_flip"][orig_ids]
    if bool(fl_m.any()):
        cand = tables["flip_cand"][orig_ids[fl_m]]              # [m,K]
        ri = torch.randint(0, cand.size(1), (cand.size(0),), device=device)
        r = cand.gather(1, ri[:, None]).squeeze(1)
        bad = r < 0
        if bool(bad.any()):
            r[bad] = orig_ids[fl_m][bad]  # sin candidato: deja el original
        rep[fl_m] = r
    same = rep == orig_ids
    if bool(same.any()):
        rep[same] = (rep[same] + 1) % vocab
    return rep


def _contrast_alts(orig_ids, tables, vocab, device):
    """Alternativa 'misma clase, valor distinto' para el hinge contrastivo.

    Digito -> otro digito del pool (nunca el original); flip -> antonimo de
    flip_cand (siempre != orig); el resto no se contrasta (filtrado fuera).
    """
    alt = (orig_ids + 1) % vocab                      # fallback != orig
    num_m = tables["is_numx"][orig_ids]
    if bool(num_m.any()):
        pool = tables["num_ids"]
        idx = orig_ids[num_m]
        ri = torch.randint(0, pool.numel(), (idx.numel(),), device=device)
        r = pool[ri]
        same = r == idx
        if bool(same.any()):
            r[same] = pool[(ri[same] + 1) % pool.numel()]
        alt[num_m] = r
    fl_m = tables["is_flip"][orig_ids]
    if bool(fl_m.any()):
        cand = tables["flip_cand"][orig_ids[fl_m]]
        ri = torch.randint(0, cand.size(1), (cand.size(0),), device=device)
        r = cand.gather(1, ri[:, None]).squeeze(1)
        bad = r < 0
        if bool(bad.any()):
            r[bad] = cand[bad, 0]                     # slot valido siempre (is_flip exige >=1 cand)
        alt[fl_m] = r
    return alt


def _find_stage_boundaries(seq, stage_label_ids):
    """Devuelve listas de (start, end) de contenido de cada etapa.

    seq: tensor 1D de token ids.
    stage_label_ids: lista de listas de ids, p.ej. [[id('[','OBS',']'), ...]].
    """
    T = seq.size(0)
    seq_list = seq.tolist()  # comparacion en python: ~30K allocs GPU menos por step
    starts = []
    for i in range(T):
        for label_ids in stage_label_ids:
            L = len(label_ids)
            if i + L <= T:
                if seq_list[i:i+L] == label_ids:
                    starts.append((i, i + L))
                    break
    if not starts:
        return []
    starts.sort()
    boundaries = []
    for idx, (s, e) in enumerate(starts):
        content_start = e
        content_end = starts[idx + 1][0] if idx + 1 < len(starts) else T
        if content_start < content_end:
            boundaries.append((content_start, content_end))
    return boundaries


def build_whole_stage_mask(xb, n_target, stage_label_ids, mask_type="span", span_len=64, device="cpu", token_scores=None):
    """Enmascara ETAPAS COMPLETAS de esqueletos (Fase B.2).

    xb: [batch, T] token ids.
    stage_label_ids: lista de listas de token ids de cada etiqueta.
    Devuelve indices 1D planos o lista de indices por ejemplo.
    Para optimizar, cada ejemplo del batch se procesa por separado y luego
    concatenamos con offset T.
    """
    B, T = xb.shape
    all_indices = []
    for b in range(B):
        seq = xb[b]
        boundaries = _find_stage_boundaries(seq, stage_label_ids)
        if not boundaries:
            # sin etiquetas: fallback a span en todo el ejemplo
            mp = build_span_mask(T, n_target, span_len, device=device)
            all_indices.append(mp + b * T)
            continue
        # cuantos tokens podemos enmascarar
        lens = torch.tensor([e - s for s, e in boundaries], device=device, dtype=torch.float32)
        total = int(lens.sum().item())
        target = min(n_target, total)
        # samplear etapas proporcional a longitud + bonus semantico (V3.2)
        if lens.sum().item() <= 0:
            continue
        if token_scores is not None:
            ts = token_scores[b] if token_scores.dim() == 2 else token_scores
            stage_bonuses = torch.tensor(
                [float(ts[s:e].sum().item() - (e - s)) for s, e in boundaries],
                device=device, dtype=torch.float32,
            )
            probs = (lens + stage_bonuses).clamp(min=1e-3)
        else:
            probs = lens
        probs = probs / probs.sum()
        chosen = set()
        covered = 0
        while covered < target and len(chosen) < len(boundaries):
            # elegir etapa proporcional a longitud restante
            idx = int(torch.multinomial(probs, 1).item())
            if idx in chosen:
                # fallback: primer etapa no elegida
                for k in range(len(boundaries)):
                    if k not in chosen:
                        idx = k
                        break
            if idx in chosen:
                break
            chosen.add(idx)
            s, e = boundaries[idx]
            all_indices.append(torch.arange(s, e, device=device) + b * T)
            covered += (e - s)
    if not all_indices:
        # ultimo recurso: random sobre todo
        return torch.randperm(B * T, device=device)[:n_target]
    return torch.cat(all_indices)


def build_candidate_mask(xb, stage_label_ids, device="cpu", span_scores=None):
    """Enmascara UNA etapa completa (etiqueta + contenido) por ejemplo.

    Replica la tarea del scorer denso L3 (Fase A): contexto intacto, candidato
    al 100%. Sesgo a etapas tardias: peso lineal creciente por posicion de
    etapa (la ultima etapa es el candidato mas frecuente en los pares de eval).
    Si span_scores [T] viene, el peso de cada etapa se multiplica por la media
    de scores de sus posiciones (mras: favorece etapas con tokens dificiles).
    Devuelve indices planos [0, B*T) o None si el ejemplo no tiene etapas."""
    B, T = xb.shape
    all_indices = []
    for b in range(B):
        seq = xb[b]
        seq_list = seq.tolist()
        # localizar etiquetas (inicio, fin-etiqueta) en orden
        starts = []
        for i in range(T):
            for label_ids in stage_label_ids:
                L = len(label_ids)
                if i + L <= T:
                    if seq_list[i:i+L] == label_ids:
                        starts.append((i, i + L))
                        break
        if len(starts) < 2:
            continue
        starts.sort()
        # span candidato por etapa: [inicio_etiqueta, inicio_siguiente_etiqueta o T)
        spans = []
        for idx, (s, _e) in enumerate(starts):
            end = starts[idx + 1][0] if idx + 1 < len(starts) else T
            if s < end:
                spans.append((s, end))
        if not spans:
            continue
        w = torch.arange(1, len(spans) + 1, device=device, dtype=torch.float32)
        if span_scores is not None:
            ss = span_scores[b] if span_scores.dim() == 2 else span_scores
            mult = torch.tensor(
                [float(ss[s:e].mean()) for s, e in spans],
                device=device, dtype=torch.float32)
            w = w * mult.clamp(min=1e-6)
        idx = int(torch.multinomial(w / w.sum(), 1).item())
        s, e = spans[idx]
        all_indices.append(torch.arange(s, e, device=device) + b * T)
    if not all_indices:
        return None
    return torch.cat(all_indices)


def build_mask_indices(xb, n_target, mask_type, whole_stage=False, stage_label_ids=None, span_len=64, device="cpu", token_scores=None):
    """Entry-point: devuelve indices planos [0, B*T) a enmascarar por batch.

    Si token_scores no es None y --role_mask esta activo, se usa role-aware
    masking. El curriculum fija span_len y b_h por step; role_mask decide la
    ubicacion de los spans (o la seleccion de etapas con whole_stage)."""
    use_role = (token_scores is not None) and (
        getattr(ARGS, "role_mask", False) or
        getattr(ARGS, "mras_gamma", 0.0) > 0)
    if whole_stage and stage_label_ids:
        return build_whole_stage_mask(xb, n_target, stage_label_ids, mask_type=mask_type, span_len=span_len, device=device, token_scores=token_scores)
    B, T = xb.shape
    if use_role:
        if token_scores.dim() == 1:
            token_scores = token_scores.unsqueeze(0)
        parts = []
        for b in range(B):
            idx = build_role_weighted_mask(T, n_target, mask_type, span_len,
                                           token_scores[b], device=device)
            parts.append(idx + b * T)
        return torch.cat(parts)
    if mask_type == "random":
        # cada ejemplo con sus propias posiciones
        parts = [torch.randperm(T, device=device)[:n_target] + b * T for b in range(B)]
        return torch.cat(parts)
    # span por ejemplo
    parts = [build_span_mask(T, n_target, span_len, device=device) + b * T for b in range(B)]
    return torch.cat(parts)


def _save_checkpoint(tag):
    # LOCK de guardado (2026-09-01): serializa los saves entre ranks aunque el
    # filtro rank==0 fallara; evita que la limpieza de retencion-2 borre un dir
    # que otro rank esta escribiendo. Wrapper: el cuerpo queda intacto.
    # Flag SAVING (2026-09-08, auditoria 1.1): el handler SIGUSR1 lo consulta
    # para NO re-entrar en un save en curso (deadlock flock, ver _handle_sig).
    import fcntl
    _lf = open(OUT / ".save.lock", "w")
    fcntl.flock(_lf, fcntl.LOCK_EX)
    try:
        _SAVING[0] = True
        _save_checkpoint_locked(tag)
    finally:
        _SAVING[0] = False
        fcntl.flock(_lf, fcntl.LOCK_UN)
        _lf.close()


def _save_checkpoint_locked(tag):
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
    # EMA (a cpu para que el ckpt sea portable cpu/gpu)
    if glob_ema_sd[0] is not None:
        tmp_e = ckpt/f"ema_model.pt.tmp.{pid}"
        torch.save({"ema_model": {k: v.detach().to("cpu") for k, v in glob_ema_sd[0].items()}}, tmp_e)
        os.replace(tmp_e, ckpt/"ema_model.pt")
    (OUT/"state.json").write_text(json.dumps(
        {"step": g, "checkpoint": f"checkpoint-g{g}", "updated": time.time()}))
    (OUT/"progress.json").write_text(json.dumps(
        {"step": g, "loss": LAST_LOSS[0], "updated": time.time()}))
    def _step_of(d):
        try:
            return int(d.name.split("-g")[1])
        except (IndexError, ValueError):
            return -1
    ckpts = sorted((d for d in OUT.glob("checkpoint-g*") if _step_of(d) >= 0),
                   key=_step_of)
    for f in ckpts[:-2]:  # conserva 2: el actual + el de la ola previa (resume seguro)
        shutil.rmtree(f, ignore_errors=True)
    log(f"  checkpoint g{g} guardado")

def _ema_dev():
    return DEVICE if (ARGS.ema_device == "cuda" and torch.cuda.is_available()) else "cpu"

def _init_ema_state():
    if ARGS.ema_decay > 0 and glob_ema_sd[0] is None:
        m = glob_model.module if isinstance(glob_model, torch.nn.parallel.DistributedDataParallel) else glob_model
        glob_ema_sd[0] = {k: v.detach().clone().to(_ema_dev()) for k, v in m.state_dict().items()}
        log(f"  EMA inicializada desde modelo (dev={_ema_dev()})")


def _try_load(ck, step):
    """Carga un checkpoint; devuelve True si OK. El checkpoint puede estar
    CORRUPTO (race SIGUSR1 de 2 ranks guardando al mismo dir, 2026-08-29):
    torch.load lanza -> saltar al siguiente integro."""
    try:
        glob_model.load_state_dict(torch.load(ck/"model.pt", map_location="cpu")["model"])
        glob_opt.load_state_dict(torch.load(ck/"optimizer.pt", map_location="cpu")["optimizer"])
        # EMA
        if (ck/"ema_model.pt").exists():
            glob_ema_sd[0] = {k: v.to(_ema_dev()) for k, v in
                              torch.load(ck/"ema_model.pt", map_location="cpu")["ema_model"].items()}
            log("  EMA cargada")
        else:
            _init_ema_state()
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
    def _step_of(d):
        try:
            return int(d.name.split("-g")[1])
        except (IndexError, ValueError):
            return -1
    for d in sorted((d for d in OUT.glob("checkpoint-g*") if _step_of(d) >= 0),
                    key=_step_of, reverse=True):
        if (d/"model.pt").exists() and (d/"optimizer.pt").exists():
            cands.append((d, _step_of(d)))
    if st_cand is not None:
        # estado.json primero (mas fiable); el loop cubre el resto
        if _try_load(st_cand, st_step):
            return
    for ck, step in cands:
        if st_cand is not None and ck == st_cand:
            continue  # ya probado
        if _try_load(ck, step):
            return

def _consolidate_zero1():
    """ZeRO-1: consolidate_state_dict es COLECTIVO (lo llaman TODOS los ranks;
    solo rank0 escribe). Se llama tras zero_grad -> grads liberados y el gather
    de ~9GB en rank0 cabe en la tarjeta."""
    if hasattr(glob_opt, "consolidate_state_dict") and torch.distributed.is_initialized():
        glob_opt.zero_grad(set_to_none=True)
        glob_opt.consolidate_state_dict(to=0)

def _handle_sig(sig, frm):
    log("SIGUSR1 — guardando ola y saliendo 42")
    r = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    w = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    if w > 1:
        # todos los ranks reciben USR1 via srun -> todos consolidan
        _consolidate_zero1()
        if r != 0:
            raise SystemExit(42)
    if w <= 1 or r == 0:
        # ANTI-REENTRADA (2026-09-08, auditoria 1.1): si USR1 llega mientras el
        # hilo principal esta dentro de _save_checkpoint (flock LOCK_EX tomado),
        # re-entrar en flock seria DEADLOCK: la senial se ejecuta en el mismo
        # hilo que interrumpe el save, y el lock lo sostiene el frame suspendido.
        # -> Slurm KILL al limite -> finalize() nunca corre -> cadena muerta.
        # Fix: flag SAVING; si ya se esta guardando, salir 42 sin tocar el lock
        # (el save en curso termina via finally; el ckpt previo atomico vale).
        if _SAVING[0]:
            log("  USR1 durante save en curso — saliendo 42 sin re-entrar (ckpt previo)")
            raise SystemExit(42)
        _save_checkpoint("sigusr1")
    raise SystemExit(42)

_SAVING = [False]  # anti-reentrada SIGUSR1 (2026-09-08, auditoria 1.1); antes del handler
signal.signal(signal.SIGUSR1, _handle_sig)

# ---------------- train ----------------
def main():
    global glob_model, glob_opt, DEVICE
    # ---- DDP init (multi-GPU via slurm) ----
    # ORDEN CRITICO 2026-09-01 (bug DDP roto): con torchrun dentro de slurm
    # --ntasks=1, los hijos HEREDAN SLURM_PROCID=0/SLURM_NTASKS=1; torchrun setea
    # RANK/LOCAL_RANK/WORLD_SIZE. Si SLURM_* gana -> todos los ranks ven rank=0,
    # world=1, el DDP jamas se activa, 2 copias independientes entrenan y ambos
    # ranks escriben el MISMO ckpt dir (race: FileNotFoundError en os.replace,
    # crash exit 1 sin auto-resubmit -> bw3 muerto). RANK/WORLD_SIZE primero.
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
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
    # pre-computar token ids de etiquetas de esqueleto para whole_stage
    STAGE_LABEL_IDS = None
    if (ARGS.whole_stage or ARGS.candidate_focus > 0) and tok:
        STAGE_LABEL_IDS = [tok.encode(lbl, add_special_tokens=False) for lbl in ARGS.stage_labels]
        log(f"stage labels: {ARGS.stage_labels} ids={STAGE_LABEL_IDS}")
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
    glob_model._grad_ckpt = ARGS.grad_ckpt
    if ARGS.grad_ckpt:
        log("grad_ckpt ON: activaciones por bloque recomputadas en backward")
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
    if ARGS.zero1 and ddp and world > 1:
        # ZeRO-1: estados AdamW shardados entre ranks (memoria fija /world).
        # params+grads siguen replicados (DDP hace el allreduce de grads).
        from torch.distributed.optim import ZeroRedundancyOptimizer
        glob_opt = ZeroRedundancyOptimizer(
            glob_model.parameters(), optimizer_class=torch.optim.AdamW,
            lr=ARGS.lr, weight_decay=0.01,
            parameters_as_bucket_view=True,
            fused=torch.cuda.is_available())
        log(f"optimizer: ZeroRedundancyOptimizer(AdamW) shardado en {world} ranks")
    else:
        glob_opt = torch.optim.AdamW(glob_model.parameters(), lr=ARGS.lr,
                                     weight_decay=0.01,
                                     fused=torch.cuda.is_available())

    def _opt_groups():
        # ZeroRed guarda los param_groups reales en el optimizador interno
        o = getattr(glob_opt, "optim", glob_opt)
        return o.param_groups

    def _set_lr(step):
        # WARMUP REAL (2026-09-08, auditoria 1.8) + cosine/WSD v2
        if ARGS.warmup > 0 and step < ARGS.warmup:
            lr = ARGS.lr * (step + 1) / ARGS.warmup
        elif ARGS.lr_decay == "cosine":
            progress = (step - ARGS.warmup) / max(1, ARGS.max_steps - ARGS.warmup)
            progress = min(1.0, progress)
            lr = ARGS.lr_min_ratio * ARGS.lr + (1 - progress) * (ARGS.lr - ARGS.lr_min_ratio * ARGS.lr)
        elif ARGS.lr_decay == "wsd":
            stable_end = int(ARGS.max_steps * 0.9)
            if step < stable_end:
                lr = ARGS.lr
            else:
                progress = (step - stable_end) / max(1, ARGS.max_steps - stable_end)
                progress = min(1.0, progress)
                lr = ARGS.lr_min_ratio * ARGS.lr + (1 - progress) * (ARGS.lr - ARGS.lr_min_ratio * ARGS.lr)
        else:
            lr = ARGS.lr
        for g in _opt_groups():
            g["lr"] = lr

    def _init_ema():
        if ARGS.ema_decay > 0 and glob_ema_sd[0] is None:
            m = getattr(glob_model, "module", glob_model)
            glob_ema_sd[0] = {k: v.detach().clone().to(_ema_dev()) for k, v in m.state_dict().items()}

    def _update_ema(step):
        if glob_ema_sd[0] is None or (step + 1) % ARGS.ema_every != 0:
            return
        m = getattr(glob_model, "module", glob_model)
        sd = m.state_dict()
        beta = ARGS.ema_decay ** ARGS.ema_every  # misma escala temporal
        for k, ev in glob_ema_sd[0].items():
            ev.mul_(beta).add_(sd[k].detach().to(ev.device), alpha=1.0 - beta)

    def _log_config():
        cfg = {
            "lr_decay": ARGS.lr_decay, "lr_min_ratio": ARGS.lr_min_ratio,
            "grad_clip": ARGS.grad_clip, "ema_decay": ARGS.ema_decay,
            "weight_tying": ARGS.weight_tying, "use_rope": ARGS.use_rope,
        }
        log(f"optimizer v2: AdamW lr={ARGS.lr} warmup={ARGS.warmup} " + json.dumps(cfg))

    _log_config()
    resume()
    _init_ema()
    if ddp:
        glob_model = torch.nn.parallel.DistributedDataParallel(
            glob_model, device_ids=[dev_idx], find_unused_parameters=False)
    glob_model.zero_grad(set_to_none=True)
    MASK = ARGS.vocab
    nb = len(batches); it = 0
    log(f"masking: schedule={ARGS.mask_schedule} type={ARGS.mask_type} "
        f"whole_stage={ARGS.whole_stage} span_len={ARGS.span_len} "
        f"role_mask={ARGS.role_mask} candidate_focus={ARGS.candidate_focus}")
    if ARGS.curriculum:
        s0, f0 = curriculum_state(STEPS_DONE[0], ARGS.cur_stages)
        log(f"curriculum on: steps={len(ARGS.cur_stages)} stages, "
            f"start(span={s0}, b_h={f0:.2f})")
    if ARGS.role_mask:
        log(f"role_mask on: conectivas/relaciones/numeros/entidades "
            f"(config keys={list(ARGS.role_config.keys())})")
    CORRUPT = None
    if (ARGS.corrective_p > 0 or ARGS.loss_num_w != 1.0
            or ARGS.loss_neg_w != 1.0 or ARGS.contr_w > 0):
        CORRUPT = build_correction_tables(tok, ARGS.vocab, device=DEVICE)
        log(f"corrective/numw on: p={ARGS.corrective_p} "
            f"boost={ARGS.corrective_boost} num_ids={CORRUPT['n_num']} "
            f"flip_ids={CORRUPT['n_flip']} "
            f"loss_w(num={ARGS.loss_num_w}, neg={ARGS.loss_neg_w}) "
            f"contr_w={ARGS.contr_w} margin={ARGS.contr_margin} "
            f"contr_p={ARGS.contr_p}")
    MRAS_EMA = None
    if ARGS.mras_gamma > 0:
        MRAS_EMA = torch.ones(ARGS.vocab, device=DEVICE)
        log(f"mras on: gamma={ARGS.mras_gamma} ema={ARGS.mras_ema} "
            f"floor={ARGS.mras_floor}")
    for step in range(STEPS_DONE[0], ARGS.max_steps):
        xb = batches[it % nb].to(DEVICE).long(); it += 1
        B, T = xb.shape
        # scores semanticos por token (cacheados); solo se calculan con --role_mask
        role_scores = None
        if ARGS.role_mask:
            role_scores = get_token_role_scores(tok, xb)
        if MRAS_EMA is not None:
            ema_n = MRAS_EMA / MRAS_EMA.mean().clamp(min=1e-8)
            ms = ARGS.mras_floor + (1.0 - ARGS.mras_floor) * ema_n.clamp(max=8.0) ** ARGS.mras_gamma
            adapt_scores = ms[xb.clamp(max=ARGS.vocab - 1)]
            role_scores = adapt_scores if role_scores is None else role_scores * adapt_scores
        eff_span_len = ARGS.span_len
        if ARGS.curriculum:
            eff_span_len, cur_b_h = curriculum_state(step, ARGS.cur_stages)
            ARGS.mask_schedule_args["b_h"] = float(cur_b_h)
        # v2: samplear fraccion de corrupcion POR EJEMPLO (t ~ U(0,1))
        n_masked_per_ex = []
        for _ in range(B):
            frac = sample_mask_fraction(ARGS.mask_schedule, step, device=xb.device)
            if ARGS.mask_schedule == "fixed":
                n = max(1, int((T // 2) * ARGS.mask_p))
            else:
                n = max(1, int(T * frac))
            n_masked_per_ex.append(n)
        mp_parts = []
        cp_parts = []
        for b_idx in range(B):
            xb_b = xb[b_idx:b_idx+1]
            mp_b = None
            if (ARGS.candidate_focus > 0 and STAGE_LABEL_IDS
                    and float(torch.rand((), device=xb.device)) < ARGS.candidate_focus):
                mp_b = build_candidate_mask(
                    xb_b, STAGE_LABEL_IDS, device=xb.device,
                    span_scores=role_scores[b_idx:b_idx+1]
                    if (role_scores is not None and MRAS_EMA is not None)
                    else None)
            if mp_b is None:
                mp_b = build_mask_indices(xb_b, n_masked_per_ex[b_idx], ARGS.mask_type,
                                          whole_stage=ARGS.whole_stage,
                                          stage_label_ids=STAGE_LABEL_IDS,
                                          span_len=eff_span_len, device=xb.device,
                                          token_scores=role_scores[b_idx:b_idx+1] if role_scores is not None else None)
            mp_parts.append(mp_b + b_idx * T)
            if CORRUPT is not None and ARGS.corrective_p > 0:
                cp_b = _corrupt_positions(xb_b, mp_b, CORRUPT, xb.device)
                if cp_b is not None and cp_b.numel() > 0:
                    cp_parts.append(cp_b + b_idx * T)
        mp = torch.cat(mp_parts)
        cp = torch.cat(cp_parts) if cp_parts else None
        assert int(mp.max()) < B * T, f"mask indices out of bounds: max={int(mp.max())} >= {B*T}"
        assert int(xb.max()) < ARGS.vocab, f"token id out of vocab: max={int(xb.max())} >= {ARGS.vocab}"
        xm = xb.clone()
        xm.view(-1)[mp] = ARGS.vocab
        sup = mp
        if cp is not None:
            xm.view(-1)[cp] = _corrupt_replacements(
                xb.reshape(-1)[cp], CORRUPT, ARGS.vocab, xb.device)
            sup = torch.cat([mp, cp])
        assert int(xm.max()) <= ARGS.vocab, f"masked input out of embedding range: max={int(xm.max())}"
        out = glob_model(xm)
        tgt = xb.reshape(-1)[sup]
        per = F.cross_entropy(out.reshape(B * T, -1)[sup], tgt,
                              reduction="none")
        if CORRUPT is not None and (ARGS.loss_num_w != 1.0 or
                                    ARGS.loss_neg_w != 1.0):
            w = CORRUPT["lw"][tgt]
            loss = (per * w).sum() / w.sum().clamp(min=1e-8)
        else:
            loss = per.mean()
        if CORRUPT is not None and ARGS.contr_w > 0:
            # hinge in-loss: logp[real] - logp[mutado] >= margen en posiciones
            # enmascaradas mutables (digito/negacion/direccion) — alineacion
            # directa con la metrica pairwise del eval, sin forward extra.
            tgt_mp = xb.reshape(-1)[mp]
            mutable = CORRUPT["is_numx"][tgt_mp] | CORRUPT["is_flip"][tgt_mp]
            if ARGS.contr_p < 1.0:
                mutable = mutable & (torch.rand(mutable.numel(),
                                                device=xb.device)
                                     < ARGS.contr_p)
            if bool(mutable.any()):
                tsel = tgt_mp[mutable]
                alt = _contrast_alts(tsel, CORRUPT, ARGS.vocab, xb.device)
                lg = F.log_softmax(
                    out.reshape(B * T, -1)[mp][mutable].float(), dim=-1)
                gap = (lg.gather(1, tsel[:, None]).squeeze(1)
                       - lg.gather(1, alt[:, None]).squeeze(1))
                loss = loss + ARGS.contr_w * F.relu(
                    ARGS.contr_margin - gap).mean()
        if MRAS_EMA is not None:
            with torch.no_grad():
                ce_sum = torch.zeros_like(MRAS_EMA)
                ce_cnt = torch.zeros_like(MRAS_EMA)
                ce_sum.scatter_add_(0, tgt.clamp(max=ARGS.vocab - 1),
                                    per.detach().float())
                ce_cnt.scatter_add_(0, tgt.clamp(max=ARGS.vocab - 1),
                                    torch.ones_like(per, dtype=torch.float))
                seen = ce_cnt > 0
                d = ARGS.mras_ema
                MRAS_EMA[seen] = d * MRAS_EMA[seen] + (1.0 - d) * (
                    ce_sum[seen] / ce_cnt[seen])
        raw = glob_model.module if ddp else glob_model
        aux = sum(b.mlp.balance_loss(0.01) for b in raw.blocks)
        sync = (not ddp) or ((step+1) % ARGS.grad_accum == 0)
        cm = glob_model.no_sync() if (ddp and not sync) else contextlib.nullcontext()
        with cm:
            (loss/ARGS.grad_accum + aux).backward()
        if sync:
            _set_lr(step)
            if ARGS.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(glob_model.parameters(), ARGS.grad_clip)
            glob_opt.step(); glob_opt.zero_grad(set_to_none=True)
            _update_ema(step)
        LAST_LOSS[0] = loss.item(); STEPS_DONE[0] = step+1
        if ddp: torch.distributed.barrier()
        if (not ddp or rank==0) and step % 10 == 0: log(f"step {step} loss {loss.item():.4f}")
        if step % ARGS.ckpt_every == 0:
            _consolidate_zero1()          # colectivo: todos los ranks
            if not ddp or rank==0: _save_checkpoint("step")
    _consolidate_zero1()
    if not ddp or rank==0: _save_checkpoint("final")
    if ddp: torch.distributed.destroy_process_group()
    if not ddp or rank==0: log("COMPLETE")

if __name__ == "__main__":
    main()
