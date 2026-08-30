#!/usr/bin/env python3
"""
smoke_l1.py — VALIDA la capa L1: dado un prompt [INSTRUCCION]+[CONTEXTO]+[ACCION]
(con la tool call OCLUSA), el dLLM (bw2/checkpoint-g7500) debe GENERAR una
tool call JSON valida (denoising iterativo tipo LLaDA).

Criterio de exito L1: >=1 tool call JSON valida (function.name conocido +
arguments parseable) + tasa de JSON valido sobre N prompts (gen + repair).

NOTA: las clases de modelo estan EMBEBIDAS aqui (copia exacta de
train_mdlm_moe.py) porque importar el trainer ejecuta su argparse (--data
required) y muere. Mantener en sync si cambia la arquitectura.

Uso: solo via slurm (necesita GPU + checkpoint en beegfs).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

BASE = "/beegfs/a474r867/ecoreasoner"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TOK_PATH = ("/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/"
            "snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
MASK_ID = 126080
VOCAB = 126080
KNOWN_FNS = {"gbif_occurrence", "bioclim_download", "maxent_train", "ecocode"}


# ---- arquitectura (copia exacta de train_mdlm_moe.py) ----
def _default_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)


class MoEMLP(nn.Module):
    def __init__(self, dim, ff, n_experts, k):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))
            for _ in range(n_experts)])

    def forward(self, x):
        B, T, D = x.shape
        g = self.gate(x)                                   # (B,T,E)
        topk = torch.topk(g, self.k, dim=-1)
        idx = topk.indices                                # (B,T,k)
        w = torch.softmax(topk.values / math.sqrt(D), dim=-1)
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            mask = (idx == e).any(dim=-1)                 # (B,T)
            if mask.any():
                w_e = w[mask].unsqueeze(-1)               # (n,k,1)
                h = x[mask]                               # (n,D)
                h = self.experts[e](h)                    # (n,D)
                # agregar pesos de todos los expertos elegidos en la posicion
                sel = idx[mask]                           # (n,k)
                contrib = torch.zeros_like(h)
                for j in range(self.k):
                    mj = (sel[:, j] == e)
                    if mj.any():
                        contrib[mj] += w_e[mj, j] * h[mj]
                out[mask] += contrib
        return out

    def balance_loss(self, alpha=0.01, probe_alpha=0.01):
        return torch.tensor(0.0, device=next(self.parameters()).device)


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
            Block(hidden, hidden * ff_mult, heads, n_experts, k)
            for _ in range(layers)])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab)
        self.apply(_default_init)

    def forward(self, ids):
        B, T = ids.shape
        h = self.tok_emb(ids) + self.pos(torch.arange(T, device=ids.device))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h))


# ---- sampling masked-diffusion (LLaDA-style) ----
def load_model(ckpt, device):
    m = MdLMMoE(VOCAB, 1024, 16, 16, 4, 768, 4, 1).to(device)
    raw = torch.load(os.path.join(ckpt, "model.pt"), map_location=device,
                     weights_only=False)
    # el trainer guarda {"model": state_dict} (_save_checkpoint)
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    m.load_state_dict(sd)
    m.eval()
    return m


def gen(m, prompt_ids, n_gen=120, steps=64, temp=0.9, device="cuda"):
    """Denoising iterativo LLaDA: enmascara n_gen tokens tras el prompt,
    predice y remaska por confianza, steps rondas."""
    L = len(prompt_ids)
    T = L + n_gen
    x = torch.full((1, T), MASK_ID, dtype=torch.long, device=device)
    x[0, :L] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        for s in range(1, steps + 1):
            logits = m(x)[0]                       # (T, vocab)
            masked = x[0] == MASK_ID
            if int(masked.sum()) == 0:
                break
            lg = logits[masked] / temp
            probs = torch.softmax(lg, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)
            conf = probs.gather(1, nxt.unsqueeze(-1)).squeeze(-1)
            x_new = x.clone()
            x_new[0, masked] = nxt
            if s < steps:
                pos = torch.nonzero(masked).squeeze(-1)
                n_keep = max(1, int(pos.numel() * (steps - s) / steps))
                order = torch.argsort(conf, descending=True)
                keep = order[:n_keep]
                remask = order[n_keep:]
                x = x_new
                x[0, pos[remask]] = MASK_ID
            else:
                x = x_new
    return x[0, L:].tolist()


def extract_json(s):
    """Primer objeto JSON balanceado en la cadena."""
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[i:j + 1])
                    except Exception:
                        break
    return None


def valid_toolcall(obj):
    if not isinstance(obj, dict):
        return False
    fn = obj.get("function")
    if not isinstance(fn, dict):
        return False
    name = fn.get("name")
    args = fn.get("arguments")
    if name not in KNOWN_FNS:
        return False
    if isinstance(args, dict):
        return True
    if isinstance(args, str):
        try:
            json.loads(args)
            return True
        except Exception:
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(BASE, "outputs/bw2/checkpoint-g7500"))
    ap.add_argument("--corpus", default=os.path.join(BASE, "data/l1/train_corpus_l1.jsonl"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-gen", type=int, default=120)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(BASE, "data/l1/smoke_l1_report.json"))
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    m = load_model(a.ckpt, device)
    print(f"[smoke] modelo cargado {a.ckpt} ({time.time()-t0:.0f}s) device={device}",
          flush=True)

    # prompts del corpus L1: kind=gen (tool call fresh) + kind=repair-M1 (error->repair)
    # texto termina en '\n[ACCION] {json}' -> truncamos justo antes del json
    prompts = []
    with open(a.corpus, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            kind = d.get("kind", "")
            if kind not in ("gen", "repair-M1"):
                continue
            txt = d.get("text", "")
            mark = "[ACCION]"
            idx = txt.rfind(mark)
            if idx < 0:
                continue
            prompts.append((kind, txt[:idx + len(mark)] + " "))
            if len(prompts) >= a.n:
                break
    if not prompts:
        sys.exit("ERROR: no prompts L1 (gen/repair-M1) encontrados")

    print(f"[smoke] prompts: {len(prompts)} (gen={sum(1 for k,_ in prompts if k=='gen')}, "
          f"repair={sum(1 for k,_ in prompts if k!='gen')})", flush=True)

    n_json = n_fn = n_repair = 0
    samples = []
    t1 = time.time()
    # diagnostico: barrer temp x steps para descartar sampler mal configurado
    cfgs = [(0.4, 128), (0.6, 128), (0.9, 64)]
    for temp, steps in cfgs:
        ok = 0
        for i, (kind, ptext) in enumerate(prompts[:10]):
            ids = tok.encode(ptext)[:600]
            if not ids:
                continue
            gen_ids = gen(m, ids, n_gen=a.n_gen, steps=steps, temp=temp, device=device)
            out = tok.decode(gen_ids, skip_special_tokens=True)
            obj = extract_json(out)
            if obj is not None and valid_toolcall(obj):
                ok += 1
            if i == 0:
                samples.append({"cfg": f"t{temp}_s{steps}", "kind": kind,
                                "gen": out[:220]})
        print(f"[smoke] cfg temp={temp} steps={steps}: valid={ok}/10", flush=True)
    # corrida principal con la mejor cfg (0.9/64 por defecto)
    temp, steps = 0.9, a.steps
    for i, (kind, ptext) in enumerate(prompts):
        ids = tok.encode(ptext)[:600]
        if not ids:
            continue
        gen_ids = gen(m, ids, n_gen=a.n_gen, steps=a.steps, device=device)
        out = tok.decode(gen_ids, skip_special_tokens=True)
        obj = extract_json(out)
        ok_json = obj is not None
        ok_fn = ok_json and valid_toolcall(obj)
        if ok_json:
            n_json += 1
        if ok_fn:
            n_fn += 1
            if kind != "gen":
                n_repair += 1
        samples.append({"kind": kind, "ok_json": ok_json, "ok_fn": ok_fn,
                        "fn": (obj.get("function", {}).get("name") if ok_json else None),
                        "gen": out[:200]})
        if (i + 1) % 10 == 0:
            print(f"[smoke] {i+1}/{len(prompts)} json={n_json} fn_valid={n_fn} "
                  f"[{time.time()-t1:.0f}s]", flush=True)

    report = {"n": len(prompts), "n_json": n_json, "n_fn_valid": n_fn,
              "json_rate": round(n_json / len(prompts), 3),
              "fn_rate": round(n_fn / len(prompts), 3),
              "repair_fn": n_repair,
              "n_gen": sum(1 for k, _ in prompts if k == "gen"),
              "n_repair": sum(1 for k, _ in prompts if k != "gen"),
              "steps": a.steps, "n_gen_tok": a.n_gen,
              "ckpt": a.ckpt, "elapsed_s": round(time.time() - t0, 1),
              "samples": samples}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[smoke] RESULTADO: {n_fn}/{len(prompts)} tool calls validas "
          f"({100*n_fn/len(prompts):.0f}%) | json={n_json} | repair={n_repair}")
    print(f"[smoke] DONE -> {a.out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()