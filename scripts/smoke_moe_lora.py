#!/usr/bin/env python3
"""
smoke_moe_lora.py — Smoke post-SFT: carga el LLaDA-MoE-7B-A1B-Instruct +
los adaptadores LoRA del SFT (lora-final), y prueba si YA genera tool calls
JSON validas en los prompts eco L1.

Mismos parametros que smoke_moe.py (raw/chat, gen_length, steps, block-wise),
pero con los adaptadores cargados. Valida el hito L1: el dLLM genera tool
calls EL SOLO tras el SFT.
"""
import argparse, json, os, sys, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = "/beegfs/a474r867/ecoreasoner"
MODEL_DIR = os.path.join(BASE, "models/LLaDA-MoE-7B-A1B-Instruct")
LORA_FINAL = os.path.join(BASE, "outputs/sft_moe_v2/lora-final/lora.pt")
MASK_ID = 156895
KNOWN_FNS = {"gbif_occurrence", "bioclim_download", "maxent_train", "ecocode"}


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    ntr = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                      dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        ntr[i, :remainder[i]] += 1
    return ntr


@torch.no_grad()
def generate(model, prompt_ids, gen_length=160, steps=125, temperature=0.0,
             block_length=32, device="cuda"):
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    L = prompt_ids.shape[1]
    x = torch.full((1, L + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :L] = prompt_ids
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, L + num_block * block_length: L + (num_block + 1) * block_length] == MASK_ID
        )
        ntr = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == MASK_ID)
            logits = model(x).logits
            x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            p = F.softmax(logits.float(), dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, L + (num_block + 1) * block_length:] = -float("inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float("inf"))
            _, sel = torch.topk(confidence, k=int(ntr[0, i]))
            x[0, sel] = x0[0, sel]
    return x[0, L:]


# ---- LoRA (mismo esquema que el trainer) ----
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r, alpha, dropout=0.1, dtype=torch.bfloat16):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_f, r, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f, dtype=dtype))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(x) @ self.lora_A @ self.lora_B * self.scale


def load_lora(model, lora_path):
    ckpt = torch.load(lora_path, map_location="cpu", weights_only=False)
    ad = ckpt["adapters"]
    n_loaded = 0
    for name, mod in model.named_modules():
        la = name + ".lora_A"
        lb = name + ".lora_B"
        if la in ad and lb in ad:
            if not hasattr(mod, "lora"):
                mod.lora = LoRALinear(mod.weight.shape[1], mod.weight.shape[0],
                                      64, 128, dtype=mod.weight.dtype).to(mod.weight.device)
                mod.forward_orig = mod.forward

                def _fwd(x, m=mod):
                    return m.forward_orig(x) + m.lora(x)
                mod.forward = _fwd
            mod.lora.lora_A.data = ad[la].to(mod.weight.device, mod.weight.dtype)
            mod.lora.lora_B.data = ad[lb].to(mod.weight.device, mod.weight.dtype)
            n_loaded += 1
    print(f"[smoke-lora] cargados {n_loaded} adaptadores de {lora_path}", flush=True)
    return model


def extract_json(s):
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
    if "function" in obj:
        fn = obj["function"]
    else:
        fn = obj
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
    ap.add_argument("--corpus", default=os.path.join(BASE, "data/l1/train_corpus_l1.jsonl"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--gen-length", type=int, default=160)
    ap.add_argument("--steps", type=int, default=125)
    ap.add_argument("--lora", default=LORA_FINAL)
    ap.add_argument("--out", default=os.path.join(BASE, "data/l1/smoke_moe_lora_v2.json"))
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                      dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    load_lora(model, a.lora)
    print(f"[smoke-lora] modelo + lora cargados ({time.time()-t0:.0f}s)", flush=True)

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
            if d.get("kind") not in ("gen", "repair-M1"):
                continue
            txt = d.get("text", "")
            idx = txt.rfind("[ACCION]")
            if idx < 0:
                continue
            prompts.append((d.get("kind"), txt[:idx + len("[ACCION]")] + " "))
            if len(prompts) >= a.n:
                break
    print(f"[smoke-lora] prompts: {len(prompts)}", flush=True)

    n_json = n_fn = n_repair = 0
    samples = []
    for i, (kind, ptext) in enumerate(prompts):
        msgs = [{"role": "user", "content": ptext}]
        s = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                    return_tensors="pt")
        if hasattr(s, "input_ids"):
            s = s["input_ids"]
        ids = s.to("cuda")
        gen_ids = generate(model, ids, gen_length=a.gen_length, steps=a.steps,
                           temperature=0.0, device="cuda")
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
                        "fn": (obj.get("function", {}).get("name")
                               if ok_json and "function" in obj else
                               (obj.get("name") if ok_json else None)),
                        "gen": out[:220]})
        if (i + 1) % 10 == 0:
            print(f"[smoke-lora] {i+1}/{len(prompts)} fn={n_fn} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    report = {"ckpt": LORA_FINAL, "n": len(prompts), "n_json": n_json,
              "n_fn_valid": n_fn, "json_rate": round(n_json / len(prompts), 3),
              "fn_rate": round(n_fn / len(prompts), 3), "repair_fn": n_repair,
              "elapsed_s": round(time.time() - t0, 1), "samples": samples}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[smoke-lora] RESULTADO L1: {n_fn}/{len(prompts)} tool calls validas "
          f"({100*n_fn/len(prompts):.0f}%) json={n_json} repair={n_repair}")
    print(f"[smoke-lora] DONE -> {a.out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()