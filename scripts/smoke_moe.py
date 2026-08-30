#!/usr/bin/env python3
"""
smoke_moe.py — SMOKE BASELINE del base LLaDA-MoE-7B-A1B-Instruct (2026-08-30).

Mide si el Instruct YA genera tool calls JSON validas sobre los prompts eco
de L1 (gbif_occurrence / bioclim_download / maxent_train / ecocode), SIN
ningun fine-tune. Dos modos:
  - RAW : continua el prompt truncado en "[ACCION] " (igual que smoke_l1).
  - CHAT: prompt como user message via chat_template del Instruct.

Generacion = LLaDA oficial (reverse process, low-confidence remasking,
mask_id 156895) portado del README. Validacion reusa extract_json +
valid_toolcall de smoke_l1 (KNOWN_FNS).

Uso: solo via slurm (GPU + modelo en beegfs).
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

BASE = "/beegfs/a474r867/ecoreasoner"
SNAP = BASE + "/models/LLaDA-MoE-7B-A1B-Instruct"
MASK_ID = 156895
KNOWN_FNS = {"gbif_occurrence", "bioclim_download", "maxent_train", "ecocode"}


# ---- sampling LLaDA oficial (README inclusionAI) ----
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
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps,
                                      device=mask_index.device,
                                      dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt_ids, gen_length=160, steps=125, temperature=0.0,
             block_length=32, device="cuda"):
    """Generacion LLaDA OFICIAL block-wise (README inclusionAI), verbatim:
    gen se divide en bloques; cada bloque se desenmascara en steps//num_blocks
    rondas transfiriendo num_transfer_tokens[i] posiciones por round. Evita el
    enclavamiento en EOS del bloque-unico con transfer lento (2026-08-30)."""
    if hasattr(prompt_ids, "input_ids"):      # BatchEncoding -> tensor
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


# ---- validacion (reuso de smoke_l1) ----
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
    ap.add_argument("--corpus", default=os.path.join(BASE, "data/l1/train_corpus_l1.jsonl"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--gen-length", type=int, default=160)
    ap.add_argument("--steps", type=int, default=125)
    ap.add_argument("--out", default=os.path.join(BASE, "data/l1/smoke_moe_baseline.json"))
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(SNAP, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16,
                                      local_files_only=True).to(device).eval()
    if hasattr(model, "config"):
        model.config.pad_token_id = tok.pad_token_id
    print(f"[smoke-moe] modelo cargado ({time.time()-t0:.0f}s) device={device}", flush=True)

    # prompts L1 kind=gen + repair-M1, truncados en "[ACCION] "
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
    if not prompts:
        sys.exit("ERROR: no prompts L1 (gen/repair-M1) encontrados")
    print(f"[smoke-moe] prompts: {len(prompts)}", flush=True)

    def run_batch(mode, temps):
        res = {}
        for temp in temps:
            n_json = n_fn = n_repair = 0
            samples = []
            for i, (kind, ptext) in enumerate(prompts):
                if mode == "raw":
                    ids = tok(ptext, return_tensors="pt")["input_ids"].to(device)[:, :600]
                else:
                    msgs = [{"role": "user", "content": ptext}]
                    s = tok.apply_chat_template(msgs, tokenize=True,
                                                add_generation_prompt=True,
                                                return_tensors="pt")
                    if hasattr(s, "input_ids"):
                        s = s["input_ids"]
                    ids = s.to(device)
                gen_ids = generate(model, ids, gen_length=a.gen_length,
                                   steps=a.steps, temperature=temp, device=device)
                out = tok.decode(gen_ids, skip_special_tokens=True)
                obj = extract_json(out)
                ok_json = obj is not None
                ok_fn = ok_json and valid_toolcall(obj)
                n_json += ok_json
                n_fn += ok_fn
                if ok_fn and kind != "gen":
                    n_repair += 1
                samples.append({"kind": kind, "ok_json": ok_json, "ok_fn": ok_fn,
                                "fn": (obj.get("function", {}).get("name") if ok_json else None),
                                "gen": out[:200]})
                if (i + 1) % 10 == 0:
                    print(f"[{mode} t{temp}] {i+1}/{len(prompts)} fn={n_fn} "
                          f"[{time.time()-t0:.0f}s]", flush=True)
            res[f"{mode}_t{temp}"] = {"n": len(prompts), "n_json": n_json,
                                      "n_fn_valid": n_fn,
                                      "json_rate": round(n_json / len(prompts), 3),
                                      "fn_rate": round(n_fn / len(prompts), 3),
                                      "repair_fn": n_repair, "samples": samples[:6]}
        return res

    report = {"ckpt": SNAP, "mask_id": MASK_ID, "gen_length": a.gen_length,
              "steps": a.steps, "elapsed_s": round(time.time() - t0, 1),
              "modes": run_batch("raw", [0.0, 0.9]),
              "chat": run_batch("chat", [0.0])}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\n[smoke-moe] RESULTADO:")
    for k, v in report["modes"].items():
        print(f"  {k}: {v['n_fn_valid']}/{v['n']} tool calls validas "
              f"({100*v['fn_rate']:.0f}%) json={v['n_json']}")
    for k, v in report["chat"].items():
        print(f"  {k}: {v['n_fn_valid']}/{v['n']} tool calls validas "
              f"({100*v['fn_rate']:.0f}%) json={v['n_json']}")
    print(f"[smoke-moe] DONE -> {a.out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()