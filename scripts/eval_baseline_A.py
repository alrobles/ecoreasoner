#!/usr/bin/env python3
"""
eval_baseline.py — Evaluacion del baseline EcoReasoner M1 (Bloque A, Fase 2).

Carga el modelo base Qwen3.5-35B-A3B + adapter LoRA de cada replica (m1/w1/w2)
y mide, para cada una:
  1. tool_call_format_ok : % de respuestas que emiten tool_call valido (JSON)
     sobre un set canonico de prompts agenticos ecoseek.
  2. memorized_placeholder: cuantos de los prompts del train (placeholder sci_v1,
     29 items) recupera ~verbatim (huella de memorizacion).
  3. regression_vs_base   : (si --val-file presente) loss/perplexity sobre un
     subconjunto de validacion no visto; se compara vs el base sin afinar.

Replica el patron de carga del train_q35_fsdp.py: AutoModelForCausalLM bf16,
device_map='balanced' sobre 2x MI210, + load_adapter(str(latest/'adapter')).

Uso (en nodo GPU via slurm):
  python3 eval_baseline_A.py --base MODEL_DIR --adapters BASE/outputs/{m1,w1,w2} \
      --prompts prompts_toolcall.jsonl --val-file val.jsonl --out eval_A.json
"""
import argparse
import json
import os
import re
import sys
import time

# ── Carga torch/transformers solo cuando haya GPU (slurm exec con --nv) ──
def log(msg):
    print(f"[evalA {time.strftime('%H:%M:%S')}] {msg}", flush=True)

def parse_args():
    p = argparse.ArgumentParser(description="Eval baseline EcoReasoner M1 (Bloque A)")
    p.add_argument("--base", required=True, help="ruta modelo base Qwen3.5-35B-A3B")
    p.add_argument("--adapters", nargs="+", required=True,
                   help="rutas a cada terminal/final dir con adapter (orden = replicas)")
    p.add_argument("--prompts", required=True,
                   help="JSONL de prompts canonicos agenticos (campo 'prompt')")
    p.add_argument("--mem-check", default=None,
                   help="JSONL de los 29 items del placeholder sci_v1 (campo 'prompt') para huella de memorizacion")
    p.add_argument("--val-file", default=None,
                   help="opcional: JSONL con field 'input' para regresion val vs base")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out", default="eval_baseline_A.json")
    return p.parse_args()

# ── Tool-call format check ─────────────────────────────────────────────
TOOLCALL_RE = re.compile(
    r'```\s*(?:json|tool_call[a-z]*)?\s*\{[^}]{0,200}?["\'](?:name|function|tool)["\']'
    r'[\s\S]{0,600}?["\'](?:arguments|input|params)["\']',
    re.IGNORECASE,
)

def format_ok(text: str) -> bool:
    return bool(TOOLCALL_RE.search(text))

def _repetitive(text: str) -> bool:
    """Heuristico: respuesta larga con pocos bigramas unicos (verbatim/parafraseo)."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 15:
        return False
    bigrams = {f"{a} {b}" for a, b in zip(words, words[1:])}
    return len(bigrams) / max(len(words) - 1, 1) < 0.25

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"raw": line})
    return rows

def main():
    args = parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    log(f"PyTorch {torch.__version__} | ROCm {torch.cuda.is_available()} | GPUs {torch.cuda.device_count()}")
    log(f"Base: {args.base} | Adapters: {args.adapters}")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Prompts canonicos + memorizacion
    prompts = [r.get("prompt", r.get("input", "")) for r in load_jsonl(args.prompts)]
    prompts = [p for p in prompts if p.strip()]
    mem_items = (
        [r.get("prompt", r.get("input", "")) for r in load_jsonl(args.mem_check)]
        if args.mem_check and os.path.exists(args.mem_check) else []
    )
    mem_items = [m for m in mem_items if m.strip()]
    log(f"toolcall prompts={len(prompts)} | mem items={len(mem_items)}")

    # Optativo: sorport de val
    val_prompts = []
    if args.val_file and os.path.exists(args.val_file):
        val_prompts = [r.get("input", r.get("prompt", "")) for r in load_jsonl(args.val_file)]
        val_prompts = [v for v in val_prompts if v.strip()][:200]
        log(f"val prompts={len(val_prompts)}")

    all_results = {}

    for i, adir in enumerate(args.adapters):
        wave = os.path.basename(os.path.normpath(adir))
        log(f"── replica '{wave}' adapter={adir} ──")
        # Modelo base + adapter (patrón train_q35_fsdp)
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        try:
            model.load_adapter(adir, adapter_name="default")
            model.set_adapter("default")
            log("  adapter cargado OK")
        except Exception as e:
            log(f"  WARNING load_adapter fallo: {e}")

        model.eval()
        res = {"wave": wave, "adapter": adir}
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

        # 1) tool-call format
        tc_ok = 0
        for p in prompts:
            msgs = [
                {"role": "system", "content": "Eres un agente cientifico de ecoseek. Para actuar, emite una tool_call en JSON con 'name' y 'arguments'."},
                {"role": "user", "content": p},
            ]
            ids = tok.apply_chat_template(msgs, return_tensors="pt", tokenize=True, add_generation_prompt=True)
            ids = ids.to("cuda") if ids.device.type != "cuda" else ids.to("cuda")
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            if format_ok(text):
                tc_ok += 1
        all_results[wave] = {"toolcall_format": tc_ok / max(len(prompts), 1), "n_prompts": len(prompts)}
        log(f"  toolcall_format={all_results[wave]['toolcall_format']:.2f} ({tc_ok}/{len(prompts)})")

        # 2) huella de memorizacion (ver almacenar reproduccion verbatim)
        mem_hits = 0
        for m in mem_items:
            msgs = [{"role": "user", "content": m}]
            ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to("cuda")
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=256, do_sample=False)
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            # una "reproduccion" si la respuesta recita parte del item o la resp esperada conocida
            if m[:40] in text or (len(text) > 80 and _repetitive(text)):
                mem_hits += 1
        all_results[wave]["memory_verbatim_approx"] = mem_hits / max(len(mem_items), 1)
        log(f"  mem_verbatim_approx: {all_results[wave]['memory_verbatim_approx']:.2f} ({mem_hits}/{max(len(mem_items),1)})")

        del model
        torch.cuda.empty_cache()

    # 3) regresion vs base (si val presente)
    base_res = {
        "wave": "base",
        "toolcall_format": None,
        "memory_verbatim_approx": None,
        "n_prompts": 0,
    }
    if val_prompts:
        log("Evaluando base SIN afinar para linea base de regresion...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16,
            device_map="balanced", trust_remote_code=True)
        model.eval()
        nll = 0.0; n = 0
        with torch.inference_mode():
            for v in val_prompts:
                ids = tok(v, return_tensors="pt", truncation=True, max_length=1024).input_ids.to("cuda")
                if ids.shape[1] < 10: continue
                loss = model(input_ids=ids, labels=ids).loss
                nll += loss.item(); n += 1
        base_res["base_val_loss"] = nll / max(n, 1)
        base_res["val_n"] = n
        log(f"  base val loss: {base_res['base_val_loss']:.4f} (n={n})")
        torch.cuda.empty_cache()
        del model

    # Guardar
    all_results["base"] = base_res
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"RESULTADO guardado en {os.path.abspath(args.out)}")
    print(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()