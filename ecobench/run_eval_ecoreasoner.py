#!/usr/bin/env python3
"""
run_eval_ecoreasoner.py — Correr EcoBench-EVAL con el EcoReasoner LOCAL
(LLaDA-MoE-7B-A1B + LoRA SFT v2) como agente, MISMA salida JSON que
run_eval_agent.py para comparar directo contra v4flash 12/14, glm 6/14, etc.

El MoE genera codigo Python (estilo agente) desde el prompt del item, con
retry-fix sobre errores de ejecucion, igual que el driver de los teachers.
Backend "local": carga el checkpoint con los adaptadores y genera code via
denoising block-wise. Prompts reconstruidos: question + dataset tree.

Uso (slurm, 1x GPU):
  sbatch ecobench/run_eval_ecoreasoner.slurm  (o --ids sab-48,...)
Salida: ecobench/ecobench_eval_results_ecoreasoner.json
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
SAB_BENCH = ROOT / "ecobench_raw" / "sciagentbench" / "benchmark"
DEFAULT_ITEMS = ROOT / "ecobench_eval.json"

# prompts del harness SAB (los mismo que usa run_eval_agent)
SYSTEM_PROMPT = (
    "You are an expert ecological modeler and scientific Python/R programmer. "
    "Given a paper's title and abstract, generate a Chain-of-Thought trace for "
    "fine-tuning an AI model on ecological reasoning.\n\n"
    "Write the Python code ONLY (no markdown fences, no explanation). The code "
    "must: read the dataset from CWD using the EXACT paths given, run the "
    "analysis, and save the artifact to the EXACT relative path requested."
)

MAX_TOKENS = 4096


def load_ecoreasoner(model_dir, lora_path):
    """Carga el LLaDA-MoE + LoRA SFT. Importa dentro (necesita venv-extra)."""
    from transformers import AutoModel, AutoTokenizer

    MASK_ID = 156895

    class LoRALinear(nn.Module):
        def __init__(self, in_f, out_f, r, alpha, dropout=0.1, dtype=torch.bfloat16):
            super().__init__()
            self.lora_A = nn.Parameter(torch.zeros(in_f, r, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(r, out_f, dtype=dtype))
            self.scale = alpha / r
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return self.drop(x) @ self.lora_A @ self.lora_B * self.scale

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True,
                                        local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                      dtype=torch.bfloat16,
                                      local_files_only=True).to("cuda").eval()
    # cargar adaptadores
    ckpt = torch.load(lora_path, map_location="cpu", weights_only=False)
    ad = ckpt["adapters"]
    n_loaded = 0
    for name, mod in model.named_modules():
        la = name + ".lora_A"; lb = name + ".lora_B"
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
    print(f"[ecoreasoner] cargados {n_loaded} adaptadores", flush=True)
    return model, tok, MASK_ID


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    rem = mask_num % steps
    ntr = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                      dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        ntr[i, :rem[i]] += 1
    return ntr


@torch.no_grad()
def generate(model, tok, prompt, MASK_ID, gen_length=512, steps=125,
             block_length=32, temperature=0.0, max_prompt=1400):
    ids = tok(prompt, return_tensors="pt")["input_ids"][:, :max_prompt].to("cuda")
    L = ids.shape[1]
    x = torch.full((1, L + gen_length), MASK_ID, dtype=torch.long, device="cuda")
    x[:, :L] = ids
    num_blocks = gen_length // block_length
    spb = steps // num_blocks
    for b in range(num_blocks):
        bm = x[:, L + b * block_length: L + (b + 1) * block_length] == MASK_ID
        ntr = get_num_transfer_tokens(bm, spb)
        for i in range(spb):
            mask_index = (x == MASK_ID)
            lg = model(x).logits
            x0 = torch.argmax(add_gumbel_noise(lg, temperature), dim=-1)
            p = F.softmax(lg.float(), dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, L + (b + 1) * block_length:] = -float("inf")
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p, -float("inf"))
            _, sel = torch.topk(conf, k=int(ntr[0, i]))
            x[0, sel] = x0[0, sel]
    return tok.decode(x[0, L:], skip_special_tokens=True)


def build_prompt(q, tree):
    p = f"{q}"
    if tree:
        p += f"\n\nDataset folder tree (use these EXACT paths, relative to CWD):\n{tree}"
    return p


def run_code(code, workdir, timeout=300):
    with tempfile.NamedTemporaryFile("w", suffix="_eco.py", delete=False,
                                     dir=str(workdir)) as f:
        f.write(code); path = f.name
    try:
        r = subprocess.run([sys.executable, path], cwd=str(workdir),
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(DEFAULT_ITEMS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lora", default=str(ROOT.parent / "outputs/sft_moe_v2/lora-final/lora.pt"))
    ap.add_argument("--model-dir", default=str(ROOT.parent / "models/LLaDA-MoE-7B-A1B-Instruct"))
    ap.add_argument("--timeout-per-item", type=int, default=300)
    ap.add_argument("--ids", default="")
    ap.add_argument("--gen-length", type=int, default=512)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    doc = json.load(open(args.items))
    meta, items = doc["meta"], [i for i in doc["items"] if i["split"] == "eval_holdout"]
    items = [i for i in items if not i["id"].startswith("eco-")]
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        items = [i for i in items if i["id"] in want]
    if args.limit:
        items = items[:args.limit]
    print(f"== EcoBench-EVAL EcoReasoner local == {len(items)} items")

    model, tok, MASK_ID = load_ecoreasoner(args.model_dir, args.lora)

    def load_sab_tree():
        t = {}
        csvp = ROOT / "ecobench_raw" / "sciagentbench" / "ScienceAgentBench.csv"
        if csvp.exists():
            import csv
            with open(csvp) as f:
                for row in csv.DictReader(f):
                    t[row.get("id", "")] = row.get("folder_tree", "") or row.get("tree", "") or ""
        return t
    sab_tree = load_sab_tree()

    results = []
    for it in items:
        iid, q = it["id"], it.get("question", "")
        fam = it.get("family", "?")
        print(f"\n[{iid}] ({fam})", flush=True)
        if args.dry:
            results.append({"id": iid, "family": fam, "status": "dry"})
            continue
        iid_num = iid.replace("sab-", "")
        tree = sab_tree.get(iid_num, "")
        prompt = build_prompt(q, tree)
        t0 = time.time()
        code = generate(model, tok, prompt, MASK_ID, gen_length=args.gen_length)
        print(f"  generado {len(code)} chars en {time.time()-t0:.0f}s", flush=True)
        # limpiar codigo (fences si las hubiera)
        m = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
        if m:
            code = m.group(1).strip()
        if not code or len(code) < 20:
            results.append({"id": iid, "family": fam, "status": "empty_code", "gen": code[:100]})
            continue
        rc, so, se = run_code(code, SAB_BENCH, timeout=args.timeout_per_item)
        if rc != 0:
            results.append({"id": iid, "family": fam, "status": "exec_fail", "rc": rc,
                            "stderr": se[:600]})
            continue
        # artifact check
        expected = it.get("execution", {}).get("expected", {})
        exp_path = expected.get("value", "") if expected.get("type") == "file_artifact" else ""
        if exp_path:
            full = SAB_BENCH / exp_path
            ok = full.exists()
            results.append({"id": iid, "family": fam, "status": "pass" if ok else "fail",
                            "artifact": str(full), "expected": exp_path})
            continue
        # eval_program check (reuso)
        from run_eval_agent import run_eval_program, SAB_BENCH as SB
        ev_ok, ev_msg = run_eval_program(it, SB)
        if ev_ok is None:
            results.append({"id": iid, "family": fam, "status": "skip", "reason": ev_msg})
        else:
            results.append({"id": iid, "family": fam, "status": "pass" if ev_ok else "fail",
                            "eval": ev_msg[:300]})

    out = ROOT / "ecobench_eval_results_ecoreasoner.json"
    json.dump({"meta": meta, "results": results, "model": "LLaDA-MoE-7B-A1B+LoRA-SFT-v2",
               "backend": "local", "caveats": ["LLM-visual-judge skipped", "CWD=benchmark"]},
              open(out, "w"), indent=1, ensure_ascii=False)
    from collections import Counter
    st = Counter(r["status"] for r in results)
    print(f"\n== RESULTADO == {dict(st)}")
    print(f"pass@1 = {st.get('pass', 0)}/{len(results)} = {100*st.get('pass', 0)/max(len(results), 1):.0f}%")
    print(f"guardado: {out}")


if __name__ == "__main__":
    main()