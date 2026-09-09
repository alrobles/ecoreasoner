#!/usr/bin/env python3
"""
EcoReasoner M1 — Qwen3.5-35B-A3B LoRA en 2x MI210, encadenado por olas (Tortuga).

Patrón validado en el smoke test de Fase 0.2 (device_map="balanced" repartiendo
las 2 GPUs + gradient_checkpointing) + LoRA rank bajo. UN solo proceso (no
FSDP/torchrun — FSDP+MoE da SIGSEGV en ROCm con ram-efficient, y full da OOM).

Ola = ejecución de hasta TARGET_STEPS pasos en 1 nodo × 2 MI210; al recibir
SIGUSR1 guarda adaptador+estado y sale 42; la siguiente ola reanuda desde el
checkpoint en beegfs (cualquier nodo). Lento pero firme — cada ola aporta.

Uso:
  python3 train_q35_fsdp.py --model $MODEL --data $DATA --output $OUT \
      --max_steps 60 --seq_len 768 --lr 2e-5 --lora_rank 16
"""
import argparse, json, os, signal, sys, time
from pathlib import Path

import torch

# ── Fix ROCm: grouped_mm no soportado + backward no materializa grad congelado ──
# 1) transformers 5.15 usa torch._grouped_mm para MoE en device "cuda" reportado por
#    ROCm; el kernel falla ("grouped gemm is not supported on ROCM"). Forzar el fallback.
# 2) _grouped_mm_fallback_backward hace zeros_like(weight) de TODOS los expertos (gigante)
#    aunque LoRA congele los pesos base. Parchear para que solo compute grad_weight si
#    weight.requires_grad (LoRA: False) → no materializa el grad de expertos → sin OOM.
try:
    import transformers.integrations.moe as _tfmoe
    _tfmoe._can_use_grouped_mm = lambda *a, **k: False  # noqa: E731

    _orig_backward = _tfmoe._grouped_mm_fallback_backward
    def _patched_backward(ctx, grad_output):
        import torch as _t
        input, weight = ctx.saved_tensors
        grad_input = _t.zeros_like(input)
        start = 0
        w_grad_needed = weight.requires_grad
        if w_grad_needed:
            grad_weight = _t.zeros_like(weight)
        for i, end in enumerate(ctx.offs.tolist()):
            if start == end:
                continue
            _t.mm(grad_output[start:end], weight[i].T, out=grad_input[start:end])
            if w_grad_needed:
                _t.mm(input[start:end].T, grad_output[start:end], out=grad_weight[i])
            start = end
        return grad_input, (grad_weight if w_grad_needed else None), None
    _tfmoe._grouped_mm_fallback_backward = _patched_backward
except Exception as _e:  # si el import falla, sigue sin fix
    print(f"WARN monkeypatch grouped_mm skip: {_e}")

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--max_steps", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--grad_accum", type=int, default=4)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--seq_len", type=int, default=768)
parser.add_argument("--save_every", type=int, default=5)
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
args = parser.parse_args()

OUT = Path(args.output)
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "train.log"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f: f.write(line + "\n")
    except Exception: pass

# ── estado global para SIGUSR1 ──────────────────────────────────────────
DONE = {"flag": False}

def _save_checkpoint(tag):
    """Guardar adapters + optimizer + progreso GLOBAL (reanudable entre olas).
    El checkpoint guarda pesos con tag que incluye el step global; state.json
    persiste el paso global acumulado y el ultimo checkpoint (lee el watchdog)."""
    g = DONE.get("global_step", 0)
    ckpt_dir = OUT / f"checkpoint-g{g}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _trainer_state = {
        "step": g,
        "saved_at": time.time(),
        "model_path": str(args.model),
        "lora_rank": args.lora_rank,
        "seq_len": args.seq_len,
    }
    with open(ckpt_dir / "trainer_state.json", "w") as f:
        json.dump(_trainer_state, f, indent=2)
    # adapter LoRA
    DONE["model"].save_pretrained(str(ckpt_dir / "adapter"))
    # optimizer state
    if "opt" in DONE:
        torch.save({"optimizer": DONE["opt"].state_dict()}, ckpt_dir / "optimizer.pt")
    # state.json global (persistente entre olas) — fuente de verdad del progreso
    state = {"step": g, "checkpoint": f"checkpoint-g{g}", "updated": time.time()}
    with open(OUT / "state.json", "w") as f:
        json.dump(state, f, indent=2)
    # progress.json — lo lee el orquestador/watchdog
    with open(OUT / "progress.json", "w") as f:
        json.dump({"step": g, "loss": DONE.get("last_loss"), "updated": time.time()}, f, indent=2)
    log(f"  checkpoint g{g} guardado")

def _handle_sigusr1(signum, frame):
    log("SIGUSR1 — guardando ola y saliendo con 42...")
    try:
        _save_checkpoint(f"sigusr1-{int(time.time())}")
        DONE["flag"] = True
    except Exception as e:
        log(f"  fallo al guardar: {e}")
    sys.exit(42)

signal.signal(signal.SIGUSR1, _handle_sigusr1)

log("=" * 60)
log("ECOREASONER M1 — Qwen3.5-35B-A3B LoRA 2×MI210 (Tortuga)")
log("=" * 60)
log(f"PyTorch {torch.__version__} | ROCm {torch.cuda.is_available()} | GPUs {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    log(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ── Datos ──────────────────────────────────────────────────────────────
from datasets import Dataset
from transformers import AutoTokenizer

with open(args.data) as f:
    raw = [json.loads(line) for line in f if line.strip()]
log(f"  {len(raw)} ejemplos")

def fmt(ex):
    user = ex.get("prompt", ex.get("user", ex.get("input", "")))
    ans = ex.get("answer", ex.get("assistant", ex.get("output", "")))
    return {"text": f"<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n{ans}<|im_end|>\n"}

tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

dataset = Dataset.from_list([fmt(e) for e in raw])
def _tok(examples):
    return tok(examples["text"], truncation=True, max_length=args.seq_len)
dataset = dataset.map(_tok, batched=True, remove_columns=["text"])
log(f"  dataset tokenizado: {len(dataset)}")

# ── Modelo: device_map balanced + LoRA (patrón smoke test) ─────────────
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

log("Cargando modelo (bf16, device_map balanced en 2×MI210)...")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    attn_implementation="sdpa",
    trust_remote_code=True,
)
model.gradient_checkpointing_enable()
model.train()

target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
lora_cfg = LoraConfig(
    r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=target_modules,
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
DONE["model"] = model
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"  LoRA r={args.lora_rank}: trainable {trainable/1e6:.1f}M ({100*trainable/sum(p.numel() for p in model.parameters()):.5f}%)")

# ── Resume: leer progreso GLOBAL de state.json (no del checkpoint local) ──
# Cada ola reanuda del estado acumulado; el checkpoint guarda pesos, el
# state.json guarda el STEP GLOBAL real (persistente entre olas).
STEPS_DONE = 0
STATE_FILE = OUT / "state.json"
latest = None
if STATE_FILE.exists():
    with open(STATE_FILE) as f:
        state = json.load(f)
    STEPS_DONE = int(state.get("step", 0))
    latest_path = state.get("checkpoint")
    if latest_path and (OUT / latest_path / "adapter").exists():
        latest = OUT / latest_path
if latest is None:
    # fallback: buscar el checkpoint de mayor step global (gN)
    cks = sorted(OUT.glob("checkpoint-g*"), key=lambda p: p.stat().st_mtime)
    if cks:
        latest = cks[-1]

if latest is not None:
    log(f"Reanudando desde checkpoint {latest} (paso global {STEPS_DONE})")
    model.load_adapter(str(latest / "adapter"), adapter_name="default")
    model.set_adapter("default")
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
else:
    log("Sin checkpoint previo — empieza desde base")

# ── Optimizer (SOLO params LoRA) ───────────────────────────────────────
opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
if latest is not None and (latest / "optimizer.pt").exists():
    opt.load_state_dict(torch.load(latest / "optimizer.pt", map_location="cpu")["optimizer"])
DONE["opt"] = opt

# ── Training loop ──────────────────────────────────────────────────────
log(f"Entrenando hasta {args.max_steps} pasos globales (seq {args.seq_len}, bs {args.batch_size}, ga {args.grad_accum}, lr {args.lr})...")
t0 = time.time()
DONE["global_step"] = STEPS_DONE
loss_acc = 0.0

while DONE["global_step"] < args.max_steps:
    DONE["global_step"] += 1
    global_step = DONE["global_step"]
    t1 = time.time()
    opt.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for _ in range(args.grad_accum):
        batch = dataset.shuffle(seed=42 + global_step)[:args.batch_size]
        ids_list = batch["input_ids"]
        max_len = max(len(x) for x in ids_list)
        padded = [x + [tok.pad_token_id] * (max_len - len(x)) for x in ids_list]
        input_ids = torch.tensor(padded, device="cuda")
        labels = input_ids.clone()
        labels[labels == tok.pad_token_id] = -100
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss / args.grad_accum
        loss.backward()
        loss_sum += out.loss.item()
    opt.step()
    loss_acc = loss_sum / args.grad_accum
    DONE["last_loss"] = loss_acc
    dt = time.time() - t1
    log(f"  step {global_step}/{args.max_steps} loss={loss_acc:.4f} ({dt:.1f}s/step)")

    if global_step % args.save_every == 0:
        _save_checkpoint(f"step-{global_step}")

log(f"=== Training completado en {(time.time()-t0)/60:.1f} min (pasos globales: {DONE['global_step']}) ===")
try:
    model.save_pretrained(str(OUT / "final"))
    tok.save_pretrained(str(OUT / "final"))
    with open(OUT / "manifest.json", "w") as f:
        json.dump({"model": args.model, "steps": DONE["global_step"], "lr": args.lr,
                   "seq_len": args.seq_len, "lora_rank": args.lora_rank,
                   "gpu": torch.cuda.get_device_name(0), "elapsed_min": round((time.time()-t0)/60,1)}, f, indent=2)
    log("Modelo final guardado.")
except Exception as e:
    log(f"WARN guardado final: {e}")
print("DONE")