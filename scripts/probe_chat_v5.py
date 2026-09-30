"""probe_chat_v5.py — probe generativo rapido del backbone v5-1b (EMA).

Decodificacion por confianza (estilo LLaDA): prompt visible, ventana de
max_new MASK, revela top-k por confianza en cada paso. Corre 3 probes:
skeleton-continuation (home turf), QA inline, y formato chat [USER]/[ASSISTANT].

Uso (cluster): srun/sbatch con apptainer exec --nv $SIF python3 probe_chat_v5.py
  CKPT=.../ema_model.pt TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/<sha>
"""
import importlib.util, os, sys, signal
from pathlib import Path
import torch

BASE = Path("/beegfs/a474r867/ecoreasoner")
CKPT = os.environ.get("CKPT", str(BASE/"runs/v5-1b/battery_curve/ckpts/g495501/ema_model.pt"))
TOKDIR = os.environ.get("TOK", "")
STEPS = int(os.environ.get("STEPS", "48"))
MAX_NEW = int(os.environ.get("MAX_NEW", "96"))
TEMP = float(os.environ.get("TEMP", "0.0"))   # 0 = argmax
SEQ_LEN = 768

ARCH = dict(vocab=126080, hidden=768, layers=12, heads=12, ff_mult=4,
            seq_len=SEQ_LEN, n_experts=16, k=1, use_rope=False,
            weight_tying=False, n_shared=0, shared_ff_div=4, fine_ff_div=1)
MASK_ID = ARCH["vocab"]

def load_v2_class():
    p = BASE/"scripts"/"train_mdlm_moe_v2.py"
    argv_bak = sys.argv
    sig_bak = signal.getsignal(signal.SIGUSR1)
    try:
        sys.argv = [p.name, "--data", "dummy", "--output", "/tmp/dummy_probe"]
        spec = importlib.util.spec_from_file_location("tmm2", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.MdLMMoE
    finally:
        sys.argv = argv_bak
        signal.signal(signal.SIGUSR1, sig_bak)

def load_model(dev):
    MdLMMoE = load_v2_class()
    model = MdLMMoE(**ARCH).to(dev)
    ck = torch.load(CKPT, map_location="cpu")
    sd = ck.get("model", ck.get("ema_model", ck)) if isinstance(ck, dict) else ck
    if all(k.startswith("module.") for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

@torch.no_grad()
def gen(model, ids, max_new, steps, temp, dev):
    import torch.nn.functional as F
    x = torch.tensor(ids + [MASK_ID]*max_new, dtype=torch.long, device=dev)[None, :]
    pos0 = len(ids)
    for _ in range(steps):
        masked = (x == MASK_ID).nonzero(as_tuple=True)[1]
        if masked.numel() == 0:
            break
        logits = model(x).squeeze(0)
        lp = F.log_softmax(logits[masked].float(), -1)
        if temp > 0:
            pred = torch.multinomial((lp/temp).softmax(-1), 1).squeeze(-1)
            conf = lp.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        else:
            conf, pred = lp.max(-1)
        k = max(1, -(-masked.numel() // steps))   # revela ~1/steps por paso
        top = conf.topk(min(k, masked.numel())).indices
        x[0, masked[top]] = pred[top]
    return x[0].tolist()

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(dev)
    n_par = sum(p.numel() for p in model.parameters())/1e9
    print(f"[probe] ckpt={CKPT}\n[probe] params={n_par:.2f}B dev={dev}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKDIR, trust_remote_code=True)
    probes = [
        ("skeleton-continuation",
         "[BACKGROUND]Species distribution models estimate habitat suitability from occurrence records. "
         "[METHODS]We fitted MaxEnt models for 14 boreal tree species using GBIF occurrences and CHELSA bioclimatic variables. "
         "[RESULTS]Model performance was strong for widespread species"),
        ("QA-inline",
         "Question: What is the main driver of species richness in tropical forests?\nAnswer:"),
        ("chat-format",
         "[USER]What is a species distribution model?[ASSISTANT]"),
        ("chat-format-2",
         "[USER]Explain in one sentence why photosynthesis matters for ecosystems.[ASSISTANT]"),
    ]
    for name, prompt in probes:
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = ids[-(SEQ_LEN - MAX_NEW):]
        out = gen(model, ids, MAX_NEW, STEPS, TEMP, dev)
        new = [t for t in out[len(ids):] if t != MASK_ID]
        print(f"\n===== {name} =====\nPROMPT: {prompt[:120]}...\nOUTPUT: {tok.decode(new)}")

if __name__ == "__main__":
    main()
