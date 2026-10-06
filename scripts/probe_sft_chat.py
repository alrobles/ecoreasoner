"""probe_sft_chat.py — probe de chat contra ckpt SFT-v2 (o cualquier CKPT).

Reutiliza loader+decode por confianza de probe_chat_v5.py. Prompts via env:
PROMPTS separados por '|||' (cada uno se envuelve en [USER]...[ASSISTANT]
salvo que ya traiga formato), o PROMPT singular.
Default: ckpt EMA g35500 de runs/sft-v2 + 3 preguntas de sondeo.

Uso (cluster): sbatch --export=ALL,CKPT=...,PROMPTS="q1|||q2" probe_sft_chat.slurm
"""
import os, sys
from pathlib import Path
import torch

BASE = Path("/beegfs/a474r867/ecoreasoner")
os.environ.setdefault("CKPT", str(BASE/"runs/sft-v2/checkpoint-g35500/ema_model.pt"))
sys.path.insert(0, str(BASE/"scripts"))
import probe_chat_v5 as P

MAX_NEW = int(os.environ.get("MAX_NEW", "128"))
STEPS = int(os.environ.get("STEPS", "96"))
TEMP = float(os.environ.get("TEMP", "0.0"))

DEFAULT_QS = [
    "What is the difference between a species' fundamental niche and its realized niche?",
    "Explain photosynthesis to a ten year old.",
    "Dame una recomendacion para empezar a leer sobre ecologia.",
]

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = P.load_model(dev)
    n_par = sum(p.numel() for p in model.parameters())/1e9
    print(f"[probe-sft] ckpt={P.CKPT}\n[probe-sft] params={n_par:.2f}B dev={dev}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["TOK"], trust_remote_code=True)

    raw = os.environ.get("PROMPTS") or os.environ.get("PROMPT") or "|||".join(DEFAULT_QS)
    qs = raw.split("|||")
    for i, q in enumerate(qs):
        prompt = q if "[USER]" in q else f"[USER] {q}\n[ASSISTANT]"
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = ids[-(P.SEQ_LEN - MAX_NEW):]
        out = P.gen(model, ids, MAX_NEW, STEPS, TEMP, dev)
        new = [t for t in out[len(ids):] if t != P.MASK_ID]
        stop_ids = {0, tok.eos_token_id, tok.pad_token_id} - {None}
        cut = next((j for j, t in enumerate(new) if t in stop_ids), len(new))
        print(f"\n===== chat {i} =====\nPROMPT: {prompt}\n"
              f"EOS@={cut}/{len(new)}\nOUTPUT: {tok.decode(new[:cut])}\n"
              f"FULL: {tok.decode(new)}")

if __name__ == "__main__":
    main()
