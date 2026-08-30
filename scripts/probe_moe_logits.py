#!/usr/bin/env python3
"""Probe diagnostico: logits degenerados del LLaDA-MoE en GPU? (2026-08-30)."""
import torch, json

DIR = "/beegfs/a474r867/ecoreasoner/models/LLaDA-MoE-7B-A1B-Instruct"
MASK = 156895

from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained(DIR, trust_remote_code=True, local_files_only=True)
msgs = [{"role": "user", "content": "Download bioclim data for temperature."}]
pr = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
if hasattr(pr, "input_ids"):
    pr = pr["input_ids"]
L = pr.shape[1]
print("prompt_len:", L, "| prompt:", repr(tok.decode(pr[0])[:120]))

for impl in ["sdpa", "eager"]:
    m = AutoModel.from_pretrained(DIR, trust_remote_code=True,
                                  dtype=torch.bfloat16, local_files_only=True,
                                  attn_implementation=impl).to("cuda").eval()
    x = torch.full((1, L + 40), MASK, dtype=torch.long, device="cuda")
    x[:, :L] = pr.to("cuda")
    with torch.no_grad():
        out = m(x).logits
    lg = out[0]  # (seq, vocab)
    masked = torch.nonzero(x[0] == MASK).squeeze(-1)
    lm = lg[masked]  # logits en posiciones enmascaradas
    print(f"\n[{impl}] logits masked: mean={lm.float().mean().item():.4f} std={lm.float().std().item():.4f} "
          f"min={lm.float().min().item():.4f} max={lm.float().max().item():.4f} nan={torch.isnan(lm).any().item()}")
    if torch.isnan(lm).any():
        print("  -> NaN detectado")
        continue
    top = torch.topk(lm[0], 5)
    toks = tok.convert_ids_to_tokens(top.indices.tolist())
    print("  pos masked 0:", [(t, round(float(p), 4)) for t, p in zip(toks, top.values.tolist())])
    print("  decode top1:", repr(tok.decode([top.indices[0].item()])))
    top2 = torch.topk(lm[10], 5)
    print("  pos masked 10:", [(t, round(float(p), 4)) for t, p in zip(tok.convert_ids_to_tokens(top2.indices.tolist()), top2.values.tolist())])
    del m; torch.cuda.empty_cache()
print("\nPROBE DONE")