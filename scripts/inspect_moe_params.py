#!/usr/bin/env python3
"""Inspecciona nombres de params del LLaDA-MoE para LoRA manual."""
from transformers import AutoModel
import re

DIR = "/beegfs/a474r867/ecoreasoner/models/LLaDA-MoE-7B-A1B-Instruct"
m = AutoModel.from_pretrained(DIR, trust_remote_code=True,
                              dtype="bfloat16", local_files_only=True)
names = list(dict(m.named_parameters()).keys())
print("TOTAL params:", sum(p.numel() for p in m.parameters()))
print("primeros 15:", names[:15])
# patrones de capas lineales
lins = [n for n in names if any(x in n for x in ("q_proj", "k_proj", "v_proj", "o_proj", "gate", "down_proj", "up_proj"))]
print("\nLINEALES (sample 25):")
for n in lins[:25]:
    print("  ", n)
# conteo por tipo de capa
from collections import Counter
pat = Counter()
for n in lins:
    base = n.split(".")[-1]
    pat[base] += 1
print("\nconteo por capa:", dict(pat))