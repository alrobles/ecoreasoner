#!/usr/bin/env python3
"""Fix 2: head_dim en el parche RoPE (2026-08-30)."""
p = "/beegfs/a474r867/ecoreasoner/models/LLaDA-MoE-7B-A1B-Instruct/modeling_lladamoe.py"
s = open(p, encoding="utf-8").read()

old = '_d = getattr(self.config, "hidden_size", None) or self.rope_kwargs.get("dim") or 2048'
new = ('_d = getattr(self.config, "head_dim", None) or '
       '(getattr(self.config, "hidden_size", 2048) // max(1, getattr(self.config, "num_attention_heads", 1)))')
assert old in s, "linea no encontrada"
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("fix2 OK")