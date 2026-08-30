#!/usr/bin/env python3
"""Parche modeling_lladamoe.py para transformers 5.15 (2026-08-30)."""
import sys

p = "/beegfs/a474r867/ecoreasoner/models/LLaDA-MoE-7B-A1B-Instruct/modeling_lladamoe.py"
s = open(p, encoding="utf-8").read()

old_init = '''        self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type, ROPE_INIT_FUNCTIONS["llama3"])

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)'''
new_init = '''        # PATCH 2026-08-30 (transformers 5.15): ROPE_INIT_FUNCTIONS elimino "default" y
        # las init nuevas exigen rope_parameters (KeyError factor). RoPE estandar manual:
        # theta=config.rope_theta, dim=config.hidden_size, scaling=1 (equivalente al original).
        _d = getattr(self.config, "head_dim", None) or (getattr(self.config, "hidden_size", 2048) // max(1, getattr(self.config, "num_attention_heads", 1)))
        _theta = getattr(self.config, "rope_theta", 50000.0) or 50000.0
        inv_freq = 1.0 / (_theta ** (torch.arange(0, _d, 2, device=(device or "cpu"), dtype=torch.float32).float() / _d))
        self.attention_scaling = 1.0'''
assert old_init in s, "init block no encontrado"
s = s.replace(old_init, new_init)

# rope_type sin "dynamic" para que forward() nunca llame _dynamic_frequency_update
s = s.replace('rope_type="llama3"', 'rope_type="linear"')
s = s.replace('self.rope_type = "llama3"', 'self.rope_type = "linear"')

open(p, "w", encoding="utf-8").write(s)
print("patched OK")