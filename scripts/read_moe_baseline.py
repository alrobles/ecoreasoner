#!/usr/bin/env python3
"""Lee smoke_moe_baseline.json y muestra resumen + muestras."""
import json

p = "/beegfs/a474r867/ecoreasoner/data/l1/smoke_moe_baseline.json"
d = json.load(open(p))
print("elapsed_s:", d.get("elapsed_s"), "| ckpt:", d.get("ckpt"))
for k, v in {**d.get("modes", {}), **d.get("chat", {})}.items():
    n = v.get("n"); nf = v.get("n_fn_valid")
    print(f"\n=== {k}: {nf}/{n} fn_valid ({100*v.get('fn_rate',0):.0f}%) json={v.get('n_json')} repair={v.get('repair_fn')} ===")
    for s in v.get("samples", []):
        print(f"  [{s.get('kind')}] ok_fn={s.get('ok_fn')} fn={s.get('fn')}")
        print(f"    GEN: {repr(s.get('gen','')[:240])}")