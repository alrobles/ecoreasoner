#!/usr/bin/env python3
"""Lee smoke_moe_lora.json y muestra las 40 muestras con detalle."""
import json

d = json.load(open("/beegfs/a474r867/ecoreasoner/data/l1/smoke_moe_lora.json"))
print(f"=== {d['n_fn_valid']}/{d['n']} fn_valid ({100*d['fn_rate']:.0f}%) | json={d['n_json']} repair={d['repair_fn']} ===")
ok = [s for s in d["samples"] if s["ok_fn"]]
print(f"\nVALIDAS ({len(ok)}):")
for s in ok:
    print(f"  [{s['kind']}] fn={s['fn']} | {repr(s['gen'][:200])}")
print("\nJSON-but-not-valid:")
for s in d["samples"]:
    if s["ok_json"] and not s["ok_fn"]:
        print(f"  [{s['kind']}] fn={s['fn']} | {repr(s['gen'][:220])}")
print("\nMuestras (10 primeras no vacias):")
c = 0
for s in d["samples"]:
    if s["gen"].strip():
        print(f"  [{s['kind']}] {repr(s['gen'][:180])}")
        c += 1
        if c >= 10:
            break