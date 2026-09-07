#!/usr/bin/env python3
"""validate_configs.py — valida los run.yaml del harness F0 (fase 3 -new).
Evita el bug del 07-09 (flow-mapping YAML inválido en `train:` que mató la
eval de 3 micro-runs completados).

Uso: python3 harness/validate_configs.py            # valida los 6 f0-*.yaml
Exit 0 si todos válidos, 1 si alguno falla.
"""
import glob, sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[fatal] pyyaml requerido", file=sys.stderr)
    sys.exit(2)

HERE = Path(__file__).resolve().parent
CONFIGS = sorted(glob.glob(str(HERE / "configs" / "f0-*.yaml")))


def main() -> int:
    bad = 0
    for p in CONFIGS:
        try:
            cfg = yaml.safe_load(Path(p).read_text())
            t = cfg["train"]
            assert t["mask_type"] in ("random", "span")
            assert isinstance(t["mask_p"], (int, float))
            assert isinstance(t["span_len"], int)
            assert isinstance(t["data_cache"], str) and t["data_cache"]
            assert cfg["model"]["n_experts"] == 1
            print(f"OK   {Path(p).name}")
        except Exception as e:
            bad += 1
            print(f"FAIL {Path(p).name}: {e}")
    if bad:
        print(f"[fatal] {bad}/{len(CONFIGS)} configs inválidos", file=sys.stderr)
        return 1
    print(f"[ok] {len(CONFIGS)} configs válidos")
    return 0


if __name__ == "__main__":
    sys.exit(main())