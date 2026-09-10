#!/usr/bin/env python3
"""Mini-test local para el curriculum de masking de train_mdlm_moe_v2.py (V3.1).

Comprueba que la funcion curriculum_state interpola span_len y b_h segun
el esquema del roadmap y es monotona no decreciente.
"""
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "scripts" / "train_mdlm_moe_v2.py"
OUT = Path("/tmp/curriculum_test_out")


def load_trainer():
    """Carga el trainer con argumentos dummy para poder importar curriculum_state."""
    _argv = sys.argv
    sys.argv = [TRAINER.name, "--data", "dummy", "--output", str(OUT)]
    try:
        spec = importlib.util.spec_from_file_location(
            "train_mdlm_moe_v2", str(TRAINER)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = _argv
    return mod


def main():
    mod = load_trainer()
    curriculum_state = mod.curriculum_state
    stages = json.loads(mod.CUR_STAGES_DEFAULT)

    print(f"stages default: {stages}")

    checks = [
        (0, 16, 0.30),
        (2000, 32, 0.60),
        (5000, 64, 0.95),
        (10000, 64, 0.95),
    ]
    for step, exp_span, exp_frac in checks:
        span, frac = curriculum_state(step, stages)
        print(f"step {step:5d} -> span_len={span}, b_h={frac:.4f}")
        assert span == exp_span, f"span mismatch en step {step}: {span} != {exp_span}"
        assert math.isclose(frac, exp_frac, rel_tol=1e-9, abs_tol=1e-9), (
            f"frac mismatch en step {step}: {frac} != {exp_frac}"
        )

    # monotonicidad no decreciente
    spans, fracs = [], []
    for step in range(-500, 11000, 100):
        span, frac = curriculum_state(step, stages)
        spans.append(span)
        fracs.append(frac)
    for i in range(1, len(spans)):
        assert spans[i] >= spans[i - 1] - 1e-9, f"span decreciente en idx {i}"
        assert fracs[i] >= fracs[i - 1] - 1e-9, f"b_h decreciente en idx {i}"
    print("[ok] monotonicidad no decreciente de span_len y b_h")

    # interpolacion suave en el interior de la primera etapa
    s1000, f1000 = curriculum_state(1000, stages)
    print(f"step  1000 -> span_len={s1000}, b_h={f1000:.4f}")
    assert 16 < s1000 < 32, f"step 1000 span_len no interpolado: {s1000}"
    assert 0.30 < f1000 < 0.60, f"step 1000 b_h no interpolado: {f1000}"

    # cleanup del dummy output
    if OUT.exists():
        shutil.rmtree(OUT, ignore_errors=True)

    print("[ok] todos los checks del curriculum pasaron")


if __name__ == "__main__":
    main()
