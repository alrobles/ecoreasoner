#!/usr/bin/env python3
"""test_role_mask.py — prueba local del masking role-aware V3.2.

Comprueba:
  (a) sin --role_mask el comportamiento es identico al V3.1 con la misma seed.
  (b) con --role_mask la fraccion de tokens informativos enmascarados es
      significativamente mayor que la de tokens neutrales (random y span).
  (c) compatibilidad con curriculum activo: role_mask respeta span_len/b_h
      del curriculum y sigue priorizando tokens de rol.
"""
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "scripts" / "train_mdlm_moe_v2.py"
OUT = Path(tempfile.mkdtemp(prefix="role_mask_test_"))


def load_trainer():
    """Carga el trainer con argumentos dummy para poder importar las funciones."""
    _argv = sys.argv
    sys.argv = [TRAINER.name, "--output", str(OUT), "--data", "dummy"]
    try:
        spec = importlib.util.spec_from_file_location(
            "train_mdlm_moe_v2", str(TRAINER)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = _argv
    return mod


class FakeTokenizer:
    """Tokenizer de juguete que imita convert_ids_to_tokens de HuggingFace."""

    def __init__(self, id2str):
        self.id2str = id2str
        self.vocab_size = max(id2str.keys()) + 1

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.id2str.get(ids, f"<tok{ids}>")
        return [self.id2str.get(i, f"<tok{i}>") for i in ids]


# Vocabulario de juguete. El prefijo '▁' simula SentencePiece: la funcion
# _normalize_token_text lo elimina antes de mirar palabras clave.
ID2STR = {
    0: "<pad>",
    1: "▁the",
    2: "▁cat",
    3: "▁because",
    4: "▁increases",
    5: "▁Panthera",
    6: "▁tigris",
    7: "▁123",
    8: "▁however",
    9: "▁and",
    10: "▁reduces",
    11: "▁despite",
    12: "▁52",
    13: "▁Quercus",
    14: "▁ilex",
    15: "▁although",
    16: "▁green",
    17: "▁tree",
    18: "▁sleeps",
    19: "▁fast",
    20: "▁above",
    21: "▁field",
    22: "▁of",
    23: "▁a",
    24: "▁runs",
    25: "▁under",
    26: "▁sun",
    27: "▁City",
    28: "▁the",
    29: "▁cat",
    30: "<s>",
    31: "</s>",
}


def make_inputs():
    tok = FakeTokenizer(ID2STR)
    xb = torch.arange(32, dtype=torch.long).unsqueeze(0)  # [1, 32]
    return tok, xb


def test_default_random_matches_baseline(mod):
    torch.manual_seed(12345)
    xb = torch.arange(32, dtype=torch.long).unsqueeze(0)
    n = 8
    got = mod.build_mask_indices(xb, n, "random", device="cpu")
    torch.manual_seed(12345)
    expected = torch.randperm(32)[:n]
    assert torch.equal(got, expected), f"random default mismatch: {got} != {expected}"
    print("[ok] (a) random default == torch.randperm con misma seed")


def test_default_span_matches_baseline(mod):
    torch.manual_seed(54321)
    xb = torch.arange(32, dtype=torch.long).unsqueeze(0)
    n = 12
    span_len = 8
    got = mod.build_mask_indices(xb, n, "span", span_len=span_len, device="cpu")
    torch.manual_seed(54321)
    expected = mod.build_span_mask(32, n, span_len, device="cpu")
    assert torch.equal(got, expected), f"span default mismatch: {got} != {expected}"
    print("[ok] (a) span default == build_span_mask con misma seed")


def test_role_scores(mod, tok, xb):
    mod._ROLE_SCORE_CACHE.clear()
    cfg = mod.ARGS.role_config
    scores = mod.get_token_role_scores(tok, xb, cfg=cfg)
    assert scores.shape == (1, 32), scores.shape
    # comprobaciones de puntajes
    assert abs(scores[0, 1].item() - 1.0) < 1e-6, "the deberia ser neutro"
    assert abs(scores[0, 3].item() - 4.0) < 1e-6, "because deberia tener bonus 3.0"
    assert abs(scores[0, 5].item() - 3.5) < 1e-6, "Panthera = entity + uppercase"
    assert abs(scores[0, 7].item() - 3.0) < 1e-6, "123 = number"
    assert scores[0, 27].item() > 1.0, "City (mayuscula) deberia tener bonus"
    print("[ok] role scores calculados correctamente")
    return scores


def test_role_bias_random(mod, tok, xb):
    mod._ROLE_SCORE_CACHE.clear()
    cfg = mod.ARGS.role_config
    role_scores = mod.get_token_role_scores(tok, xb, cfg=cfg)
    role_mask = (role_scores[0] > 1.0 + 1e-6)
    n_trials = 300
    n = 10
    counts = torch.zeros(32)
    for t in range(n_trials):
        torch.manual_seed(9000 + t)
        idx = mod.build_mask_indices(xb, n, "random", device="cpu", token_scores=role_scores)
        counts[idx] += 1
    probs = counts / n_trials
    role_avg = probs[role_mask].mean().item()
    non_avg = probs[~role_mask].mean().item()
    ratio = role_avg / max(non_avg, 1e-9)
    print(f"[info] random role_avg={role_avg:.3f} non_role_avg={non_avg:.3f} ratio={ratio:.2f}")
    assert role_avg > non_avg, "role random: los tokens informativos no se enmascaran mas"
    assert ratio > 1.5, f"role random ratio demasiado bajo: {ratio:.2f}"
    print("[ok] (b) random role-aware enmascara mas tokens informativos")


def test_role_bias_span(mod, tok, xb):
    mod._ROLE_SCORE_CACHE.clear()
    cfg = mod.ARGS.role_config
    role_scores = mod.get_token_role_scores(tok, xb, cfg=cfg)
    role_mask = (role_scores[0] > 1.0 + 1e-6)
    n_trials = 200
    n = 16
    span_len = 8
    role_tok_masked = 0
    total_tok_masked = 0
    for t in range(n_trials):
        torch.manual_seed(20000 + t)
        idx = mod.build_mask_indices(xb, n, "span", span_len=span_len,
                                     device="cpu", token_scores=role_scores)
        total_tok_masked += idx.numel()
        role_tok_masked += int(role_mask[idx].sum().item())
    role_frac = role_tok_masked / max(total_tok_masked, 1)
    baseline_frac = role_mask.sum().item() / 32
    print(f"[info] span role_frac_masked={role_frac:.3f} baseline_frac={baseline_frac:.3f}")
    assert role_frac > baseline_frac, "role span no eleva la fraccion de tokens de rol"
    assert role_frac > baseline_frac * 1.2, f"role span mejora insuficiente: {role_frac:.3f} vs {baseline_frac:.3f}"
    print("[ok] (b) span role-aware enmascara mas regiones informativas")


def test_curriculum_compatibility(mod, tok, xb):
    mod._ROLE_SCORE_CACHE.clear()
    cfg = mod.ARGS.role_config
    role_scores = mod.get_token_role_scores(tok, xb, cfg=cfg)
    role_mask = (role_scores[0] > 1.0 + 1e-6)
    stages = json.loads(mod.CUR_STAGES_DEFAULT)
    T = 32
    results = []
    for step in (1000, 1500):
        span, b_h = mod.curriculum_state(step, stages)
        n_target = max(1, int(T * b_h))
        n_trials = 100
        role_tok_masked = 0
        total_tok_masked = 0
        for t in range(n_trials):
            torch.manual_seed(30000 + step + t)
            idx = mod.build_mask_indices(xb, n_target, "span", span_len=span,
                                         device="cpu", token_scores=role_scores)
            total_tok_masked += idx.numel()
            role_tok_masked += int(role_mask[idx].sum().item())
        role_frac = role_tok_masked / max(total_tok_masked, 1)
        baseline_frac = role_mask.sum().item() / T
        results.append((step, span, b_h, n_target, role_frac, baseline_frac))
        print(f"[info] curriculum step={step}: span={span} b_h={b_h:.3f} n_target={n_target} "
              f"role_frac={role_frac:.3f} baseline={baseline_frac:.3f}")
        assert role_frac > baseline_frac, f"curriculum step {step} no prioriza role tokens"
    for step, span, b_h, n_target, role_frac, baseline in results:
        assert span > 0, f"span_len invalido en step {step}"
        assert 0.0 <= b_h <= 1.0, f"b_h invalido en step {step}"
    print("[ok] (c) role_mask compatible con curriculum (span_len/b_h variables)")


def main():
    mod = load_trainer()
    tok, xb = make_inputs()

    # (a) default OFF
    mod.ARGS.role_mask = False
    test_default_random_matches_baseline(mod)
    test_default_span_matches_baseline(mod)

    # role scores
    test_role_scores(mod, tok, xb)

    # (b) role ON
    mod.ARGS.role_mask = True
    test_role_bias_random(mod, tok, xb)
    test_role_bias_span(mod, tok, xb)

    # (c) curriculum + role
    mod.ARGS.curriculum = True
    mod.ARGS.cur_stages = json.loads(mod.CUR_STAGES_DEFAULT)
    test_curriculum_compatibility(mod, tok, xb)

    # cleanup
    if OUT.exists():
        shutil.rmtree(OUT, ignore_errors=True)

    print("[ok] todos los tests de role_mask pasaron")


if __name__ == "__main__":
    main()
