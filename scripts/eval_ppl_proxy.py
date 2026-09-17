#!/usr/bin/env python3
"""eval_ppl_proxy.py — pseudo-perplejidad para el dLLM enmascarado.

Un dLLM no define perplexity autoregresiva; el proxy estandar es la CE de
denoising a fracciones de mascara fijas sobre texto held-out. Aqui el
held-out son los documentos CORRECTOS del holdout v4 (ctx+ok): nunca se
tocan las etiquetas ni el miembro `bad`, asi no contamina la metrica de
razonamiento. Determinista: seed fija -> mismas mascaras para todos los
checkpoints (comparable entre runs).

Salida JSON: ce_mean, ppl_proxy=exp(ce_mean), desglose por fraccion.

Uso:
  python3 eval_ppl_proxy.py --ckpt RUN/model.pt --model-override '{...}' \
      --pairs-dir runs/pairs_hard_v4_holdout --out RUN/ppl_proxy.json
"""
import argparse, importlib.util, json, math, random, sys
from pathlib import Path

import torch
import torch.nn.functional as F

_trainer_path = Path(__file__).resolve().parent / "train_mdlm_moe_v2.py"
_argv = sys.argv
sys.argv = [_trainer_path.name, "--data", "dummy", "--output", "/tmp/dummy_ppl"]
import signal as _signal  # noqa: E402
_prev_usr1 = _signal.getsignal(_signal.SIGUSR1)
try:
    _spec = importlib.util.spec_from_file_location(
        "train_mdlm_moe_v2", _trainer_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    MdLMMoE = _mod.MdLMMoE
finally:
    sys.argv = _argv
    _signal.signal(_signal.SIGUSR1, _prev_usr1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-override", required=True)
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--pairs-file", default="pairs_L3.jsonl",
                    help="nivel del que tomar docs (ctx+ok = doc correcto)")
    ap.add_argument("--n-docs", type=int, default=192)
    ap.add_argument("--fracs", default="0.15,0.50,0.85")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = torch.device(a.device)
    mcfg = json.loads(a.model_override)
    model = MdLMMoE(
        vocab=mcfg["vocab"], hidden=mcfg["hidden"], layers=mcfg["layers"],
        heads=mcfg["heads"], ff_mult=mcfg["ff_mult"],
        seq_len=mcfg["seq_len"], n_experts=mcfg["n_experts"], k=mcfg["k"],
    ).to(dev)
    ck = torch.load(a.ckpt, map_location=dev)
    if isinstance(ck, dict) and "model" in ck:
        ck = ck["model"]
    if isinstance(ck, dict) and "ema_model" in ck:
        ck = ck["ema_model"]
    try:
        model.load_state_dict(ck, strict=True)
    except RuntimeError:
        ck = {k[len("module."):]: v for k, v in ck.items()}
        model.load_state_dict(ck, strict=True)
    model.eval()
    MASK = mcfg["vocab"]

    docs = []
    pf = Path(a.pairs_dir) / a.pairs_file
    for line in open(pf):
        d = json.loads(line)
        seq = d["ctx"] + d["ok"]
        if len(seq) >= 32:
            docs.append(seq[: mcfg["seq_len"]])
        if len(docs) >= a.n_docs:
            break
    fracs = [float(x) for x in a.fracs.split(",")]
    per = {}
    with torch.no_grad():
        for f in fracs:
            ces = []
            for s in docs:
                x = torch.tensor(s, device=dev)
                t = x.numel()
                m = torch.rand(t, device=dev) < f
                if not bool(m.any()):
                    m[torch.randint(0, t, (1,), device=dev)] = True
                xm = x.clone()
                xm[m] = MASK
                out = model(xm.unsqueeze(0))
                ces.append(F.cross_entropy(out[0][m], x[m]).item())
            ce = sum(ces) / len(ces)
            per[f"f{f}"] = {"ce": ce, "ppl": math.exp(ce), "n": len(ces)}
    pooled = sum(r["ce"] * r["n"] for r in per.values()) / \
        sum(r["n"] for r in per.values())
    out = {"ckpt": a.ckpt, "pairs": str(pf), "n_docs": len(docs),
           "seed": a.seed, "ce_mean": pooled,
           "ppl_proxy": math.exp(pooled), "per_frac": per}
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"ppl_proxy": out["ppl_proxy"], "ce_mean": pooled,
                      "n_docs": len(docs)}))


if __name__ == "__main__":
    main()
