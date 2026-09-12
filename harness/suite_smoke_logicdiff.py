#!/usr/bin/env python3
"""suite_smoke_logicdiff.py — batería L0-L3 con scoring estilo LogicDiff (Fase A).

Extiende suite_smoke_v2: en vez de una sola pasada con máscara aleatoria,
puntúa cada candidato con denoising ORDENADO por rol lógico
(PREMISE → CONNECTIVE → DERIVED → CONCLUSION → FILLER), con revelado
teacher-forced: en cada etapa se mide CE del grupo y se revelan sus tokens
verdaderos antes de seguir. Así, los tokens de CONCLUSION se puntúan con el
máximo contexto del candidato — la prueba inferencial real.

Modos (--modes, separados por coma):
  base        — v2: máscara aleatoria mask_p sobre TODA la secuencia (control)
  dense       — todo el candidato enmascarado de una vez; CE en cand
  staged      — reveal por grupos de rol (reglas) en orden de dependencia
  staged_rev  — mismo pero orden inverso (ablación de dirección)
  staged_rand — grupos aleatorios de igual tamaño (ablación multi-step)
  staged_head — roles predichos por la cabeza sobre posiciones enmascaradas
                (mecanismo LogicDiff genuino; requiere --head)
  payload     — solo tokens "payload" del candidato enmascarados
                (conectivas/verbos-relación/números/entidades; role_score>1)

Cada modo escribe <out-dir>/<mode>/battery_L{0..3}.json + battery.json,
compatible con scripts/verdict.py --battery.

Uso:
  python harness/suite_smoke_logicdiff.py \
      --ckpt runs/f0-span-v3-role/checkpoint-g10000/model.pt \
      --config harness/configs/f0-span-v3-role.yaml \
      --pairs-dir runs/pairs_hard_v3_eval \
      --out-dir runs/logicdiff-eval-v3role \
      --modes base,dense,staged,staged_rand,staged_head,payload \
      --head runs/logicdiff-head-v3role/head.pt \
      --tokenizer /beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
"""
import argparse, json, random, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---- import del modelo v2 ----
_trainer_path = Path(__file__).resolve().parents[1] / "scripts" / "train_mdlm_moe_v2.py"
import importlib.util  # noqa: E402

_argv = sys.argv
sys.argv = [_trainer_path.name, "--data", "dummy", "--output", "/tmp/dummy_ld"]
try:
    _spec = importlib.util.spec_from_file_location("train_mdlm_moe_v2", _trainer_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    MdLMMoE = _mod.MdLMMoE
    _token_role_score = _mod._token_role_score
    _normalize_token_text = _mod._normalize_token_text
finally:
    sys.argv = _argv

ROLE2ID = {"FILLER": 0, "PREMISE": 1, "CONNECTIVE": 2, "DERIVED": 3,
           "CONCLUSION": 4}
STAGE2ROLE = {"OBSERVACION": 1, "HIPOTESIS": 1, "PREDICCION": 3,
              "EVIDENCIA": 3, "CONCLUSION": 4}
CONNECTIVES = {
    "because", "despite", "therefore", "thus", "however", "since", "while",
    "whereas", "yet", "but", "so", "if", "although", "consequently",
    "furthermore", "moreover", "nevertheless", "otherwise", "hence",
    "accordingly", "due", "unless", "besides", "instead", "meanwhile",
    "likewise", "similarly", "conversely",
}
ORDER_FWD = [1, 2, 3, 4, 0]   # PREMISE → CONNECTIVE → DERIVED → CONCLUSION → FILLER
ORDER_REV = [0, 4, 3, 2, 1]
N_ROLES = 5


class RoleHead(torch.nn.Module):
    def __init__(self, hidden, head_hidden, n_roles=N_ROLES):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden, head_hidden), torch.nn.GELU(),
            torch.nn.Linear(head_hidden, n_roles))
    def forward(self, h):
        return self.net(h)


def load_model(args, mcfg):
    dev = torch.device(args.device)
    model = MdLMMoE(
        vocab=mcfg["vocab"], hidden=mcfg["hidden"], layers=mcfg["layers"],
        heads=mcfg["heads"], ff_mult=mcfg["ff_mult"], seq_len=mcfg["seq_len"],
        n_experts=mcfg["n_experts"], k=mcfg["k"],
        use_rope=args.use_rope or mcfg.get("use_rope", False),
        weight_tying=args.weight_tying or mcfg.get("weight_tying", False),
    ).to(dev)
    if args.ckpt.upper() == "RANDOM":
        print("[warn] ckpt=RANDOM -> modelo sin entrenar (control de piso)")
        model.eval()
        return model, dev
    ck = torch.load(args.ckpt, map_location=dev)
    if isinstance(ck, dict) and "model" in ck:
        ck = ck["model"]
    try:
        model.load_state_dict(ck, strict=True)
    except RuntimeError:
        ck = {k[len("module."):]: v for k, v in ck.items()}
        model.load_state_dict(ck, strict=True)
    model.eval()
    return model, dev


def load_head(path, dev):
    d = torch.load(path, map_location=dev)
    cfg = d["config"]
    head = RoleHead(cfg["hidden"], cfg["head_hidden"], cfg["n_roles"]).to(dev)
    head.load_state_dict(d["head"])
    head.eval()
    return head


# ---------------- roles por reglas sobre ids ----------------

def build_label_id_seqs(tok):
    """[(nombre_etapa, ids_de_la_etiqueta)] en el orden de STAGE2ROLE."""
    return [(s, tok.encode(f"[{s}]", add_special_tokens=False))
            for s in STAGE2ROLE]


def rule_roles(ids, tok_strings, label_id_seqs):
    """Roles por reglas para una secuencia de ids (lista o tensor 1D).

    - Contenido entre etiqueta de etapa y la siguiente etiqueta → rol de etapa.
    - Tokens de la propia etiqueta ([CONCLUSION] etc.) → FILLER.
    - Conectivas (texto del token) → CONNECTIVE (prioridad sobre etapa).
    """
    ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
    T = len(ids)
    roles = [0] * T
    # etiquetas de etapa: (start, end, role)
    tags = []
    i = 0
    while i < T:
        matched = False
        for stage, lbl_ids in label_id_seqs:
            L = len(lbl_ids)
            if L and i + L <= T and ids[i:i + L] == lbl_ids:
                tags.append((i, i + L, STAGE2ROLE[stage]))
                i += L
                matched = True
                break
        if not matched:
            i += 1
    tags.sort()
    for ti, (s, e, r) in enumerate(tags):
        cend = tags[ti + 1][0] if ti + 1 < len(tags) else T
        for j in range(e, cend):
            roles[j] = r
    # conectivas: prioridad sobre rol de etapa
    for j in range(T):
        t = _normalize_token_text(tok_strings[ids[j]]).lower()
        t = re.sub(r"[^a-z0-9]", "", t)
        if t in CONNECTIVES:
            roles[j] = 2
    return roles


# ---------------- scoring ----------------

def ce_on(model, xm, pos, seq):
    """forward + sum-CE en posiciones pos del seq (xm = entrada enmascarada)."""
    logits = model(xm.unsqueeze(0)).squeeze(0)
    return F.cross_entropy(logits[pos], seq[pos], reduction="sum"), logits


def score_base(model, seq, mask_p, rng, mask_id):
    T = seq.shape[0]
    n = max(1, int(mask_p * T))
    idx = torch.tensor(rng.sample(range(T), n), dtype=torch.long,
                       device=seq.device)
    masked = seq.clone(); masked[idx] = mask_id
    logits = model(masked.unsqueeze(0)).squeeze(0)
    return F.cross_entropy(logits[idx], seq[idx]).item()


def _staged_ce(model, seq, cand_start, mask_id, order, cand_roles=None,
               head=None):
    """Reveal teacher-forced por grupos de rol. Devuelve (mean_ce, n_tok).

    cand_roles: lista de roles por posición (reglas) o None → la cabeza
    predice roles de las posiciones aún enmascaradas en cada paso.
    """
    T = seq.shape[0]
    dev = seq.device
    cand = torch.arange(cand_start, T, device=dev)
    if cand.numel() == 0:
        return 0.0, 0
    xm = seq.clone()
    xm[cand] = mask_id
    still = cand.clone()          # posiciones cand aún enmascaradas
    total_ce, ntok = 0.0, 0
    for g in order:
        if still.numel() == 0:
            break
        if head is not None:
            with torch.no_grad():
                logits, hidden = model(xm.unsqueeze(0),
                                       output_hidden_states=True)
            logits = logits.squeeze(0)
            pred = head(hidden[0, still]).argmax(-1)
            pos = still[pred == g]
        else:
            logits = model(xm.unsqueeze(0)).squeeze(0)
            rl = torch.tensor([cand_roles[int(p)] for p in still],
                              device=dev)
            pos = still[rl == g]
        if pos.numel() == 0:
            continue
        ce = F.cross_entropy(logits[pos], seq[pos], reduction="sum")
        total_ce += ce.item(); ntok += int(pos.numel())
        xm[pos] = seq[pos]
        still = still[rl != g] if head is None else still[pred != g]
    # remanente (roles fuera de order, no debería pasar)
    if still.numel() > 0:
        logits = model(xm.unsqueeze(0)).squeeze(0)
        ce = F.cross_entropy(logits[still], seq[still], reduction="sum")
        total_ce += ce.item(); ntok += int(still.numel())
    return (total_ce / max(ntok, 1)), ntok


def score_candidate(model, mode, ctx, cand, mask_p, rng, mask_id,
                    tok_strings=None, label_id_seqs=None, head=None,
                    role_cfg=None):
    """Devuelve loss escalar del candidato cand dado ctx."""
    dev = next(model.parameters()).device
    seq = torch.tensor(list(ctx) + list(cand), dtype=torch.long, device=dev)
    cs = len(ctx)
    T = seq.shape[0]
    if cs >= T:
        return float("nan")

    if mode == "base":
        return score_base(model, seq, mask_p, rng, mask_id)

    if mode == "dense":
        pos = torch.arange(cs, T, device=dev)
        xm = seq.clone(); xm[pos] = mask_id
        logits = model(xm.unsqueeze(0)).squeeze(0)
        return F.cross_entropy(logits[pos], seq[pos]).item()

    if mode == "payload":
        # enmascara solo tokens informativos del cand (role_score>1);
        # el resto del cand queda visible (máximo contexto).
        cand_ids = seq[cs:].tolist()
        scores = [_token_role_score(_normalize_token_text(tok_strings[i]),
                                    role_cfg or {}) for i in cand_ids]
        rel = torch.tensor([j for j, s in enumerate(scores) if s > 1.0],
                           device=dev) + cs
        if rel.numel() == 0:
            rel = torch.arange(cs, T, device=dev)
        xm = seq.clone(); xm[rel] = mask_id
        logits = model(xm.unsqueeze(0)).squeeze(0)
        return F.cross_entropy(logits[rel], seq[rel]).item()

    # modos staged
    cand_roles = None
    if mode != "staged_head":
        cand_roles = rule_roles(seq, tok_strings, label_id_seqs)
    if mode == "staged":
        ce, _ = _staged_ce(model, seq, cs, mask_id, ORDER_FWD,
                           cand_roles=cand_roles)
        return ce
    if mode == "staged_rev":
        ce, _ = _staged_ce(model, seq, cs, mask_id, ORDER_REV,
                           cand_roles=cand_roles)
        return ce
    if mode == "staged_head":
        ce, _ = _staged_ce(model, seq, cs, mask_id, ORDER_FWD, head=head)
        return ce
    if mode == "staged_rand":
        # grupos aleatorios con los MISMOS tamaños que los grupos de rol
        cand = list(range(cs, T))
        cr = [cand_roles[j] for j in cand]
        sizes = sorted([cr.count(g) for g in set(cr)])
        rng.shuffle(cand)
        groups, i0 = [], 0
        for s in sizes:
            groups.append(cand[i0:i0 + s]); i0 += s
        xm = seq.clone(); xm[cs:] = mask_id
        total_ce, ntok = 0.0, 0
        for grp in groups:
            if not grp:
                continue
            pos = torch.tensor(grp, dtype=torch.long, device=dev)
            logits = model(xm.unsqueeze(0)).squeeze(0)
            ce = F.cross_entropy(logits[pos], seq[pos], reduction="sum")
            total_ce += ce.item(); ntok += len(grp)
            xm[pos] = seq[pos]
        return total_ce / max(ntok, 1)

    raise ValueError(f"modo desconocido: {mode}")


def eval_mode(model, mode, pairs, ecfg, mask_id, seed, device,
              tok_strings=None, label_id_seqs=None, head=None,
              role_cfg=None):
    rng = random.Random(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    ok_wins, deltas = 0, []
    sub_wins, sub_n = {}, {}
    with torch.no_grad():
        for item in pairs:
            ctx, ok, bad = item[:3]
            subtype = item[3] if len(item) > 3 else None
            l_ok = score_candidate(model, mode, ctx, ok, ecfg["mask_p"], rng,
                                   mask_id, tok_strings, label_id_seqs, head,
                                   role_cfg)
            l_bad = score_candidate(model, mode, ctx, bad, ecfg["mask_p"], rng,
                                    mask_id, tok_strings, label_id_seqs, head,
                                    role_cfg)
            w = l_ok < l_bad
            ok_wins += w
            deltas.append(l_bad - l_ok)
            if subtype:
                sub_wins[subtype] = sub_wins.get(subtype, 0) + int(w)
                sub_n[subtype] = sub_n.get(subtype, 0) + 1
    rep = {
        "pairwise_acc": round(ok_wins / len(pairs), 4),
        "n_pairs": len(pairs),
        "mean_delta": round(float(np.mean(deltas)) if deltas else 0.0, 5),
        "elapsed_s": round(time.time() - t0, 1),
    }
    if sub_n:
        rep["l3_subtype_acc"] = {
            s: {"acc": round(sub_wins[s] / sub_n[s], 4), "n": sub_n[s]}
            for s in sorted(sub_n)}
    return rep


def _load_pairs(path, max_ctx, max_cand):
    pairs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ctx, ok, bad = rec["ctx"][:max_ctx], rec["ok"][:max_cand], rec["bad"][:max_cand]
        if len(ctx) >= 2 and len(ok) and len(bad):
            pairs.append((ctx, ok, bad, rec.get("l3_subtype")))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--modes", default="base,dense,staged,staged_rand,payload")
    ap.add_argument("--head", default=None, help="head.pt (modo staged_head)")
    ap.add_argument("--tokenizer", default=None,
                    help="necesario para reglas/payload (id->string)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mask-id", type=int, default=None)
    ap.add_argument("--use-rope", action="store_true")
    ap.add_argument("--weight-tying", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="limitar pares por nivel (debug)")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg, ecfg = cfg["model"], cfg["eval"]
    mask_id = args.mask_id if args.mask_id is not None else mcfg["vocab"]
    seed = cfg["seed"]; tag = cfg["out"]["tag"]

    model, dev = load_model(args, mcfg)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    need_tok = any(m in ("staged", "staged_rev", "staged_rand", "payload")
                   for m in modes)
    tok_strings, label_id_seqs = None, None
    if need_tok:
        if not args.tokenizer:
            ap.error("los modos rule/payload requieren --tokenizer")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                            local_files_only=True,
                                            trust_remote_code=True)
        tok_strings = tok.convert_ids_to_tokens(list(range(mcfg["vocab"])))
        label_id_seqs = build_label_id_seqs(tok)

    head = None
    if "staged_head" in modes:
        if not args.head:
            ap.error("staged_head requiere --head")
        head = load_head(args.head, dev)

    role_cfg = None
    try:
        role_cfg = json.loads(cfg["train"].get("role_config", "") or "{}")
    except Exception:
        role_cfg = {}

    pairs_dir = Path(args.pairs_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(pairs_dir.glob("pairs_L*.jsonl"))
    max_ctx, max_cand = mcfg["seq_len"] // 2, mcfg["seq_len"] // 4

    results = {}
    for mode in modes:
        mdir = out_dir / mode
        mdir.mkdir(parents=True, exist_ok=True)
        results[mode] = {}
        for fp in files:
            lvl = fp.stem.split("_L")[-1]
            if lvl not in ("0", "1", "2", "3"):
                continue
            pairs = _load_pairs(fp, max_ctx, max_cand)
            if args.limit:
                pairs = pairs[:args.limit]
            if not pairs:
                continue
            rep = eval_mode(model, mode, pairs, ecfg, mask_id, seed, dev,
                            tok_strings, label_id_seqs, head, role_cfg)
            rep.update({"tag": f"{tag}_{mode}", "level": f"L{lvl}",
                        "mode": mode, "seed": seed})
            (mdir / f"battery_L{lvl}.json").write_text(
                json.dumps({"discrimination": rep, "mode": mode}, indent=2))
            results[mode][lvl] = rep
            print(json.dumps({"mode": mode, "level": f"L{lvl}",
                              "pairwise_acc": rep["pairwise_acc"],
                              "mean_delta": rep["mean_delta"],
                              "n": rep["n_pairs"],
                              "s": rep["elapsed_s"]}), flush=True)
        # resumen del modo (formato battery.json de v2)
        lv = results[mode]
        if lv:
            summ = {
                "tag": f"{tag}_{mode}", "seed": seed, "mode": mode,
                "levels": {f"L{k}": v for k, v in lv.items()},
                "discrimination": {
                    "pairwise_acc": round(
                        sum(v["pairwise_acc"] for v in lv.values()) / len(lv), 4),
                    "n_pairs": sum(v["n_pairs"] for v in lv.values()),
                    "mean_delta": round(
                        sum(v["mean_delta"] for v in lv.values()) / len(lv), 5),
                },
            }
            (mdir / "battery.json").write_text(json.dumps(summ, indent=2))
    (out_dir / "logicdiff_summary.json").write_text(
        json.dumps({"ckpt": args.ckpt, "modes": results,
                    "seed": seed, "pairs_dir": str(pairs_dir)}, indent=2))
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
