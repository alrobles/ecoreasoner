#!/usr/bin/env python3
"""train_mdlm_moe_v3_contrastive.py — Fine-tuning contrastivo L3 para dLLM.

Toma un checkpoint MdLMMoE preentrenado y ajusta con ranking sobre pares
L0-L3 (especialmente L3) + una pequena perdida MLM auxiliar.

El objetivo es que el modelo asigne mayor log-likelihood condicional a la
continuacion cientificamente correcta (ok) que al hard negative (bad).

Uso:
  /bb/bwvenv/bin/python /beegfs/a474r867/ecoreasoner/scripts/train_mdlm_moe_v3_contrastive.py \
      --ckpt /beegfs/a474r867/ecoreasoner/runs/f0-span-esqueleto-v2-piloto/checkpoint-g10000/model.pt \
      --config /beegfs/a474r867/ecoreasoner/harness/configs/f0-span-esqueleto-v2.yaml \
      --pairs /beegfs/a474r867/ecoreasoner/runs/pairs_hard_v3/pairs_L3.jsonl \
      --data_cache /beegfs/a474r867/ecoreasoner/data/train_ids_skeleton_v2.npz \
      --output /beegfs/a474r867/ecoreasoner/runs/f0-span-v3-contrastive \
      --max_steps 3000 --lr 1e-4 --alpha 0.2
"""
import argparse, contextlib, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Reutiliza clases del trainer v2 (no toca el .py original).
# train_mdlm_moe_v2 parsea sys.argv al importar; le pasamos un output dummy
# y luego restauramos argv para nuestro propio argparse.
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "scripts"))
_old_argv = sys.argv[:]
DUMMY_OUT = "/tmp/.contrastive_import_dummy"
Path(DUMMY_OUT).mkdir(parents=True, exist_ok=True)
sys.argv = ["train_mdlm_moe_v2.py", "--output", DUMMY_OUT, "--data_cache", "/tmp/nonexistent.npz"]
import train_mdlm_moe_v2 as base
sys.argv = _old_argv


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint model.pt a cargar")
    p.add_argument("--config", required=True, help="YAML del run base (model hparams)")
    p.add_argument("--pairs", required=True, help="jsonl con pares L0-L3 tokenizados")
    p.add_argument("--data_cache", required=True, help="cache .npz para MLM auxiliar")
    p.add_argument("--output", required=True, help="directorio de salida")
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lr_decay", default="cosine")
    p.add_argument("--lr_min_ratio", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.2,
                   help="peso de la perdida MLM auxiliar")
    p.add_argument("--margin", type=float, default=0.0,
                   help="margen para ranking (0 = softplus puro)")
    p.add_argument("--mask_schedule", default="uniform",
                   help="uniform | fixed | cosine para MLM auxiliar")
    p.add_argument("--mask_schedule_args", default='{"b_l":0.05,"b_h":0.95}')
    p.add_argument("--mask_p", type=float, default=0.15)
    p.add_argument("--span_len", type=int, default=64)
    p.add_argument("--whole_stage", action="store_true",
                   help="si el corpus es de esqueletos, activar masking por etapa")
    p.add_argument("--stage_labels", default="[OBSERVACION],[HIPOTESIS],[PREDICCION],[EVIDENCIA],[CONCLUSION]")
    p.add_argument("--seed", type=int, default=7331)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def log(msg, outdir):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        (Path(outdir) / "train.log").open("a").write(line + "\n")
    except Exception:
        pass


def load_hparams(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["model"]


def build_model(mcfg):
    return base.MdLMMoE(
        mcfg["vocab"], mcfg["hidden"], mcfg["layers"], mcfg["heads"],
        mcfg["ff_mult"], mcfg["seq_len"], mcfg.get("n_experts", 1),
        mcfg.get("k", 1), use_rope=mcfg.get("use_rope", False),
        weight_tying=mcfg.get("weight_tying", False)
    )


def load_pairs(pairs_path, max_len=768, n_max=None):
    """Carga pares L0-L3. Si contienen l3_subtype se guarda."""
    rows = []
    with open(pairs_path) as f:
        for i, line in enumerate(f):
            if n_max and i >= n_max:
                break
            try:
                d = json.loads(line)
                rows.append(d)
            except json.JSONDecodeError:
                continue
    return rows


def train_eval_split(pairs, eval_frac=0.2, seed=7331):
    """Split por CONTEXTO (no por fila): los pares que comparten ctx van todos
    al mismo lado, garantizando que eval nunca vea un contexto de train.

    Los ctx de pairs_hard_v3 son unicos (verificado 0 duplicados en disco),
    pero el split por ctx es la unica forma a prueba de futuros generadores
    que reusen contextos.
    """
    by_ctx = {}
    for i, p in enumerate(pairs):
        by_ctx.setdefault(tuple(p["ctx"]), []).append(i)
    ctx_list = list(by_ctx.keys())
    rng = random.Random(seed)
    rng.shuffle(ctx_list)
    n_eval_ctx = max(1, int(len(ctx_list) * eval_frac))
    eval_ctx = set(ctx_list[:n_eval_ctx])
    train_idx = [i for ctx, idxs in by_ctx.items() if ctx not in eval_ctx for i in idxs]
    eval_idx = [i for ctx, idxs in by_ctx.items() if ctx in eval_ctx for i in idxs]
    return [pairs[i] for i in train_idx], [pairs[i] for i in eval_idx]


def collate_ranking(batch, mask_id, seq_len, pad_id=0, device="cuda"):
    """Devuelve seqs, targets, pos_mask, lens para un batch de pares.

    Cada item del batch: {'ctx': ids, 'ok': ids, 'bad': ids}.
    Construimos secuencias ctx+ok y ctx+bad, con los tokens del sufijo ok/bad
    enmascarados con mask_id. El padding se hace con pad_id.
    """
    B = len(batch)
    seqlens = []
    for d in batch:
        for key in ("ok", "bad"):
            t = d["ctx"] + d[key]
            seqlens.append(len(t))
    T = min(max(seqlens), seq_len)

    seqs = torch.full((B * 2, T), pad_id, dtype=torch.long, device=device)
    targets = torch.full((B * 2, T), pad_id, dtype=torch.long, device=device)
    pos_mask = torch.zeros(B * 2, T, dtype=torch.bool, device=device)
    real_len = torch.zeros(B * 2, dtype=torch.long, device=device)

    for b, d in enumerate(batch):
        for k, key in enumerate(("ok", "bad")):
            idx = 2 * b + k
            ctx = d["ctx"][-seq_len:]  # truncar contexto si es muy largo
            tail = d[key][:seq_len - len(ctx)]
            full = ctx + tail
            L = len(full)
            if L > T:
                full = full[:T]
                L = T
            # Sufijo a enmascarar (todo lo que viene despues del contexto)
            ctx_len = min(len(ctx), L)
            seqs[idx, :L] = torch.tensor(full, device=device)
            # Dejar contexto visible, enmascarar sufijo
            for pos in range(ctx_len, L):
                seqs[idx, pos] = mask_id
            targets[idx, :L] = torch.tensor(full, device=device)
            pos_mask[idx, ctx_len:L] = True
            real_len[idx] = L
    return seqs, targets, pos_mask, real_len


def build_mlm_batches(data_cache, batch_size, seq_len, tok, rng, device="cuda"):
    """Carga data_cache .npz y devuelve lista de tensores [B, T]."""
    npz = np.load(data_cache)
    arr = npz["ids"]
    lengths = npz["lengths"]
    eos_id = int(npz.get("eos_id", tok.eos_token_id or tok.pad_token_id or 0))
    VB = tok.vocab_size
    if int(arr.max()) >= VB:
        arr = np.where(arr >= VB, 0, arr)
    all_ids = arr.astype(np.int64)
    if eos_id >= VB:
        eos_id = 0
    batches = base._pack_with_eos(all_ids, lengths, eos_id, batch_size, seq_len)
    # mover a device y mezclar
    batches = [b.to(device) for b in batches]
    rng.shuffle(batches)
    return batches


def sample_mask_fraction(schedule, b_l, b_h, rng=None):
    """Igual a train_mdlm_moe_v2.sample_mask_fraction."""
    if schedule == "fixed":
        return b_h
    if schedule == "uniform":
        return rng.random() * (b_h - b_l) + b_l
    if schedule == "cosine":
        t = rng.random()
        return b_l + (b_h - b_l) * 0.5 * (1 + math.cos((1 - t) * math.pi))
    return 0.5


def build_mlm_mask(xb, mask_p, mask_id, span_len, rng, whole_stage=False,
                   stage_label_ids=None):
    """Enmascara tokens para MLM auxiliar. Simplificado respecto a v2."""
    B, T = xb.shape
    n = max(1, int(T * mask_p))
    if whole_stage and stage_label_ids:
        return base.build_mask_indices(
            xb, n, "span", whole_stage=True,
            stage_label_ids=stage_label_ids, span_len=span_len,
            device=xb.device, token_scores=None)
    # span simple
    return base.build_mask_indices(
        xb, n, "span", whole_stage=False, span_len=span_len,
        device=xb.device, token_scores=None)


def _set_lr(step, opt, lr, warmup, max_steps, lr_decay, lr_min_ratio):
    """Cosine/WSD learning rate schedule."""
    if step < warmup:
        scale = (step + 1) / warmup
    else:
        if lr_decay == "cosine":
            t = (step - warmup) / max(1, max_steps - warmup)
            scale = lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(t * math.pi))
        elif lr_decay == "wsd":
            decay_start = int(max_steps * 0.8)
            if step < decay_start:
                scale = 1.0
            else:
                t = (step - decay_start) / max(1, max_steps - decay_start)
                scale = (1 - t) * (1 - lr_min_ratio) + lr_min_ratio
        else:
            scale = 1.0
    for g in opt.param_groups:
        g["lr"] = lr * scale


def ranking_loss(logits, targets, pos_mask, margin=0.0):
    """Calcula CE por par, luego ranking loss.

    logits/targets: [B, T, V] / [B, T]; pos_mask [B, T] indica qué tokens
    forman parte del sufijo a puntuar. Se asume que los pares estan
    intercalados: par i -> filas 2i (ok) y 2i+1 (bad).
    """
    V = logits.size(-1)
    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    flat_mask = pos_mask.reshape(-1)

    # Cross-entropy por token, reducir a media por fila (par)
    B = logits.size(0)
    per_token_ce = F.cross_entropy(flat_logits, flat_targets, reduction='none')
    per_token_ce = per_token_ce * flat_mask.float()

    row_loss = per_token_ce.view(B, -1).sum(dim=1)
    row_n = pos_mask.view(B, -1).sum(dim=1).float().clamp(min=1)
    row_ce = row_loss / row_n

    # score = -CE * n  (log-likelihood aproximada)
    row_score = -row_ce * row_n

    # pares ok=2i, bad=2i+1
    n_pairs = B // 2
    ok_score = row_score[0::2]
    bad_score = row_score[1::2]
    diff = bad_score - ok_score + margin  # queremos bad_score << ok_score
    loss = F.softplus(diff).mean()

    with torch.no_grad():
        acc = (ok_score > bad_score).float().mean().item()
    return loss, acc, row_ce.mean().item()


def main():
    a = parse()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    mcfg = load_hparams(a.config)
    mask_id = mcfg["vocab"]
    pad_id = 0

    log("Cargando modelo...", out)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = build_model(mcfg).to(device)
    state = torch.load(a.ckpt, map_location=device)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    log(f"Modelo cargado: {model.n_params() / 1e6:.1f}M params", out)

    log("Cargando pares...", out)
    pairs = load_pairs(a.pairs)
    log(f"Pares: {len(pairs)}", out)
    if not pairs:
        sys.exit("[fatal] sin pares")

    # Split por CONTEXTO (AUDITORIA v3.3): eval NUNCA ve ctx de train.
    train_pairs, eval_pairs = train_eval_split(pairs, eval_frac=0.2, seed=a.seed)
    log(f"Split por contexto: train={len(train_pairs)} eval={len(eval_pairs)}", out)

    log("Cargando cache MLM...", out)
    tok = base._load_tokenizer()  # reutiliza tokenizer de v2
    # mask_id = slot MASK real del tokenizer; validar contra YAML.
    if hasattr(tok, "vocab_size") and tok.vocab_size != mcfg["vocab"]:
        log(f"[warn] tok.vocab_size={tok.vocab_size} != yaml vocab={mcfg['vocab']}; "
            f"usando tok.vocab_size (Embedding es vocab+1)", out)
    mask_id = tok.vocab_size if hasattr(tok, "vocab_size") else mcfg["vocab"]
    mlm_batches = build_mlm_batches(a.data_cache, a.batch_size, mcfg["seq_len"], tok, random, device)
    log(f"Batches MLM: {len(mlm_batches)}", out)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)

    if a.whole_stage:
        stage_labels = [x.strip() for x in a.stage_labels.split(",") if x.strip()]
        stage_label_ids = [tok.encode(lbl, add_special_tokens=False)[0] for lbl in stage_labels]
        log(f"Stage label ids: {stage_label_ids}", out)
    else:
        stage_label_ids = None

    mask_sched = json.loads(a.mask_schedule_args) if isinstance(a.mask_schedule_args, str) else a.mask_schedule_args
    b_l = float(mask_sched.get("b_l", 0.05))
    b_h = float(mask_sched.get("b_h", 0.95))

    log("Iniciando fine-tuning contrastivo...", out)
    model.train()
    rng = random.Random(a.seed)
    step = 0
    it_mlm = 0
    last = time.time()
    best_acc = 0.0
    mem_hits = 0  # contador de chequeos consecutivos de memorizacion (AUDITORIA)
    loss = torch.zeros(())
    racc = 0.0
    while step < a.max_steps:
        # Ranking batch
        batch = [rng.choice(train_pairs) for _ in range(a.batch_size)]
        seqs, targets, pos_mask, _ = collate_ranking(batch, mask_id, mcfg["seq_len"], pad_id, device)
        logits = model(seqs)
        rloss, racc, rce = ranking_loss(logits, targets, pos_mask, a.margin)

        # MLM auxiliar
        xb = mlm_batches[it_mlm % len(mlm_batches)]
        it_mlm += 1
        frac = sample_mask_fraction(a.mask_schedule, b_l, b_h, rng)
        n = max(1, int(xb.size(1) * frac))
        mp = base.build_mask_indices(xb, n, "span", whole_stage=False, span_len=a.span_len,
                                     device=xb.device, token_scores=None)
        xm = xb.clone()
        xm.view(-1)[mp] = mask_id
        mlogits = model(xm)
        mloss = F.cross_entropy(mlogits.reshape(-1, mcfg["vocab"])[mp],
                                xb.reshape(-1)[mp])

        loss = rloss + a.alpha * mloss

        _set_lr(step, opt, a.lr, a.warmup, a.max_steps, a.lr_decay, a.lr_min_ratio)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()

        # Monitoreo
        if step % a.log_every == 0:
            log(f"step {step} loss={loss.item():.4f} rank={rloss.item():.4f} "
                f"rank_acc={racc:.3f} rank_ce={rce:.4f} mlm={mloss.item():.4f} "
                f"lr={opt.param_groups[0]['lr']:.2e}", out)

        if step % a.save_every == 0 and step > 0:
            save_dir = out / f"checkpoint-g{step}"
            save_dir.mkdir(exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step}, save_dir / "model.pt")
            log(f"checkpoint guardado: {save_dir}", out)

        # Evaluacion rapida con pares holdout (AUDITORIA v3.3: eval real)
        if step % (a.save_every // 2) == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                seqs_e, targets_e, pos_mask_e, _ = collate_ranking(eval_pairs[:16], mask_id, mcfg["seq_len"], pad_id, device)
                log_e, eval_acc, eval_ce = ranking_loss(model(seqs_e), targets_e, pos_mask_e, a.margin)
                # guard de memorizacion: train cerca de 1.0 y eval sin senal
                if racc > 0.95 and eval_acc < 0.60:
                    log(f"[warn] POSIBLE MEMORIZACION: train_acc={racc:.3f} eval_acc={eval_acc:.3f} "
                        f"(holdout, ctx NO vistos)", out)
                    mem_hits += 1
                else:
                    mem_hits = 0
                if mem_hits >= 4:
                    log(f"[fatal] memorizacion confirmada (4 chequeos seguidos): archivar run "
                        f"como INVALIDO, NO interpretar acc de battery", out)
                    (out / "MEMORIZATION.flag").write_text(
                        f"train_acc={racc:.4f} eval_acc={eval_acc:.4f} step={step}\n")
                    # no paramos: dejamos terminar para estudiar, pero el flag ya
                    # invalida el run a ojos de cualquier watchdog/lector.
            model.train()
            log(f"  eval rank_loss={log_e.item():.4f} eval_acc={eval_acc:.3f} (holdout)", out)

        step += 1

    # Guardar final
    final_dir = out / "checkpoint-g-final"
    final_dir.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "step": step}, final_dir / "model.pt")

    # Eval final completa sobre TODO el holdout (AUDITORIA v3.3)
    model.eval()
    with torch.no_grad():
        eval_acc_all = 0.0
        eval_loss_all = 0.0
        for i in range(0, len(eval_pairs), a.batch_size):
            chunk = eval_pairs[i:i + a.batch_size]
            seqs_e, targets_e, pos_mask_e, _ = collate_ranking(
                chunk, mask_id, mcfg["seq_len"], pad_id, device)
            l, acc, _ = ranking_loss(model(seqs_e), targets_e, pos_mask_e, a.margin)
            eval_acc_all += acc * len(chunk)
            eval_loss_all += l.item() * len(chunk)
        eval_acc_all = eval_acc_all / len(eval_pairs)
        eval_loss_all = eval_loss_all / len(eval_pairs)
    log(f"EVAL FINAL (holdout {len(eval_pairs)} pares, ctx NO vistos): "
        f"acc={eval_acc_all:.4f} loss={eval_loss_all:.4f}", out)

    summary = {
        "steps": step,
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "last_train_loss": loss.item(),
        "last_train_acc": racc,
        "eval_acc_holdout": round(eval_acc_all, 4),
        "eval_loss_holdout": round(eval_loss_all, 4),
        "memorization": bool(mem_hits >= 4),
        "verdict": "MEMORIZATION" if (racc > 0.95 and eval_acc_all < 0.60)
                   else "GO_CANDIDATE" if eval_acc_all >= 0.55 else "NO_SIGNAL",
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"VEREDICTO: {summary['verdict']} (train_acc={racc:.3f} eval_acc={eval_acc_all:.3f})", out)
    # flag
    (out / "training_complete.flag").write_text(f"COMPLETE steps={step} verdict={summary['verdict']}\n")
    log(f"COMPLETO: {step} steps -> {final_dir}", out)


if __name__ == "__main__":
    main()
