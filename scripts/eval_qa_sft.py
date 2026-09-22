"""eval_qa_sft.py — eval generativa de un ckpt SFT sobre eval_devin_hard.

Corre el modelo (denoise por confianza, estilo LLaDA) sobre cada pregunta
del eval held-out y puntúa la respuesta con las MISMAS heurísticas de
groundedness que qa_filter.py usó para curar el SFT:

  - gold-recall: fracción de content-words del gold `a` presentes en la
    generación.
  - num_ok: todo número del gold aparece (normalizado) en la generación.

Prompt = formato exacto del SFT: "[USER] <q>\\n[ASSISTANT]" (qa_filter.py).
--with-passage antepone el pasaje (variante open-book; NO es el formato
entrenado — solo para diagnóstico).

Salida: JSONL {pid,type,q,gold,gen,gold_recall,num_ok,n_gen} + resumen
por type a stdout y --summary.

Resume por pid: si --out ya existe, salta los pids ya escritos (olas).

Uso:
  python3 eval_qa_sft.py --ckpt runs/sft-v1 --ema \
      --tokenizer $TOK --eval data/eval_devin_hard.jsonl \
      --out runs/sft-v1/eval_devin_hard.gen.jsonl \
      [--steps 24 --max-new 128 --temp 0.7 --limit 0 --with-passage]
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_proto_v2 import resolve_ckpt, load_model, generate_conf, generate  # noqa: E402
from qa_filter import content_words, norm_num, _NUM  # noqa: E402

USER_M = "[USER]"
ASST_M = "[ASSISTANT]"


def score(gen, gold):
    gw = content_words(gold)
    gset = set(content_words(gen))
    rec = sum(1 for w in gw if w in gset) / len(gw) if gw else 0.0
    gnums = {norm_num(n) for n in _NUM.findall(gen)}
    gold_nums = [norm_num(n) for n in _NUM.findall(gold)]
    num_ok = all(n in gnums for n in gold_nums) if gold_nums else None
    return rec, num_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode", choices=["remask", "conf"], default="conf")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--with-passage", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    dev = torch.device(a.device)
    ckpt = resolve_ckpt(a.ckpt, ema=a.ema)
    print(f"[evalqa] ckpt={ckpt} dev={dev}", flush=True)
    model, arch = load_model(ckpt, dev)
    mask_id = arch["vocab"]
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    gen = generate_conf if a.decode == "conf" else generate

    done = set()
    if Path(a.out).exists():
        for line in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["pid"])
            except (json.JSONDecodeError, KeyError):
                pass

    recs = [json.loads(l) for l in open(a.eval, encoding="utf-8")]
    if a.limit:
        recs = recs[:a.limit]
    todo = [r for r in recs if str(r.get("pid")) not in done]
    print(f"[evalqa] eval={len(recs)} done={len(done)} todo={len(todo)} "
          f"steps={a.steps} max_new={a.max_new} temp={a.temp} decode={a.decode}",
          flush=True)

    t0 = time.time()
    fout = open(a.out, "a", encoding="utf-8")
    for i, r in enumerate(todo):
        q = r["q"].strip()
        prompt = f"{USER_M} {q}\n{ASST_M}"
        if a.with_passage and r.get("passage"):
            prompt = f"{r['passage'].strip()}\n{prompt}"
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = ids[-(arch["seq_len"] - a.max_new):]
        out = gen(model, ids, a.max_new, a.steps, a.temp, rng, mask_id)
        new_ids = [t for t in out[len(ids):] if t != mask_id]
        text = tok.decode(new_ids).strip()
        rec, num_ok = score(text, r["a"])
        fout.write(json.dumps({
            "pid": r.get("pid"), "type": r.get("type"), "q": q,
            "gold": r["a"], "gen": text, "gold_recall": round(rec, 4),
            "num_ok": num_ok, "n_gen": len(new_ids),
        }, ensure_ascii=False) + "\n")
        fout.flush()
        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"[evalqa] {i + 1}/{len(todo)} "
                  f"({dt / (i + 1):.2f}s/q, eta {(len(todo) - i - 1) * dt / (i + 1) / 60:.0f}min)",
                  flush=True)
    fout.close()

    agg = defaultdict(lambda: [0, 0.0, 0, 0])
    for line in open(a.out, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = r.get("type", "?")
        agg[t][0] += 1
        agg[t][1] += r["gold_recall"]
        if r.get("num_ok") is not None:
            agg[t][2] += 1
            agg[t][3] += int(r["num_ok"])
    print("\n=== RESUMEN por type ===")
    print(f"{'type':<14} {'n':>5} {'gold_recall':>11} {'num_ok':>12}")
    tot = [0, 0.0, 0, 0]
    for t in sorted(agg):
        n, rs, nn, nk = agg[t]
        for j in range(4):
            tot[j] += agg[t][j]
        print(f"{t:<14} {n:>5} {rs / n:>11.3f} "
              f"{(f'{nk}/{nn} = {nk / nn:.3f}') if nn else '-':>12}")
    if tot[0]:
        print(f"{'TOTAL':<14} {tot[0]:>5} {tot[1] / tot[0]:>11.3f} "
              f"{(f'{tot[3]}/{tot[2]} = {tot[3] / tot[2]:.3f}') if tot[2] else '-':>12}")
    if a.summary:
        json.dump({t: {"n": v[0], "gold_recall": v[1] / v[0],
                       "num_n": v[2], "num_ok": v[3]}
                   for t, v in agg.items()},
                  open(a.summary, "w"), indent=1)


if __name__ == "__main__":
    main()
