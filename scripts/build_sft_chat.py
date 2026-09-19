#!/usr/bin/env python3
"""build_sft_chat.py — construye el corpus chat-SFT v1 (EN) para el dLLM-MoE.

Fuentes públicas (parquet ya descargados):
  - allenai/sciq            : question + correct_answer + support (ciencia)
  - HuggingFaceTB/smol-smoltalk : messages[] user/assistant (chat general,
                                diseñado para SFT de modelos pequeños)

Salida: JSONL con {"prompt": str, "response": str, "src": str}
El trainer SFT enmascara SOLO la response (receta LLaDA 2.3). El formato
de superficie se fija aquí (markers estilo esqueleto):
    [USER] <q>\n[ASSISTANT] <a>
→ en el npz, prompt = "[USER] <q>\n[ASSISTANT]", response = " <a>".

Filtros: respuesta 8..512 chars (sciq corta ok), prompt <= ~400 tok aprox
(chars/4), inglés asumido (ambas fuentes son EN).

Uso:
  python3 build_sft_chat.py --sciq sciq.parquet --smoltalk smol.parquet \
      --out chat_sft_v1.jsonl [--n-smol 25000] [--seed 0]
"""
import argparse, json, random, re
import pyarrow.parquet as pq

USER_M, ASST_M = "[USER]", "[ASSISTANT]"


def norm_ws(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def from_sciq(path, out):
    t = pq.read_table(path).to_pylist()
    n = 0
    for r in t:
        q, a, sup = norm_ws(r["question"]), norm_ws(r["correct_answer"]), norm_ws(r["support"])
        if not q or not a:
            continue
        # respuesta = answer + primera(s) oración(es) del soporte (≤~600 chars)
        expl = sup[:600].rsplit(". ", 1)[0]
        resp = a if not expl or a.lower() in expl.lower()[:120] else f"{a}. {expl}"
        out.append({"prompt": f"{USER_M} {q}\n{ASST_M}", "response": f" {resp}", "src": "sciq"})
        n += 1
    return n


def from_smoltalk(path, out, n_max, rng):
    t = pq.read_table(path).to_pylist()
    rng.shuffle(t)
    n = 0
    for r in t:
        if n >= n_max:
            break
        msgs = r.get("messages") or []
        # primer intercambio user->assistant (single-turn para v1)
        pair = None
        for i, m in enumerate(msgs[:-1]):
            if m.get("role") == "user" and msgs[i+1].get("role") == "assistant":
                pair = (m["content"], msgs[i+1]["content"]); break
        if not pair:
            continue
        q, a = norm_ws(pair[0]), pair[1].strip()
        if not q or not (8 <= len(a) <= 2048) or len(q) > 1600:
            continue
        out.append({"prompt": f"{USER_M} {q}\n{ASST_M}", "response": f" {a}", "src": "smoltalk"})
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sciq", required=True)
    ap.add_argument("--smoltalk", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-smol", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    recs = []
    n1 = from_sciq(args.sciq, recs)
    n2 = from_smoltalk(args.smoltalk, recs, args.n_smol, rng)
    rng.shuffle(recs)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sft_chat] sciq={n1} smoltalk={n2} total={len(recs)} -> {args.out}")


if __name__ == "__main__":
    main()
