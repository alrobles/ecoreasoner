#!/usr/bin/env python3
"""eval_controller_replication.py — HITO 3 (opción D): replicación sobre ground truth.

Dado el corpus de trazas teacher (prompt -> tool-call gold), reproduce el prompt
contra el controller (deepseek local) y mide cuántas tool-calls replica:

  match_fun : misma FUNCIÓN (exacta)
  match_arg : TODOS los argumentos gold presentes en la llamada reproducida,
              con similitud de tokens >= umbral (semántica, tolera variantes
              ortográficas/mayúsculas: "peninsula de yucatan" vs "Península de Yucatán")

Ratio honesta, sin atajos: la decisión es "llamó a la herramienta correcta para
la tarea", no coherencia superficial (lección L0-L3).

Uso:
  python3 scripts/eval_controller_replication.py --gold /tmp/gold_prompts.json \
      --backend ollama --model deepseek-v4-flash:latest [--limit N] [--out REPORT]
"""
import argparse, json, re, sys, time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ecobench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_controller_verificator import llm_call, parse_controller_output, build_controller_prompt
from verify_toolcall import verify

def _norm(s):
    s = (s or "").lower()
    s = re.sub(r"[áàâ]", "a", s); s = re.sub(r"[éèê]", "e", s)
    s = re.sub(r"[íìî]", "i", s); s = re.sub(r"[óòô]", "o", s)
    s = re.sub(r"[úùûü]", "u", s); s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _sim(a, b):
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()

def arg_match(gold_val, pred_val, thr=0.8):
    """Tolerante: igualdad, contiene, o similitud de tokens >= thr."""
    g, p = str(gold_val or "").strip(), str(pred_val or "").strip()
    if not g:
        return True
    if not p:
        return False
    if _norm(g) == _norm(p):
        return True
    if _norm(g) in _norm(p) or _norm(p) in _norm(g):
        return True
    return _sim(g, p) >= thr

def replicate_one(item, model, backend):
    """(match_fun, match_args, pred_fun, pred_args, gold_fun, gold_args, repairs, err)."""
    gold_fun = item["gold"][0]["tool"]
    gold_args = item["gold"][0]["args"]
    prompt = build_controller_prompt(item["prompt"])
    for attempt in range(1, 4):
        text, err = llm_call(prompt, model, backend)
        if err:
            return (False, False, None, None, gold_fun, gold_args, [], f"llm: {err}")
        parsed, perr = parse_controller_output(text)
        if perr:
            prompt += f"\n\nTu respuesta anterior no era JSON. Devuelve solo la tool-call JSON."
            continue
        ver = verify(text)
        if not ver["ok"]:
            prompt += f"\n\nTool-call inválida: {ver['error']}. Repara y reemite solo JSON."
            continue
        fun_ok = ver["function"] == gold_fun
        args_ok = fun_ok and all(arg_match(gold_args.get(k), ver["args"].get(k)) for k in gold_args)
        return (fun_ok, args_ok, ver["function"], ver["args"], gold_fun, gold_args,
                ver["repairs"], None)
    return (False, False, None, None, gold_fun, gold_args, [], "max retries")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="json con [{prompt, gold:[{tool,args}]}]")
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--model", default="deepseek-v4-flash:latest")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="replication_results.jsonl")
    ap.add_argument("--report", default="replication_report.json")
    a = ap.parse_args()

    items = json.load(open(a.gold))
    if a.limit:
        items = items[:a.limit]
    print(f"== REPLICACIÓN ({a.backend}/{a.model}) == {len(items)} prompts", flush=True)

    t0 = time.time()
    rows = []
    for i, it in enumerate(items, 1):
        r = replicate_one(it, a.model, a.backend)
        fun_ok, args_ok, pf, pa, gf, ga, reps, err = r
        row = {"i": i, "prompt": it["prompt"][:160],
               "gold_fun": gf, "pred_fun": pf, "gold_args": ga, "pred_args": pa,
               "match_fun": fun_ok, "match_args": args_ok, "repairs": reps, "error": err}
        rows.append(row)
        tag = "OK" if args_ok else ("FUN" if fun_ok else "MIS")
        print(f"  [{i}/{len(items)}] {tag} gold={gf} pred={pf} args_ok={args_ok}"
              f"{' repairs='+str(len(reps)) if reps else ''}{' err='+str(err)[:60] if err else ''}",
              flush=True)
        with open(a.out, "a" if i > 1 else "w") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(rows)
    nfun = sum(1 for r in rows if r["match_fun"])
    narg = sum(1 for r in rows if r["match_args"])
    el = time.time() - t0
    report = {"model": a.model, "backend": a.backend, "n": n,
              "n_match_func": nfun, "match_func_ratio": round(nfun/n, 4) if n else None,
              "n_match_args": narg, "match_args_ratio": round(narg/n, 4) if n else None,
              "elapsed_s": round(el), "sec_per_item": round(el/max(n,1), 1),
              "out": a.out}
    json.dump(report, open(a.report, "w"), indent=1)
    print(f"\n== RESULTADO == match_func {nfun}/{n} ({100*nfun/max(n,1):.0f}%) | "
          f"match_args {narg}/{n} ({100*narg/max(n,1):.0f}%) | {el:.0f}s")
    return 0 if narg == n else 1

if __name__ == "__main__":
    sys.exit(main())