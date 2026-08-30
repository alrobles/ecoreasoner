#!/usr/bin/env python3
"""
build_l1_synth.py — Sintetiza el dataset de L1 (tool-use) desde trazas B1.

Objetivo: que el dLLM (masked-diffusion) aprenda a GENERAR tool calls validas
de forma autonoma + REPARAR tras un error, a partir de las trazas
[prompt -> trajectory(tool_calls) -> final] de la destilacion B1.

Por cada tool call REAL de una traza se generan:
  A. GENERACION  : [INSTRUCCION]+[CONTEXTO]  -> [ACCION] <tool call correcta>
  B. ERROR+REPAIR: [INSTRUCCION]+[CONTEXTO]  -> [ERROR] <detalle sintetico>
                                                              -> [ACCION] <repair correcta>
  C. RESPUESTA   : [INSTRUCCION]+[CONTEXTO]  -> [RESPUESTA] <texto final>

Mutaciones sinteticas de error (deterministas, sobre el JSON de la tool call):
  M1 JSON truncado | M2 comilla/llave rota | M3 typo en valor string |
  M4 typo en nombre de funcion | M5 tool_call_id roto

El serializado es TEXTO PLANO (el dLLM es masked-diffusion sobre tokens):
cada doc es un fragmento autocontenido <= 768 tok aprox (contexto truncado a
a.max_ctx_chars), con [ACCION]/[RESPUESTA] SIEMPRE al final -> el modelo
aprende a completar el JSON/respuesta al denoisar el final.

Salida: data/l1/train_corpus_l1.jsonl  {text, pmid, domain:"tooluse",
source:"l1-synth", arxiv_id, title, year}
"""
from __future__ import annotations
import argparse, glob, json, os, random, re, sys, time
from collections import Counter

if not os.environ.get("SLURM_JOB_ID") and not os.environ.get("L1_ALLOW_LOCAL"):
    sys.exit("ERROR: build_l1_synth.py SOLO via Slurm (o export L1_ALLOW_LOCAL=1 para prueba local).")

CTX_CUT = 1400  # chars max de contexto por doc (~600 tok)

def trunc(s, n=CTX_CUT):
    s = re.sub(r"\s+", " ", s or "").strip()
    if len(s) <= n:
        return s
    cut = s[:n]
    return cut[: cut.rfind(" ")] if " " in cut else cut

def serial_toolcall(tc):
    """tool call a texto plano canonico. Si fn == 'ecocode' (formato B), el
    'arguments' es CODIGO plano, no JSON -> serializar directo."""
    fn = tc.get("function", {})
    name = fn.get("name", "unknown")
    args = fn.get("arguments", "{}")
    if name == "ecocode":
        # codigo plano: recortar a CTX_CUT para no exceder el contexto
        return trunc(args, CTX_CUT)
    if isinstance(args, dict):
        args = json.dumps(args, ensure_ascii=False)
    return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)

def mut_json_trunc(orig: str) -> str:
    # quitar el ultimo bloque {..} o llave de cierre
    i = orig.rfind("}")
    if i > 0:
        return orig[:i] + orig[i+1:].replace("}", " ") if i+1 < len(orig) else orig[:i]
    return orig[:-1]

def mut_quote_broken(orig: str) -> str:
    # duplicar la primera comilla de un valor string
    m = re.search(r'"[^"]{4,}"', orig)
    if m:
        return orig[:m.start()+1] + m.group(0) + orig[m.end():]
    return orig

def mut_value_typo(orig: str) -> str:
    # alterar 1 caracter interior del PRIMER valor string (ej onca->onco)
    m = re.search(r'"[^"]{5,}"', orig)
    if not m:
        return orig
    val = m.group(0)
    if len(val) <= 4:
        return orig
    mid = len(val) // 2
    ch = val[mid]
    repl = "o" if ch.lower() in "aeiou" else "o"
    newval = val[:mid] + repl + val[mid+1:]
    return orig[:m.start()] + newval + orig[m.end():]

def mut_fn_typo(orig: str) -> str:
    # cambiar una letra del nombre de funcion (gbif_occurrence -> gbif_occurence)
    m = re.search(r'"name":\s*"([a-z_]+)"', orig)
    if not m:
        return orig
    name = m.group(1)
    if len(name) < 4:
        return orig
    i = len(name)//2
    repl = "o" if name[i] not in "o" else "i"
    newname = name[:i] + repl + name[i+1:]
    return orig[:m.start(1)] + newname + orig[m.end(1):]

def mut_id_broken(tool_call_json: str, tc: dict) -> str:
    # simular id roto: marcar el id en el texto? el id no esta en el JSON;
    # lo anadimos como [TOOL_CALL] <json> con nota.
    return tool_call_json

ERROR_TEMPLATES = {
    "M1": "Error: JSON de tool call malformado: se esperaba '}}' al final. Repara la tool call.",
    "M2": "Error: JSON de tool call malformado: comillas desbalanceadas. Repara la tool call.",
    "M3": "Error: argumento invalido: el valor no corresponde a una opcion valida para este parametro. Repara la tool call.",
    "M4": "Error: funcion desconocida '{fn}': no existe esa herramienta en el esquema. Repara la tool call.",
    "M5": "Error: tool_call_id invalido o ausente: la llamada no se puede asociar a una ejecucion. Repara la tool call.",
}

MUTATIONS = [
    ("M1", mut_json_trunc, "JSON truncado"),
    ("M2", mut_quote_broken, "comilla rota"),
    ("M3", mut_value_typo, "typo en valor"),
    ("M4", mut_fn_typo, "typo en funcion"),
    ("M5", mut_id_broken, "id roto"),
]

def parse_trace(d):
    """Devuelve (prompt, steps[], final) donde steps = [(assistant_text, tool_calls)].
    Soportar 2 formatos:
      A) trajectory[][] con tool_calls JSON (distill_data / distill_v4_round2)
      B) context/reasoning/code (sci_v2_b1, destilacion masiva code_valid)
    """
    prompt = d.get("prompt", "") or ""
    traj = d.get("trajectory")
    if traj:
        steps = []
        cur_text = ""
        cur_calls = []
        for msg in traj:
            role = msg.get("role")
            if role == "user" and not prompt:
                prompt = msg.get("content", "")
            elif role == "assistant":
                cur_text = (msg.get("content") or "").strip()
                cur_calls = msg.get("tool_calls") or []
                if cur_calls:
                    steps.append((cur_text, cur_calls))
        final = d.get("final", "") or ""
        return prompt, steps, final
    # formato B: context->reasoning->code
    ctx = d.get("context", "") or ""
    reason = d.get("reasoning", "") or ""
    code = d.get("code", "") or ""
    if code and ctx:
        code_txt = code.strip()
        code_txt = re.sub(r"^```\w*\n|\n```$", "", code_txt).strip()
        # paso unico: la "accion" es el bloque de codigo python (no JSON)
        synthetic_call = {"function": {"name": "ecocode", "arguments": code_txt}}
        return ctx, [(reason, [synthetic_call])], ""
    return prompt, [], ""

def build_docs(prompt, steps, final, trace_id, rng):
    docs = []
    # contexto acumulado (texto de pasos previos) truncado
    ctx = ""
    for si, (text, calls) in enumerate(steps):
        for tc in calls:
            fn = tc.get("function", {}).get("name", "unknown")
            tc_json = serial_toolcall(tc)
            # A. generacion
            doc = {"text": f"[INSTRUCCION] {trunc(prompt)}\n[CONTEXTO] {trunc(ctx)}\n"
                          f"[ACCION] {tc_json}",
                   "kind": "gen", "fn": fn}
            docs.append(doc)
            # B. error + repair (cada mutation -> 1 doc)
            for code, mutfn, _label in MUTATIONS:
                broken = mutfn(tc_json) if code != "M5" else tc_json
                err = ERROR_TEMPLATES[code].format(fn=fn)
                if code == "M5":
                    err = ERROR_TEMPLATES["M5"]
                doc = {"text": f"[INSTRUCCION] {trunc(prompt)}\n[CONTEXTO] {trunc(ctx)}\n"
                              f"[ERROR] {err}\n[ACCION] {tc_json}",
                       "kind": f"repair-{code}", "fn": fn}
                docs.append(doc)
        # avanzar contexto con el texto del paso + un resumen de la accion
        if text:
            ctx = ctx + " " + trunc(text, 400)
    if final:
        doc = {"text": f"[INSTRUCCION] {trunc(prompt)}\n[CONTEXTO] {trunc(ctx)}\n"
                       f"[RESPUESTA] {trunc(final, 1400)}",
               "kind": "final", "fn": None}
        docs.append(doc)
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", default=[
        "/beegfs/a474r867/ecoreasoner/data/distill_data.jsonl",
        "/beegfs/a474r867/ecoreasoner/data/distill_v4_round2.jsonl",
    ])
    ap.add_argument("--glob", default="/beegfs/a474r867/ecoreasoner/data/sci_v2_b1*.jsonl",
                    help="glob adicional de trazas B1")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/l1/train_corpus_l1.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t0 = time.time()
    files = list(a.input)
    for g in a.glob.split():
        files += sorted(glob.glob(g))
    files = [f for f in dict.fromkeys(files) if os.path.exists(f) and os.path.getsize(f) > 0]
    print(f"[l1-synth] fuentes: {len(files)}", flush=True)
    for f in files:
        print(f"  - {f} ({os.path.getsize(f)//1024}KB)", flush=True)

    rng = random.Random(a.seed)
    n_trace = 0
    docs = []
    for fp in files:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                try: d = json.loads(ln)
                except Exception: continue
                prompt, steps, final = parse_trace(d)
                if not prompt or not steps:
                    continue
                n_trace += 1
                docs.extend(build_docs(prompt, steps, final, d.get("id",""), rng))
    if not docs:
        sys.exit("ERROR: no se genero ningun doc (revisa el formato de las trazas)")

    random.shuffle(docs)
    kinds = Counter(d["kind"] for d in docs)
    fns = Counter(d["fn"] for d in docs if d["fn"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for d in docs:
            rec = {
                "text": d["text"],
                "pmid": f"l1-{n_trace}-{d['kind']}",
                "domain": "tooluse",
                "source": "l1-synth",
                "arxiv_id": None, "title": "", "year": None,
                "kind": d["kind"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    toks = sum(len(d["text"])//4 for d in docs)
    print(f"[l1-synth] DONE {len(docs)} docs (de {n_trace} trazas), ~{toks/1e6:.1f}M tok -> {a.out} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    print("kinds:", dict(kinds), flush=True)
    print("funciones:", dict(fns), flush=True)

if __name__ == "__main__":
    main()