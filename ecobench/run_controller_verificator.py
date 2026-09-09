#!/usr/bin/env python3
"""run_controller_verificator.py — Opción D, HITO 2: Controller (deepseek local)
que genera tool-calls + Verificator (verify_toolcall) + retry con feedback.

Flujo por ítem (diseño validado en el postmortem F2):
  1. Controller recibe la tarea + definición de las 3 herramientas.
  2. Controller emite UNA tool-call (JSON) + razonamiento breve.
  3. Verificator extrae/valida/repara (M1-M5) contra los schemas empíricos.
  4. Si la tool-call es inválida tras repair -> feedback con el error -> regenerar
     (hasta 2 retries).
  5. La tool-call válida se RESUELVE (mock del sandbox: devuelve el JSON de
     argumentos como resultado — la ejecución real ecológica llega en HITO 3).

Crítico (lección F2): NO se evalúa al controller con la misma métrica que decide;
aquí el criterio es la tool-call RESUELTA y la tasa de JSON válidas tras
verificación — medible y a prueba de atajos (L0-L3 ya probaron que el modelo
chico no lo logra; el controller SÍ, porque es el teacher).

Uso:
  python3 run_controller_verificator.py --items ecobench_eval.json --limit 5
      [--backend ollama|openrouter] [--model deepseek-v4-flash:latest]
Salida: controller_verificator_results.json + print por ítem.
"""
from __future__ import annotations
import argparse, json, os, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "scripts"))
from verify_toolcall import verify, SCHEMAS  # el verificator del HITO 1

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:20006/v1/chat/completions")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

TOOLS_DEF = [
    {"name": "gbif_occurrence",
     "description": "Busca registros de presencia de una especie en GBIF dentro de una región geográfica.",
     "parameters": {"species": "string, nombre científico de la especie (ej. 'Panthera onca')",
                    "region": "string, región geográfica (ej. 'peninsula de yucatan')"}},
    {"name": "bioclim_download",
     "description": "Descarga capas bioclimáticas (CHELSA/ERA5) para una región y año.",
     "parameters": {"region": "string, región geográfica",
                    "year": "string, año (ej. '2020')"}},
    {"name": "maxent_train",
     "description": "Entrena un modelo de nicho (MaxEnt) para una especie con capas bioclimáticas.",
     "parameters": {"species": "string, especie", "layers": "string, capas (ej. 'bioclim_19')"}},
]

SYSTEM_PROMPT = f"""Eres un controller de agentes ecológicos. Dada una tarea, decide QUÉ herramienta
llamar y con QUÉ argumentos, emitiendo EXACTAMENTE UNA tool-call en JSON.
Herramientas disponibles: {json.dumps(TOOLS_DEF, ensure_ascii=False)}

Formato de respuesta (solo JSON, sin markdown, sin explicación extra):
{{"tool": "<nombre>", "arguments": {{"param1": "valor1", ...}}, "rationale": "<1 frase, por qué esta llamada>"}}

Reglas:
- Usa SOLO los nombres y parámetros exactos de la definición.
- Los argumentos deben ser strings.
- Si la tarea no es una llamada de herramienta clara, responde con "tool": null."""

def llm_call(prompt, model, backend):
    """Una llamada al LLM. Devuelve (texto_completo, error)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    if backend == "ollama":
        payload = json.dumps({"model": model, "messages": messages,
                              "temperature": 0.1, "max_tokens": 1500,
                              "reasoning_effort": "none"}).encode()
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
    else:
        key = ""
        for p in [os.path.expanduser("~/env/openrouter-key"), os.path.expanduser("~/env/hermes-ecoseek-key")]:
            if os.path.exists(p):
                key = open(p).read().strip(); break
        if not key:
            return None, "no openrouter key"
        payload = json.dumps({"model": model, "messages": messages,
                              "temperature": 0.1, "max_tokens": 1500}).encode()
        req = urllib.request.Request(OPENROUTER_URL, data=payload,
                                     headers={"Authorization": f"Bearer {key}",
                                              "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            res = json.loads(r.read())
        if "choices" not in res or not res["choices"]:
            return None, f"sin choices: {str(res)[:200]}"
        msg = res["choices"][0].get("message") or {}
        text = msg.get("content") or ""
        if not text.strip():  # reasoning-first (v4-flash)
            rz = msg.get("reasoning") or ""
            if not rz:
                rz = "\n".join(d.get("text", "") for d in (msg.get("reasoning_details") or []) if isinstance(d, dict))
            text = rz or ""
        return text, None
    except Exception as e:
        return None, str(e)

def parse_controller_output(text):
    """Extrae {tool, arguments} del output del controller (JSON o tool-call)."""
    import re
    # JSON directo
    for cand in (text.strip(), text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        if not cand:
            continue
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and d.get("tool"):
                return d, None
        except (json.JSONDecodeError, ValueError):
            continue
    # formato tool-call (parser del verificator)
    from verify_toolcall import extract_toolcalls
    tcs = extract_toolcalls(text)
    if tcs:
        tc = tcs[0]
        return {"tool": tc["name"], "arguments": tc["args"]}, None
    return None, "output sin tool-call JSON"

def resolve_tool(tool, args, iid):
    """Resolución de la herramienta. Con --real llama a tools_resolver (GBIF API
    real + stack HPC); sin él, mock simbólico (HITO 2). Devuelve (status, msg)."""
    if _USE_REAL:
        from tools_resolver import resolve
        return resolve(tool, args)
    if tool == "gbif_occurrence":
        return "ok", f"gbif: {args.get('species')} en {args.get('region')} -> 42 registros (mock)"
    if tool == "bioclim_download":
        return "ok", f"bioclim {args.get('region')} {args.get('year')} -> 19 capas (mock)"
    if tool == "maxent_train":
        return "ok", f"maxent {args.get('species')} layers={args.get('layers')} -> AUC 0.85 (mock)"
    return "fail", f"tool desconocida: {tool}"

_USE_REAL = False  # set por --real en main()

def build_controller_prompt(task):
    """Prompt del controller: tarea + instrucción estricta de formato JSON tool-call.
    (La versión previa pasaba la pregunta cruda y el teacher respondía código/texto,
    no una tool-call — el llm_call directo con instrucción JSON SÍ funciona)."""
    return (
        f"{task}\n\n"
        f"Emite EXACTAMENTE UNA tool-call en JSON con este formato:\n"
        f"{{\"tool\": \"<nombre>\", \"arguments\": {{\"param\": \"valor\", ...}}, \"rationale\": \"<1 frase>\"}}\n"
        f"Herramientas: gbif_occurrence(species,region) | bioclim_download(region,year) | "
        f"maxent_train(species,layers).\n"
        f"SOLO JSON, sin markdown ni explicación fuera."
    )

def run_one(iid, q, model, backend):
    """Devuelve dict con el resultado del ítem tras el bucle controller->verificator."""
    out = {"id": iid, "attempts": 0, "tool": None, "args": None,
           "verified": False, "repairs": [], "resolve": None, "error": None}
    prompt = build_controller_prompt(q)
    for attempt in range(1, 4):  # 1 + 2 retries con feedback
        out["attempts"] = attempt
        text, err = llm_call(prompt, model, backend)
        if err:
            out["error"] = f"llm: {err}"; break
        parsed, perr = parse_controller_output(text)
        if perr:
            # feedback: pedir regenerar (no contamina el conteo de JSON válidas)
            prompt = prompt + f"\n\nTu respuesta anterior no contenía una tool-call JSON válida. Devuelve solo JSON con la herramienta y argumentos correctos."
            out["error"] = perr
            continue
        # verificator: repara M1-M5
        ver = verify(text)
        if not ver["ok"]:
            prompt = prompt + f"\n\nTool-call inválida: {ver['error']}. Repara y reemite solo JSON."
            out["error"] = ver["error"]
            continue
        out["tool"] = ver["function"]
        out["args"] = ver["args"]
        out["verified"] = True
        out["repairs"] = ver["repairs"]
        status, msg = resolve_tool(ver["function"], ver["args"], iid)
        out["resolve"] = msg
        out.pop("error", None)
        break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(ROOT / "ecobench_eval.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="deepseek-v4-flash:latest")
    ap.add_argument("--backend", choices=["ollama", "openrouter"], default="ollama")
    ap.add_argument("--ids", default="")
    ap.add_argument("--real", action="store_true",
                    help="resolver REAL (GBIF API + stack HPC) en vez de mock")
    a = ap.parse_args()
    global _USE_REAL
    _USE_REAL = a.real
    doc = json.load(open(a.items))
    meta = doc["meta"]
    items = [i for i in doc["items"] if i["split"] == "eval_holdout" and not i["id"].startswith("eco-")]
    if a.ids:
        want = {x.strip() for x in a.ids.split(",") if x.strip()}
        items = [i for i in items if i["id"] in want]
    if a.limit:
        items = items[:a.limit]
    print(f"== Controller+Verificator ({a.backend}/{a.model}) == {len(items)} ítems")
    results = []
    for it in items:
        iid = it["id"]
        print(f"\n[{iid}] ", end="", flush=True)
        r = run_one(iid, it.get("question", ""), a.model, a.backend)
        results.append(r)
        if r["verified"]:
            print(f"OK tool={r['tool']} args={json.dumps(r['args'], ensure_ascii=False)[:80]} "
                  f"repairs={len(r['repairs'])} -> {r['resolve']}")
        else:
            print(f"FAIL {r.get('error')}")
    out = ROOT / "controller_verificator_results.json"
    json.dump({"meta": meta, "model": a.model, "backend": a.backend,
               "results": results}, open(out, "w"), indent=1, ensure_ascii=False)
    n = len(results)
    nv = sum(1 for r in results if r["verified"])
    print(f"\n== RESULTADO == tool-calls válidas tras verificator: {nv}/{n} = {100*nv/max(n,1):.0f}%")
    print(f"guardado: {out}")

if __name__ == "__main__":
    main()