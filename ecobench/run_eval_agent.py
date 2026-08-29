#!/usr/bin/env python3
"""
run_eval_agent.py — Driver del agente para EcoBench-EVAL (ejecución real).

Conecta el LLM (teacher v4-flash local o OpenRouter) al runner de EcoBench:
  1. Para cada ítem del split eval_holdout, pide al LLM generar código Python que
     resuelva la pregunta (dado el dataset de entrada real).
  2. Ejecuta el código en un sandbox con CWD = benchmark/ (los paths son relativos).
  3. Verifica contra el eval_program real de ScienceAgentBench (cuando es ejecución
     pura) o con verify_numeric/artifact (nuestro fallback).

CAVEATS (importantes, NO ignorar):
  - Algunos eval_programs de SAB usan `gpt4_visual_judge` (juez de imagen LLM) y
    `gold_results/` que puede no existir -> esos se marcan `skip_llm_judge` (no
    son verificacion-por-ejecucion pura; se documentan, no se fuerzan).
  - Los gold_programs/gold data de SAB contienen canary "NEVER IN TRAINING" ->
    NUNCA usar gold como training data; solo referencia de verificación.
  - Los paths de datos SAB son relativos a `ecobench_raw/sciagentbench/benchmark/`
    -> el sandbox corre con ese CWD.
  - No se evalúa el split 'train' (nunca; anti-contaminación).

Uso:
  python3 run_eval_agent.py [--items ecobench_eval.json] [--limit N]
                            [--model deepseek-v4-flash:latest] [--backend ollama|openrouter]
                            [--timeout-per-item 300]
"""
from __future__ import annotations
import argparse, json, os, sys, subprocess, tempfile, time, urllib.request, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_ITEMS = ROOT / "ecobench_eval.json"
# CWD del sandbox: los paths SAB son relativos a benchmark/
SAB_BENCH = ROOT / "ecobench_raw" / "sciagentbench" / "benchmark"
SAB_CSV = ROOT / "ecobench_raw" / "sciagentbench" / "ScienceAgentBench.csv"
CANARY = "NEVER IN TRAINING"  # los gold de SAB lo llevan -> no usar como train

def load_sab_tree():
    """Carga instance_id -> dataset_folder_tree del CSV SAB (para inyectar al prompt)."""
    import csv
    m = {}
    if SAB_CSV.exists():
        for r in csv.DictReader(open(SAB_CSV)):
            m[r.get("instance_id")] = r.get("dataset_folder_tree","")
    return m

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:20006/v1/chat/completions")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Python de ejecución (entorno con libs GIS): default venv_dl si existe
PYTHON_EXEC = os.environ.get("EVAL_PYTHON", "/beegfs/a474r867/venv_dl/bin/python")

SYSTEM_PROMPT = """You are an expert ecological/gis programmer. Write a single self-contained
Python program that solves the task below using the dataset files provided (they are in the
current working directory, under `datasets/<name>/`). The program must:
- import all needed libraries (geopandas, rasterio, numpy, pandas, matplotlib, seaborn, etc.)
- ALWAYS create the output directory `pred_results/` (os.makedirs("pred_results", exist_ok=True))
- read the data from the EXACT relative paths shown in the dataset folder tree, ALWAYS prefixed
  with `datasets/<DatasetName>/` (e.g. `gpd.read_file("datasets/ElkMovement/xxx.geojson")`).
  NEVER invent paths and NEVER read from a bare filename — always include the dataset directory.
- keep projections simple: use GeoDataFrame.to_crs(epsg=XXXX) with the numeric epsg code as an
  integer argument (e.g. `to_crs(epsg=3347)`), NEVER `to_crs(epsg:3347)`.
- geometry ops: prefer `gdf.geometry.union_all()` or `shapely.ops.unary_union` (both valid);
  if you import from shapely use `from shapely.ops import unary_union` (works in all versions)
- save the output artifact to the exact path requested (e.g. pred_results/xxx.png or .csv)
- be runnable with `python3 <file>` from the current working directory
Output ONLY the Python code between ```python and ``` markers. No explanation."""

# Fix reasoning-first: subir max_tokens para que el razonamiento no agote el
# budget de salida y deje content vacío (deepseek-v4-flash, validado 2026-08-28)
MAX_TOKENS = 8000

def llm_generate(prompt, model, backend, retries=3):
    """Genera código con el LLM (ollama local o openrouter). Retorna (código, error).
    Retries internos ante rate-limit/timeout de OpenRouter (backoff)."""
    import time as _t
    last_err = None
    for attempt in range(retries):
        try:
            if backend == "ollama":
                payload = json.dumps({"model": model, "messages": [
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":prompt}],
                    "temperature": 0.1, "max_tokens": MAX_TOKENS}).encode()
                req = urllib.request.Request(OLLAMA_URL, data=payload,
                                             headers={"Content-Type":"application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=600) as r:
                    res = json.loads(r.read())
                if "choices" not in res or not res["choices"]:
                    last_err = f"ollama sin choices: {str(res)[:200]}"
                    _t.sleep(5*(attempt+1)); continue
                text = res["choices"][0]["message"].get("content") or ""
            else:  # openrouter
                key = ""
                for p in [os.path.expanduser("~/env/openrouter-key"), os.path.expanduser("~/env/hermes-ecoseek-key")]:
                    if os.path.exists(p):
                        key = open(p).read().strip(); break
                if not key:
                    return None, "no openrouter key"
                payload = json.dumps({"model": model, "messages": [
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":prompt}],
                    "temperature": 0.1, "max_tokens": MAX_TOKENS}).encode()
                req = urllib.request.Request(OPENROUTER_URL, data=payload,
                                             headers={"Authorization":f"Bearer {key}",
                                                      "Content-Type":"application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=600) as r:
                    res = json.loads(r.read())
                if "choices" not in res or not res["choices"]:
                    # rate limit / error -> backoff y reintentar
                    last_err = f"openrouter sin choices: {str(res.get('error') or res)[:200]}"
                    _t.sleep(8*(attempt+1)); continue
                msg = res["choices"][0].get("message") or {}
                text = msg.get("content") or ""
                # Fix reasoning-first (2026-08-28): deepseek-v4-flash emite el
                # contenido en `reasoning`/`reasoning_details` y deja content vacío
                # cuando el budget se agota razonando -> 7/14 empty_code sin esto.
                if not text.strip():
                    rz = msg.get("reasoning") or ""
                    if not rz:
                        rdd = msg.get("reasoning_details") or []
                        rz = "\n".join(d.get("text", "") for d in rdd if isinstance(d, dict))
                    text = rz or ""
            break
        except Exception as e:
            last_err = str(e)
            _t.sleep(5*(attempt+1))
    else:
        return None, last_err or "llm_generate falló tras retries"
    # extraer bloque python (limpiar markers si el regex no matcheó)
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
    else:
        # sin fences: quitar cualquier ``` sobrante y leading/trailing
        code = re.sub(r"^```(?:python)?\s*", "", text.strip())
        code = re.sub(r"```\s*$", "", code).strip()
    return code, None

def run_code(code, workdir, timeout=300):
    """Ejecuta el código en workdir; retorna (exit_code, stdout, stderr).
    PYTHONNOUSERSITE=1 para que el venv_dl NO herede site-packages del user
    (que tiene geopandas/pyogrio viejo y rompe read_file)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=str(workdir)) as f:
        f.write(code); script = f.name
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    try:
        r = subprocess.run([PYTHON_EXEC, script], cwd=str(workdir),
                           capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, r.stdout[-2000:], r.stderr[-3000:]
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT {timeout}s"
    finally:
        os.unlink(script)

def llm_fix_code(code, error, prompt, model, backend, max_retries=2):
    """Realimenta el stderr al LLM para que corrija el código (self-consistency).
    Devuelve (código_corregido, None) o (None, error_final)."""
    for attempt in range(1, max_retries+1):
        fix_prompt = (
            f"The previous Python program FAILED with this error:\n---\n{error[-2000:]}\n---\n"
            f"Here is the program that failed:\n```python\n{code}\n```\n"
            f"Fix the program so it runs correctly and solves the original task: {prompt[:1500]}\n"
            f"Output ONLY the corrected Python code between ```python and ``` markers."
        )
        new_code, err = llm_generate(fix_prompt, model, backend)
        if err or not new_code or len(new_code) < 20:
            return None, f"fix retry {attempt} fallo: {err or 'vacio'}"
        code = new_code
    return code, None

def run_eval_program(item, workdir):
    """Ejecuta el eval_program del ítem si existe y es puro (no LLM-visual judge)."""
    ev = item.get("execution", {}).get("eval_script", "")
    if not ev:
        return None, "no eval_script"
    ev_path = SAB_BENCH / "eval_programs" / ev
    if not ev_path.exists():
        return None, f"eval_program no existe: {ev}"
    src = ev_path.read_text()
    # Caveat: si usa gpt4_visual_judge o gold_results, NO es verificación pura
    if "visual_judge" in src or "score_figure" in src or "gold_results" in src:
        return None, f"SKIP (usa LLM-visual judge / gold_results): {ev}"
    # ejecutar
    with tempfile.NamedTemporaryFile("w", suffix="_eval.py", delete=False,
                                     dir=str(SAB_BENCH / "eval_programs")) as f:
        f.write(src); ev_script = f.name
    try:
        r = subprocess.run([sys.executable, ev_script], cwd=str(SAB_BENCH),
                           capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            return 1, (r.stdout or r.stderr)[-500:]
        return 0, (r.stdout or r.stderr)[-500:]
    except subprocess.TimeoutExpired:
        return 0, "eval TIMEOUT"
    finally:
        os.unlink(ev_script)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(DEFAULT_ITEMS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="deepseek-v4-flash:latest")
    ap.add_argument("--backend", choices=["ollama","openrouter"], default="ollama")
    ap.add_argument("--timeout-per-item", type=int, default=600)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--ids", default="", help="Solo evaluar estos ids (coma-separado, ej: sab-48,sab-86)")
    args = ap.parse_args()

    doc = json.load(open(args.items))
    meta, items = doc["meta"], [i for i in doc["items"] if i["split"]=="eval_holdout"]
    # Caveat: los ítems propios (eco-*) aún apuntan a datos no materializados
    # (breviceps_274_pu.rds, phylo_fit_results.rds, era5-corrected-stack) -> skip
    # con razón hasta que se materialicen. Los GIS de SAB tienen datos reales.
    items = [i for i in items if not i["id"].startswith("eco-")]
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        items = [i for i in items if i["id"] in want]
    if args.limit: items = items[:args.limit]
    print(f"== EcoBench-EVAL agent ({args.backend}/{args.model}) == {len(items)} ítems GIS")
    print(f"   caveats: LLM-visual-judge SKIP | gold NO-train | CWD={SAB_BENCH}")

    results = []
    sab_tree = load_sab_tree()
    for it in items:
        iid, q = it["id"], it.get("question","")
        fam = it.get("family","?")
        print(f"\n[{iid}] ({fam})", flush=True)
        if args.dry:
            results.append({"id":iid,"family":fam,"status":"dry","expected":it.get("execution",{}).get("expected")})
            continue
        # 1) generar código — inyectar la estructura real del dataset si es sab-*
        prompt = q
        iid_num = iid.replace("sab-","")
        tree = sab_tree.get(iid_num, "")
        if tree:
            prompt = f"{q}\n\nDataset folder tree (use these EXACT paths, relative to CWD):\n{tree}"
        code, err = llm_generate(prompt, args.model, args.backend)
        if err:
            print(f"  LLM error: {err}"); results.append({"id":iid,"family":fam,"status":"llm_error","error":err}); continue
        if not code or len(code) < 20:
            print("  LLM devolvió código vacío"); results.append({"id":iid,"family":fam,"status":"empty_code"}); continue
        print(f"  código: {len(code)} chars")
        # 2) ejecutar en sandbox (CWD = benchmark/) con retry-fix ante errores
        SAB_BENCH.mkdir(parents=True, exist_ok=True)
        rc, so, se = run_code(code, SAB_BENCH, timeout=args.timeout_per_item)
        attempts = 1
        while rc != 0 and attempts <= 2:
            print(f"  ejecución FAIL rc={rc} (intento {attempts}); enviando error al LLM...")
            fixed, fix_err = llm_fix_code(code, se or so, prompt, args.model, args.backend, max_retries=1)
            if not fixed:
                print(f"  fix falló: {fix_err}"); break
            code = fixed
            rc, so, se = run_code(code, SAB_BENCH, timeout=args.timeout_per_item)
            attempts += 1
        if rc != 0:
            print(f"  ejecución FAIL rc={rc}: {se[:200]}")
            results.append({"id":iid,"family":fam,"status":"exec_fail","rc":rc,
                            "stderr":se[:800],"attempts":attempts})
            continue
        print(f"  ejecución OK (intentos {attempts})")
        # 3) verificar artefacto (existe? path esperado?)
        expected = it.get("execution",{}).get("expected",{})
        exp_path = expected.get("value","") if expected.get("type")=="file_artifact" else ""
        if exp_path:
            full = SAB_BENCH / exp_path
            ok = full.exists()
            print(f"  artifact {exp_path}: {'OK' if ok else 'NO EXISTE'}")
            results.append({"id":iid,"family":fam,"status":"pass" if ok else "fail",
                            "artifact":str(full),"expected":exp_path})
            continue
        # 4) si no es artifact, correr eval_program puro
        ev_ok, ev_msg = run_eval_program(it, SAB_BENCH)
        if ev_ok is None:
            print(f"  eval: {ev_msg}"); results.append({"id":iid,"family":fam,"status":"skip","reason":ev_msg})
        else:
            print(f"  eval_program -> {'PASS' if ev_ok else 'FAIL'}: {ev_msg[:120]}")
            results.append({"id":iid,"family":fam,"status":"pass" if ev_ok else "fail","eval":ev_msg[:300]})

    out = ROOT / "ecobench_eval_results_agent.json"
    json.dump({"meta":meta,"results":results,"model":args.model,"backend":args.backend,
               "caveats":["LLM-visual-judge skipped","gold_no_train","CWD=benchmark"]},
              open(out,"w"), indent=1, ensure_ascii=False)
    from collections import Counter
    st = Counter(r["status"] for r in results)
    print(f"\n== RESULTADO == {dict(st)}")
    print(f"pass@1 = {st.get('pass',0)}/{len(results)} = {100*st.get('pass',0)/max(len(results),1):.0f}%")
    print(f"guardado: {out}")

if __name__ == "__main__":
    main()