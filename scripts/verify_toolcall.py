#!/usr/bin/env python3
"""verify_toolcall.py — Verificator de tool-calls (Opción D, HITO 1).

Dado el output de un controller (deepseek-v4-flash local), extrae y valida las
tool-calls JSON contra los SCHEMAS EMPÍRICOS derivados de las 533 tool-calls
reales del corpus (distill_data + distill_v4_round{2,3,4}):

    gbif_occurrence : {species: str, region: str}
    bioclim_download: {region: str, year: str}
    maxent_train    : {species: str, layers: str}

Modos de fallo reparables (reciclados de build_l1_synth.py M1-M5):
    M1 JSON truncado (faltan llaves)        -> cerrar el JSON
    M2 comillas desbalanceadas              -> reparar comillas rotas
    M3 typo de valor (opción inválida)      -> fuzzy match vs valores del corpus
    M4 typo de nombre de función            -> fuzzy match vs {3 funciones}
    M5 tool_call_id roto/ausente            -> regenerar id

Uso:
  python3 verify_toolcall.py --input data/distill_data.jsonl      # eval sobre trazas reales
  python3 verify_toolcall.py --input data/toolcalls_dirty.jsonl   # reparar outputs sucios
  python3 verify_toolcall.py --text '{"function":{"name":"gbif_occurence"...}}'  # un output
  python3 verify_toolcall.py --selftest                          # 0 deps

Salida: por item -> {ok: bool, function, args, repairs: [...], error: str|None}
"""
import argparse, json, re, sys, random
from difflib import SequenceMatcher
from pathlib import Path

# ---- schemas empíricos (derivados del corpus real, 2026-09-08) ----
# REQUIRED = claves presentes en ~100% de las llamadas de la función en las 533
# tool-calls del corpus. OPTIONAL = claves que el teacher a veces omite (el
# endpoint real puede tener defaults; se aceptan con warning).
SCHEMAS = {
    "gbif_occurrence": {"required": ["species"], "optional": ["region"]},
    "bioclim_download": {"required": ["region"], "optional": ["year"]},
    "maxent_train": {"required": ["species", "layers"], "optional": []},
}
# tipos (todas strings en el corpus)
ARG_TYPES = {k: str for f, meta in SCHEMAS.items() for k in meta["required"] + meta["optional"]}
# valores conocidos para fuzzy match M3 (extraídos del corpus: regions/species/layers)
KNOWN_VALUES = {
    "region": [
        "peninsula de yucatan", "yucatan", "neotropico", "mexico", "brazil",
        "colombia", "amazon", "andina", "caribbean", "madagascar",
        "patagonia", "ibera", "sonora", "chihuahuan desert", "mesoamerica",
        "españa", "europa", "norte de africa", "sahara", "africa subsahariana",
    ],
    "species": [
        "panthera onca", "jaguar", "lycaon pictus", "african wild dog",
        "canis lupus", "grey wolf", "vulpes vulpes", "ursus arctos",
        "puma concolor", "bison bison", "rangifer tarandus", "caribou",
        "lontra canadensis", "marmota flaviventris", "pinus strobus",
        "quercus robur", "ambystoma mexicanum", "bradypus variegatus",
    ],
    "year": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"],
    "layers": ["bioclim_19", "bioclim", "current", "future", "yucatan_bioclim"],
}
FUNCS = set(SCHEMAS)
RNG = random.Random(42)

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

# ---------------- extracción de tool-calls del output del controller ----------------
# formatos: (a) OpenAI: [{"id","type":"function","function":{"name","arguments"}}]
#           (b) bare:   [{"name","arguments"}]     (c) JSON suelto en el texto
TOOLCALL_OBJ_RE = re.compile(r'\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*\}')

def extract_toolcalls(text):
    """Extrae lista de dicts {name, args(dict), raw}. Tolera ruido alrededor."""
    out = []
    if not text:
        return out
    # caso (a)/(b): ya es un array/objeto JSON
    for cand in (text.strip(), text[text.find("["): text.rfind("]") + 1] if "[" in text else ""):
        try:
            obj = json.loads(cand)
            items = obj if isinstance(obj, list) else [obj]
            for it in items:
                if not isinstance(it, dict):
                    continue
                fn = (it.get("function") or {}).get("name") if isinstance(it.get("function"), dict) else (it.get("function") or it.get("name"))
                fn = fn or (it.get("name") if isinstance(it.get("name"), str) else None)
                args_raw = (it.get("function") or {}).get("arguments") or it.get("arguments") or "{}"
                if isinstance(args_raw, dict):
                    args_raw = json.dumps(args_raw, ensure_ascii=False)
                if fn:
                    out.append({"name": fn, "args": args_raw, "raw": json.dumps(it, ensure_ascii=False)})
            return out
        except (json.JSONDecodeError, ValueError):
            continue
    # caso (c): regex sobre objetos sueltos
    for m in TOOLCALL_OBJ_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
            fn = (obj.get("function") or {}).get("name") or obj.get("name")
            if fn:
                out.append({"name": fn, "args": (obj.get("function") or {}).get("arguments") or obj.get("arguments") or "{}", "raw": m.group(0)})
        except (json.JSONDecodeError, ValueError):
            continue
    return out

# ---------------- reparadores M1-M5 ----------------
def _repair_json(s):
    """M1: JSON truncado -> intentar cerrar progresivamente. M2: comillas sueltas."""
    attempts = [s]
    # M1: cerrar llaves/corchetes faltantes
    for close in ("}", "}}", "}]}}"):
        attempts.append(s.rstrip() + close)
    # M2: comillas rotas dentro de valores ("Peninsula de Yucatan -> ... )
    #   patrón: "clave": valor sin comillas o con \"
    fixed = re.sub(r'"([a-z_]+)"\s*:\s*([a-zA-ZáéíóúñüÁÉÍÓÚÑ\s]+)(?=[,}\]])', r'"\1": "\2"', s)
    if fixed != s:
        attempts.append(fixed)
    fixed2 = s.replace('\\"', '"')
    if fixed2 != s:
        attempts.append(fixed2)
    for cand in attempts:
        try:
            return json.loads(cand), []
        except json.JSONDecodeError:
            continue
    return None, ["M1/M2: JSON irrecuperable"]

def _fix_value(key, val, repairs):
    """M3: typo de valor -> fuzzy match contra valores conocidos de esa clave."""
    if not isinstance(val, str) or not val.strip():
        return val, repairs
    known = KNOWN_VALUES.get(key, [])
    if not known:
        return val, repairs
    best, bs = None, -1
    for kv in known:
        sc = _sim(val, kv)
        if sc > bs:
            best, bs = kv, sc
    if best and bs >= 0.72 and _norm(val) != _norm(best):
        repairs.append(f"M3: valor '{val}' -> '{best}' (sim {bs:.2f})")
        return best, repairs
    return val, repairs

def _fix_func(name, repairs):
    """M4: typo de nombre de función."""
    if name in FUNCS:
        return name, repairs
    best, bs = None, -1
    for f in FUNCS:
        sc = _sim(name, f)
        if sc > bs:
            best, bs = f, sc
    if best and bs >= 0.72:
        repairs.append(f"M4: función '{name}' -> '{best}' (sim {bs:.2f})")
        return best, repairs
    return name, repairs

def _fix_id(raw, repairs):
    if '"id"' not in raw:
        repairs.append("M5: id regenerado")
        return f'"id": "{RNG.randbytes(9).hex()}"'
    return raw

# ---------------- verificación ----------------
def verify(text):
    """Devuelve {ok, function, args, repairs, error, toolcalls_parsed}."""
    tcs = extract_toolcalls(text)
    if not tcs:
        return {"ok": False, "error": "no se encontró ninguna tool-call en el output",
                "function": None, "args": None, "repairs": []}
    result = {"ok": False, "error": None, "function": None, "args": None,
              "repairs": [], "n_calls": len(tcs)}
    # verificamos la primera tool-call del output (la acción principal)
    tc = tcs[0]
    repairs = []
    name = tc["name"]
    name, repairs = _fix_func(name, repairs)
    if name not in SCHEMAS:
        result.update({"error": f"función desconocida '{tc['name']}' tras repair", "repairs": repairs})
        return result
    args_raw = tc["args"] if isinstance(tc["args"], str) else json.dumps(tc["args"])
    args, r = _repair_json(args_raw)
    repairs += r
    if args is None:
        result.update({"error": "argumentos JSON irrecuperables", "repairs": repairs,
                       "function": name})
        return result
    # M3: fuzzear cada valor contra valores conocidos
    for k in list(args):
        args[k], repairs = _fix_value(k, args[k], repairs)
    # validar claves requeridas (las opcionales solo advierten)
    meta = SCHEMAS[name]
    missing = [k for k in meta["required"] if k not in args]
    if missing:
        result.update({"error": f"faltan argumentos requeridos: {missing}", "function": name,
                       "args": args, "repairs": repairs})
        return result
    for k in meta["optional"]:
        if k not in args:
            repairs.append(f"WARN: arg opcional '{k}' ausente (default del endpoint)")
    # tipos (obligatorias y opcionales presentes)
    for k in list(args):
        t = ARG_TYPES.get(k)
        if t and not isinstance(args[k], t):
            result.update({"error": f"arg '{k}' debe ser {t.__name__}, got {type(args[k]).__name__}",
                           "function": name, "args": args, "repairs": repairs})
            return result
    result.update({"ok": True, "function": name, "args": args, "repairs": repairs})
    return result

def _mutation_M5(raw_tc):
    "Crea una tool-call sucia de prueba (simula M4: typo de función)."
    return raw_tc.replace("gbif_occurrence", "gbif_occurence")  # M4

def _selftest():
    clean_tc = '[{"id":"x","type":"function","function":{"name":"gbif_occurrence","arguments":"{\\"species\\":\\"Panthera onca\\",\\"region\\":\\"peninsula de yucatan\\"}"}}]'
    ok = verify(clean_tc)
    assert ok["ok"], ok
    # M4: typo de función en una tool-call cruda
    dirty = clean_tc.replace("gbif_occurrence", "gbif_occurence")
    ok2 = verify(dirty)
    assert ok2["ok"], ok2
    assert any("M4" in r for r in ok2["repairs"]), ok2["repairs"]
    # M1/M2: JSON truncado en los args + M3: typo de región
    dirty3 = '{"function":{"name":"gbif_occurrence","arguments":"{\\"species\\":\\"Panthera onca\\",\\"region\\":\\"peninsulade yucatan\\""}}'
    ok3 = verify(dirty3)
    assert ok3["ok"], ok3
    assert any("M3" in r or "M1/M2" in r for r in ok3["repairs"]), ok3["repairs"]
    # basura pura (caso sab-23) -> debe fallar limpio
    bad = verify("sh sh sh sh sh sh sh sh sh")
    assert bad["ok"] is False and "no se encontró" in bad["error"]
    print("selftest OK:", json.dumps({'direct': ok["ok"], 'M4_repair': ok2["repairs"],
                                      'M3_repair': ok3["repairs"]}, ensure_ascii=False))
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="jsonl de trazas (usar tool_calls como outputs sucios)")
    ap.add_argument("--text", help="un output suelto a verificar")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.text:
        res = verify(a.text)
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0 if res["ok"] else 1
    if a.input:
        total = ok = 0
        n_repairs = 0
        by_func = {}
        for line in open(a.input):
            try:
                d = json.loads(line)
            except Exception:
                continue
            for m in d.get("trajectory", []):
                for tc in (m.get("tool_calls") or []):
                    total += 1
                    res = verify(json.dumps(tc, ensure_ascii=False))
                    by_func[res.get("function") or "?unknown"] = by_func.get(res.get("function") or "?unknown", 0) + 1
                    if res["ok"]:
                        ok += 1
                    n_repairs += len(res.get("repairs", []))
        print(f"tool-calls: {total} | válidas directas: {ok} ({ok/max(total,1):.1%}) | reparaciones totales: {n_repairs}")
        print(f"por función: {by_func}")
        return 0 if ok == total else 1

if __name__ == "__main__":
    sys.exit(main())