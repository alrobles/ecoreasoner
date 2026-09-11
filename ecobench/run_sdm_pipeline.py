#!/usr/bin/env python3
"""run_sdm_pipeline.py — Pipeline SDM de ejemplo con controller+verificator+resolver.

Flujo:
  1. Recibe pregunta (ej. "Modela el nicho de Panthera onca en Yucatan").
  2. Controller (local v4-flash u ollama) emite secuencia de tool-calls.
  3. Verificator valida/repara cada tool-call.
  4. Resolver ejecuta GBIF, bioclim y maxent.
  5. Reporta traza y resultado final.

Modos:
  --demo: tool-calls hardcodeados (sin ollama) para probar resolver.
  --real: usa ollama/openrouter real.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "scripts"))
sys.path.insert(0, str(ROOT))

from verify_toolcall import verify
from tools_resolver import resolve
from run_controller_verificator import run_one, OLLAMA_URL


def demo_pipeline(species: str, region: str, year: str, layers: str):
    """Ejecuta una secuencia hardcodeada de tool-calls con verificación y resolución real."""
    steps = [
        {"tool": "gbif_occurrence", "arguments": {"species": species, "region": region}},
        {"tool": "bioclim_download", "arguments": {"region": region, "year": year}},
        {"tool": "maxent_train", "arguments": {"species": species, "layers": layers}},
    ]
    trace = []
    for step in steps:
        text = json.dumps(step, ensure_ascii=False)
        ver = verify(text)
        if not ver["ok"]:
            trace.append({"step": step["tool"], "ok": False, "error": ver["error"]})
            break
        status, msg = resolve(ver["function"], ver["args"])
        trace.append({
            "step": ver["function"],
            "args": ver["args"],
            "repairs": ver["repairs"],
            "resolve_status": status,
            "resolve_msg": msg,
            "ok": status in ("ok", "partial"),
        })
    return trace


def llm_pipeline(question: str, model: str, backend: str):
    """Controller real con retry; cada tool-call validado y resuelto."""
    # HITO 2 simplificado: usamos run_one varias veces con memoria de resultados.
    # En MVP complejo se necesita memoria de estadio (gbif_result, bioclim_result).
    return run_one("sdm-pipeline", question, model, backend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="Modela el nicho de Panthera onca en la peninsula de yucatan para 2020")
    ap.add_argument("--species", default="Panthera onca")
    ap.add_argument("--region", default="peninsula de yucatan")
    ap.add_argument("--year", default="2020")
    ap.add_argument("--layers", default="bioclim_19")
    ap.add_argument("--demo", action="store_true", help="tool-calls hardcodeados, sin LLM")
    ap.add_argument("--model", default="deepseek-v4-flash:latest")
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--out", default="ecobench/sdm_pipeline_result.json")
    a = ap.parse_args()

    if a.demo:
        print("[demo] Pipeline hardcodeado sin LLM")
        trace = demo_pipeline(a.species, a.region, a.year, a.layers)
    else:
        print(f"[llm] Backend {a.backend}/{a.model}")
        trace = llm_pipeline(a.question, a.model, a.backend)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "question": a.question,
        "species": a.species,
        "region": a.region,
        "year": a.year,
        "layers": a.layers,
        "demo": a.demo,
        "trace": trace,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"Resultado guardado: {out}")

    ok = sum(1 for t in trace if t.get("ok"))
    print(f"Pasos OK: {ok}/{len(trace)}")
    for t in trace:
        print(f"  - {t['step']}: {t.get('resolve_status', t.get('ok'))}")


if __name__ == "__main__":
    main()
