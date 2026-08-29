#!/usr/bin/env python3
"""
distill_b1_multiteacher.py — Destilacion multi-teacher para EcoReasoner Bloque B1.

Fuentes: PubMed FTS5 (97GB, 36M papers) + GBIF Literature FTS (146MB, 61K papers)
Teachers: DeepSeek-V4-Flash local (:20006) con fallback a OpenRouter Nemotron free
Salida: trazas [CONTEXT]→[REASONING]→[CODE] con metadatos, code_valid, teacher

Uso:
  python3 distill_b1_multiteacher.py --query "species distribution model" --limit 50 --output sci_v2_b1.jsonl --append
  python3 distill_b1_multiteacher.py --batch b1_queries.txt --limit 50 --output sci_v2_b1.jsonl --append
  python3 distill_b1_multiteacher.py --source gbif --query "MaxEnt" --limit 30 --output sci_v2_b1.jsonl --append
"""
from __future__ import annotations
import argparse, json, os, sys, time, re, ast, sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import urllib.request, urllib.error

DB_DIR = os.environ.get("PUBMED_INDEX_DIR", "/home/a474r867/work/pubmed/index")
DB_PATH = os.path.join(DB_DIR, "pubmed_fts.db")
GBIF_DB = os.environ.get("GBIF_LIT_DB", "/home/a474r867/work/gbif_literature/gbif_literature_fts.db")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:20006/v1/chat/completions")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-v4-flash:latest")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Fix 1: clave OpenRouter multi-ruta + re-resolucion dinamica del teacher ---
def _load_openrouter_key():
    """Busca la clave OpenRouter en varias rutas y en env (fix: antes solo
    ~/env/openrouter-key, inexistente en HPC -> 'All teachers failed')."""
    candidates = [
        os.path.expanduser("~/env/openrouter-key"),
        os.path.expanduser("~/env/hermes-ecoseek-key"),
        os.path.expanduser("~/.config/openrouter/key"),
    ]
    for path in candidates:
        if os.path.exists(path):
            key = open(path).read().strip()
            if key:
                return key
    return os.environ.get("OPENROUTER_KEY", "")

def _find_ollama_endpoint():
    """Resuelve el endpoint real del teacher v4-flash en el HPC (red interna).

    El túnel 127.0.0.1:20006 solo existe en reumanlab (máquina local), NO en el
    login node del HPC. Desde el HPC hay que apuntar directo al nodo:puerto del
    job ollama-v4serve (efímero: rota cada ~6h). Devuelve la URL v1 o None.
    """
    try:
        import subprocess
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "a474r867"), "-n", "ollama-v4serve",
             "-t", "R", "-h", "-o", "%i|%N"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        if not out:
            return None
        jobid, node = out.split("|")[0], out.split("|")[1].strip()
        workdir = os.path.expanduser("~/work/ollama")
        for prefix in (f"v4flash-serve-output-{jobid}",):
            pf = os.path.join(workdir, prefix)
            if os.path.exists(pf):
                port = None; host = None
                for line in open(pf, errors="ignore"):
                    if line.startswith("Port:"): port = line.split(":",1)[1].strip()
                    if line.startswith("Node:"): host = line.split(":",1)[1].strip()
                if port and host:
                    return f"http://{host}:{port}/v1/chat/completions"
    except Exception:
        pass
    return None

COT_SYSTEM_PROMPT = """You are an expert ecologist and scientific programmer. Your task is to read a scientific paper abstract and generate a Chain-of-Thought reasoning trace.

For each paper, produce a structured response in THREE sections:

[CONTEXT]
Summarize the ecological problem, hypothesis, and methodology from the paper in 3-5 sentences. Include the key ecological question and the data/methods approach.

[REASONING]
Walk through the scientific reasoning step by step. For each step, explain:
- WHAT ecological question or computational challenge this step addresses
- WHY this specific method was chosen (over alternatives)
- HOW the method works (conceptually, not just technically)
- WHAT ecological assumptions it makes and their implications
Include 4-7 reasoning steps.

[CODE]
Write Python or R code that implements the core method from this paper. Use real ecological packages. For species distribution modeling (SDM/MaxEnt), prefer the maxentcpp R package (alrobles/maxentcpp) over dismo. Other packages: scikit-learn, raster, sf, terra, StatsModels, PyMC, glmmTMB, sdm, ensemble SDM, etc. The code should be:
- Runnable (valid syntax, imports included)
- Self-contained (simulated data if real data unavailable)
- Commented with ecological interpretation
- 20-60 lines

IMPORTANT: Output ONLY the CoT trace. No preamble, no "Here is...", no markdown formatting outside the three sections."""

COT_USER_TEMPLATE = """Paper: {title}
Journal: {journal} ({year})
Authors: {authors}
MeSH Terms: {mesh_terms}

Abstract:
{abstract}

Generate a Chain-of-Thought reasoning trace for this paper. Follow the [CONTEXT] → [REASONING] → [CODE] format exactly."""

def search_pubmed(query, limit=50, year_min=None, year_max=None, mesh_filter="Ecology", language="eng"):
    if not os.path.exists(DB_PATH):
        sys.exit(f"FTS index not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    sql = "SELECT a.pmid, a.doi, a.title, a.abstract, a.journal, a.pub_year, a.authors, a.mesh_terms, a.keywords, a.language, rank FROM articles_fts f JOIN articles a ON a.pmid = f.rowid WHERE articles_fts MATCH ?"
    params = [query]
    if year_min: sql += " AND a.pub_year >= ?"; params.append(year_min)
    if year_max: sql += " AND a.pub_year <= ?"; params.append(year_max)
    if mesh_filter: sql += " AND a.mesh_terms LIKE ?"; params.append(f"%{mesh_filter}%")
    if language: sql += " AND a.language = ?"; params.append(language)
    sql += " AND a.abstract IS NOT NULL AND a.abstract != ''"
    sql += " LIMIT ?"; params.append(limit)
    results = [dict(row) for row in conn.execute(sql, params).fetchall()]
    conn.close()
    return results

def search_gbif(query, limit=50, year_min=None, year_max=None):
    if not os.path.exists(GBIF_DB):
        return []
    conn = sqlite3.connect(GBIF_DB, timeout=30)
    conn.row_factory = sqlite3.Row
    sql = "SELECT gbif_id, title, abstract, authors, keywords, topics, doi, year, source, publisher FROM literature WHERE literature MATCH ?"
    params = [query]
    if year_min: sql += " AND year >= ?"; params.append(str(year_min))
    if year_max: sql += " AND year <= ?"; params.append(str(year_max))
    sql += " AND abstract IS NOT NULL AND abstract != ''"
    sql += " LIMIT ?"; params.append(limit)
    results = [dict(row) for row in conn.execute(sql, params).fetchall()]
    conn.close()
    # Normalizar al mismo formato que PubMed
    for r in results:
        r["pmid"] = r.get("gbif_id")
        r["journal"] = r.get("source") or r.get("publisher") or "Unknown"
        r["pub_year"] = r.get("year")
        r["mesh_terms"] = r.get("topics") or r.get("keywords") or "None"
    return results

def _ollama_url_resolved():
    """URL del teacher: env override > job descubierto > tunel local por defecto."""
    env = os.environ.get("OLLAMA_URL", "")
    if env:
        return env
    found = _find_ollama_endpoint()
    return found if found else "http://127.0.0.1:20006/v1/chat/completions"

def call_ollama(system_prompt, user_prompt, model=OLLAMA_MODEL, max_tokens=4096, retries=3):
    payload = json.dumps({"model": model, "messages": [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], "temperature": 0.2, "max_tokens": max_tokens, "reasoning_effort": "none"}).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        url = _ollama_url_resolved()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type":"application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            msg = result["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning") or ""
            return {"success": True, "content": content, "model": result.get("model", model), "usage": result.get("usage", {})}
        except Exception as e:
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(8 * (attempt + 1))   # backoff 8s/16s antes de rendirse
    return {"success": False, "error": f"{last_err} (url={_ollama_url_resolved()})"}

def call_openrouter(system_prompt, user_prompt, model="nvidia/nemotron-3-ultra-550b-a55b:free", max_tokens=4096, retries=2):
    key = _load_openrouter_key()
    if not key: return {"success": False, "error": "No OpenRouter key (busque en ~/env/openrouter-key, ~/env/hermes-ecoseek-key, OPENROUTER_KEY)"}
    payload = json.dumps({"model": model, "messages": [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], "temperature": 0.2, "max_tokens": max_tokens}).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(OPENROUTER_URL, data=payload, headers={"Authorization": f"Bearer {key}", "Content-Type":"application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            msg = result["choices"][0]["message"]
            content = msg.get("content") or ""
            return {"success": True, "content": content, "model": result.get("model", model), "usage": result.get("usage", {})}
        except Exception as e:
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(5)
    return {"success": False, "error": last_err}

def call_teacher(system_prompt, user_prompt):
    r = call_ollama(system_prompt, user_prompt)
    if r["success"] and r["content"].strip(): return r, "local"
    r = call_openrouter(system_prompt, user_prompt, "nvidia/nemotron-3-ultra-550b-a55b:free")
    if r["success"] and r["content"].strip(): return r, "openrouter-ultra"
    r = call_openrouter(system_prompt, user_prompt, "nvidia/nemotron-3-super-120b-a12b:free")
    if r["success"] and r["content"].strip(): return r, "openrouter-super"
    return {"success": False, "error": "All teachers failed"}, "none"

def parse_cot(content):
    sections = {"context": "", "reasoning": "", "code": ""}
    patterns = {"context": r"\[CONTEXT\]\s*\n?(.*?)(?=\[REASONING\]|\[CODE\]|$)", "reasoning": r"\[REASONING\]\s*\n?(.*?)(?=\[CODE\]|$)", "code": r"\[CODE\]\s*\n?(.*?)$"}
    for key, pat in patterns.items():
        m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
        if m: sections[key] = m.group(1).strip()
    if not sections["context"] or not sections["reasoning"]: return None
    return sections

def validate_code(code):
    if not code.strip(): return False
    clean = re.sub(r"^```(?:python|r)?\s*\n", "", code.strip())
    clean = re.sub(r"\n```\s*$", "", clean)
    try:
        ast.parse(clean)
        return True
    except SyntaxError:
        r_indicators = ['library(', 'function(', '<-', 'data.frame', 'glm(', 'lm(', 'raster(', 'maxent_']
        return any(ind in clean for ind in r_indicators)

def main():
    parser = argparse.ArgumentParser(description="B1 multi-teacher distillation")
    parser.add_argument("--query", help="FTS5 search query")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="sci_v2_b1.jsonl")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--year-min", type=int, default=2015)
    parser.add_argument("--year-max", type=int, default=2026)
    parser.add_argument("--mesh-filter", default="Ecology")
    parser.add_argument("--batch", help="File with one query per line")
    parser.add_argument("--source", choices=["pubmed","gbif","both"], default="pubmed")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    if args.batch:
        with open(args.batch) as f:
            queries = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    elif args.query:
        queries = [args.query]
    else:
        sys.exit("Need --query or --batch")

    seen_pmids = set()
    if args.append and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try: seen_pmids.add(json.loads(line)["pmid"])
                except: pass
        print(f"Loaded {len(seen_pmids)} existing PMIDs for dedup", file=sys.stderr)

    mode = "a" if args.append else "w"
    total_generated = 0
    total_failed = 0

    with open(args.output, mode, encoding="utf-8") as fout:
        for qi, query in enumerate(queries):
            print(f"\n[Query {qi+1}/{len(queries)}] '{query}'", file=sys.stderr, flush=True)
            papers = []
            if args.source in ("pubmed", "both"):
                papers += search_pubmed(query, limit=args.limit, year_min=args.year_min, year_max=args.year_max, mesh_filter=args.mesh_filter)
            if args.source in ("gbif", "both"):
                papers += search_gbif(query, limit=args.limit, year_min=args.year_min, year_max=args.year_max)
            print(f"  Found {len(papers)} papers ({args.source})", file=sys.stderr, flush=True)

            for i, paper in enumerate(papers):
                pmid = paper.get("pmid", f"unknown_{qi}_{i}")
                if pmid in seen_pmids: continue
                title = paper.get("title", "")[:120]
                print(f"  [{i+1}/{len(papers)}] {pmid}: {title}", file=sys.stderr, flush=True)
                user_prompt = COT_USER_TEMPLATE.format(
                    title=title, journal=paper.get("journal","Unknown"), year=paper.get("pub_year","Unknown"),
                    authors=(paper.get("authors","") or "Unknown")[:300],
                    mesh_terms=(paper.get("mesh_terms","") or "None")[:200],
                    abstract=(paper.get("abstract","") or "No abstract")[:3000])
                result, teacher = call_teacher(COT_SYSTEM_PROMPT, user_prompt)
                if not result["success"]:
                    print(f"    FAILED: {result['error']}", file=sys.stderr, flush=True)
                    total_failed += 1; time.sleep(args.delay); continue
                sections = parse_cot(result["content"])
                if not sections:
                    sections = {"context": "[PARSE_FAILED]", "reasoning": result["content"][:3000], "code": ""}
                cv = validate_code(sections["code"])
                trace = {"pmid": pmid, "doi": paper.get("doi"), "title": paper.get("title"),
                         "journal": paper.get("journal"), "pub_year": paper.get("pub_year"),
                         "authors": paper.get("authors"), "mesh_terms": paper.get("mesh_terms"),
                         "search_query": query, "source": args.source,
                         "context": sections["context"], "reasoning": sections["reasoning"],
                         "code": sections["code"], "code_valid": cv,
                         "teacher": teacher, "model": result.get("model"), "usage": result.get("usage"),
                         "generated_at": datetime.utcnow().isoformat()}
                fout.write(json.dumps(trace, ensure_ascii=False) + "\n")
                fout.flush(); seen_pmids.add(pmid); total_generated += 1
                print(f"    OK ({teacher}, code_valid={cv})", file=sys.stderr, flush=True)
                time.sleep(args.delay)

    print(f"\n{'='*60}\nDone! Generated {total_generated} traces ({total_failed} failed)\nOutput: {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()
