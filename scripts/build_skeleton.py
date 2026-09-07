#!/usr/bin/env python3
"""
build_skeleton.py — extractor de ESQUELETOS DE ARGUMENTO (Fase 3, F0b).

Convierte full-text cientifico en el atomo de entrenamiento del diseno Fase3-new:

    [OBSERVACION]<sep>[HIPOTESIS]<sep>[PREDICCION]<sep>[EVIDENCIA]<sep>[CONCLUSION]

= aprender la GRAMATICA de la inferencia (que pasa de una etapa a la siguiente),
no la superficie textual. 0 GPU: heuristica de secciones IMRaD + abstracts
estructurados + frases-faro.

MODO SAMPLE:
    python3 build_skeleton.py --input X.jsonl --mode sample --per-domain 150 \
        --report /tmp/skel_report.json
    -> procesa hasta N docs POR DOMINIO (aleatorio) y reporta COBERTURA por
       estrategia y por dominio. NO escribe corpus.

MODO FULL:
    python3 build_skeleton.py --input X.jsonl --mode full --out \
        data/skeleton/train_skeleton.jsonl --per-domain 50000 --min-stages 3
    -> filtra docs con >= --min-stages etapas, balancea a --per-domain por
       dominio, escribe el corpus de esqueletos + reporte al lado.

Etapas: obs, hyp, pred, evid, conc. <sep> real usado: salto de linea mas
etiqueta textual "[Etapa]" (tokens normales del tokenizer LLaDA, de modo que
el denoising span puede reconstruir etapas enteras).
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# strip de LaTeX (copiado del pipeline arXiv; sin dependencias)
# ---------------------------------------------------------------------------
_LATEX_CLEAN = [
    (r"\\documentclass[^\n]*", " "),
    (r"\\usepackage[^\n]*", " "),
    (r"\\renewcommand[^\n]*", " "),
    (r"\\bibliographystyle[^\n]*", " "),
    (r"\\begin\{document\}", " "),
    (r"\\end\{document\}", " "),
    (r"\\maketitle", " "),
    (r"\\date\{[^}]*\}", " "),
    (r"\\title\{[^}]*\}", " "),
    (r"\\author\{[^}]*\}", " "),
    (r"\\thanks\{[^}]*\}", " "),
    (r"\\begin\{[a-z]*\}", " "),
    (r"\\end\{[a-z]*\}", " "),
    (r"\\textbf\{([^}]*)\}", r"\1"),
    (r"\\textit\{([^}]*)\}", r"\1"),
    (r"\\emph\{([^}]*)\}", r"\1"),
    (r"\\em\s*\{([^}]*)\}", r"\1"),
    (r"\\textrm\{([^}]*)\}", r"\1"),
    (r"\\textsc\{([^}]*)\}", r"\1"),
    (r"\\mathrm\{([^}]*)\}", r"\1"),
    (r"\\mathcal\{([^}]*)\}", r"\1"),
    (r"\\cdot", " * "), (r"\\times", " x "), (r"\\approx", " ~ "),
    (r"\\pm", " +/- "), (r"\\le", " <="), (r"\\ge", " >="), (r"\\ll", " << "),
    (r"\\gg", " >> "), (r"\\&", " & "), (r"\\%", " % "), (r"\\$", " $ "),
    (r"\\#", " # "), (r"\\_", "_"), (r"\\\{", "{"), (r"\\\}", "}"),
    (r"\\[a-zA-Z]+\*?", " "),   # quitar comandos restantes
    (r"~", " "), (r"\\(\\|\[|\])", " "),
    (r"\s+", " "),
]
_LATEX_CLEAN_RE = [(re.compile(p), r) for p, r in _LATEX_CLEAN]


def strip_latex(src: str) -> str:
    if not src or ("\\documentclass" not in src and "\\begin{" not in src):
        return src
    for pat, rep in _LATEX_CLEAN_RE[:8]:   # preambulo: frases enteras
        src = pat.sub(rep, src)
    for pat, rep in _LATEX_CLEAN_RE[8:]:
        src = pat.sub(rep, src)
    return src.strip()


# ---------------------------------------------------------------------------
#  Frases-faro por etapa (regex sobre oraciones en minusculas)
# ---------------------------------------------------------------------------
_OBS = r"(?:we (?:report|present|investigate|examine|study|analy[sz]e)|this (?:study|paper|work|article)|the aim|objective of|to better understand)"
_HYP = r"(?:we (?:hypothesi[sz]e|hypothesi[sz]ed|expect(?:ed)?|tested? whether)|our (?:hypothesis|prediction)|the hypothesis|a hypothesis|could (?:explain|affect|drive|determine))"
_PRED = r"(?:we (?:predict|expect)|predict(?:ed)? that|we (?:therefore|thus) (?:expect|predict)|would (?:lead|show|increase|decrease|lower)|if .{10,80} then|will be (?:positively|negatively))"
_EVID = r"(?:we (?:found|observed|show[ed]?|demonstrate[ed]?|report(?:ed)? that)|our (?:results|findings|data|analys[ei]s) (?:show|suggest|indicate|demonstrate|reveal|support)|results (?:of|from|show|indicate)|statistically significant|significant(?:ly)? (?:effect|difference|association))"
_CONC = r"(?:we conclude|in conclusion|these (?:results|findings|data) (?:suggest|indicate|support|demonstrate)|taken together|overall,? (?:our|these)|implications? (?:of|for)|suggest(?:ing|s)? that)"

_FARO_RE = {
    "conc": re.compile(_CONC),
    "evid": re.compile(_EVID),
    "pred": re.compile(_PRED),
    "hyp": re.compile(_HYP),
    "obs": re.compile(_OBS),
}

# Etiquetas de abstracts estructurados — regex INLINE (no anclada a inicio de
# linea), robusto a texto corrido SIN newlines (el corpus real es asi).
_STRUCT_PAT = re.compile(
    r"(?i)(?:Background|Introduction|Background\s*/?\s*Objectives?|Objective[s]?|Aim[s]?|"
    r"Design|Setting|Participants|Methods?|Methodology|Measurements?|"
    r"Results?|Findings?|Main findings?|Outcomes?|Conclusions?|"
    r"Discussion|Interpretation)\s*[:.]\s*"
)

# Encabezados de seccion IMRaD (linea corta tipo titulo de seccion)
_IMRAD_HEADER = re.compile(
    r"^\s*(?:"
    r"(?:introduction|background|introducit|[oó]n|methods?|material(?:s)? and methods?|"
    r"methodology|experimental (?:section|methods?|procedure|setup)|"
    r"results?|findings?|results and discussion|discussion|conclusions?|"
    r"discussion and conclusions?|implications?|summary|synthesis|"
    r"[0-9]+\.?\s*(?:introduction|background|methods?|results?|discussion|conclusion))"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _norm_label(lab: str) -> str:
    lab = lab.lower().strip(" :.")
    if lab in ("background", "background/objectives", "introduction", "objective",
               "objectives", "aim", "aims", "setting", "participants", "simple"):
        return "obs"
    if lab in ("methods", "methodology", "design", "measurements"):
        return "met"
    if lab in ("results", "findings", "main findings", "outcomes", "observations"):
        return "evid"
    if lab in ("conclusions", "conclusion", "discussion", "interpretation", "implications",
               "summary"):
        return "conc"
    if lab in ("prediction", "predictions", "predicted", "hypothesis", "hypotheses"):
        return "hyp"
    return "unk"


_STAGES = ("obs", "hyp", "pred", "evid", "conc")

# ---------------------------------------------------------------------------
#  Utilidades de texto
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z]\s*|[0-9]|[{\\[])")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", strip_latex(text or "")).strip()


def _sents(text: str, max_chars: int = 4000):
    return [s.strip() for s in _SENT_SPLIT.split(_clean(text)[:max_chars]) if s.strip()]


# ---------------------------------------------------------------------------
#  Extractores por estrategia (cada una devuelve dict etapa->texto)
# ---------------------------------------------------------------------------
def abstract_structured(text: str) -> dict:
    """Abstract estructurado con etiquetas INLINE: 'Background: ... Results: ...'."""
    head = _clean(text)[:7000]
    ms = list(_STRUCT_PAT.finditer(head))
    if len(ms) < 2:
        return {}
    # densidad: el abstract estructurado tiene las etiquetas juntas al inicio
    if ms[-1].start() - ms[0].start() > 6000:
        return {}
    out: dict = {}
    for i, m in enumerate(ms):
        start, end = m.end(), (ms[i + 1].start() if i + 1 < len(ms) else len(head))
        body = head[start:end].strip(" :\n")[:900]
        if len(body) < 20 and i + 1 < len(ms):
            continue
        kind = _norm_label(m.group(0).lower())
        if kind == "unk" or kind == "met":
            continue
        if kind not in out:
            out[kind] = body
    return out


def imrad_sections(text: str) -> dict:
    """Encabezados de seccion IMRaD (linea corta)."""
    if "\n" not in (text or ""):
        return {}
    lines = [l.strip() for l in (text or "").split("\n")]
    cap = min(len(lines), 800)
    cur, out = None, {}
    acc = []

    def flush():
        nonlocal acc
        if cur and acc:
            body = " ".join(x for x in acc if x)[:1500]
            if len(body) > 120:
                out.setdefault(cur, body)
        acc = []

    for i in range(cap):
        l = lines[i]
        is_header = len(l) < 90 and _IMRAD_HEADER.match(l + "\n")
        if is_header:
            flush()
            ll = l.lower()
            cur = ("obs" if ll.startswith(("intro", "backg"))
                   else "met" if ll.startswith(("method", "experimental", "model",
                                                "study area", "data", "statist"))
                   else "evid" if ll.startswith(("result", "finding"))
                   else "conc" if ll.startswith(("discussion", "conclu", "summary", "implic"))
                   else None)
        elif cur and cur != "met":
            acc.append(l)
    flush()
    return {k: v for k, v in out.items() if k != "met"}


def phrases_fallback(text: str) -> dict:
    """Frases-faro sobre el texto completo. hyp/obs tempranas, evid/conc tardias."""
    sents = _sents(text)
    out: dict = {}
    if not sents:
        return out
    mid = max(1, len(sents) // 3)
    early, late = sents[: mid + 1], sents[mid:]
    for grp, win in ((late, "evid"), (late, "conc"), (early, "hyp"), (early, "obs")):
        for s in grp:
            if _FARO_RE[win].search(s.lower()) and len(s) > 45:
                out[win] = s[:400]
                break
    return out


def need_obs(text: str, stages: dict) -> dict:
    if "obs" in stages:
        return stages
    for s in _sents(text, 2000)[:4]:
        if len(s) > 40:
            stages["obs"] = s[:200]
            break
    return stages


# ---------------------------------------------------------------------------
#  Serializador
# ---------------------------------------------------------------------------
_SEP_MAP = {
    "obs": "[OBSERVACION]", "hyp": "[HIPOTESIS]", "pred": "[PREDICCION]",
    "evid": "[EVIDENCIA]", "conc": "[CONCLUSION]",
}


def serialize(stages: dict) -> str:
    parts = [f"{_SEP_MAP[k]} {stages[k].strip()}" for k in _STAGES if stages.get(k)]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="jsonl de full-text (v7_curated/phys)")
    ap.add_argument("--mode", choices=["sample", "full"], default="sample")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--per-domain", type=int, default=150)
    ap.add_argument("--min-stages", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kept_by_domain = defaultdict(list)
    seen_domains = set()
    n_total = 0
    stage_hist = Counter()
    strat_count = Counter()
    strat_kept = Counter()

    def add_sample(d):
        dom = d.get("domain", "?")
        seen_domains.add(dom)
        text = d.get("text") or ""
        stages, strat = {}, "none"
        ab = abstract_structured(text)
        if ab:
            stages.update(ab)
            strat = "abstract"
        if len({k: v for k, v in stages.items() if k in _STAGES}) < args.min_stages:
            im = imrad_sections(text)
            if im:
                stages.update(im)
                strat = "imrad" if len(im) >= 2 else strat
        if len({k for k in _STAGES if k in stages}) < args.min_stages:
            ph = phrases_fallback(text)
            for k, v in ph.items():
                stages.setdefault(k, v)
            if ph and strat == "none":
                strat = "phrases"
        stages = need_obs(text, stages)
        nst = len([k for k in _STAGES if k in stages])
        stage_hist[nst] += 1
        if nst >= 2:
            strat_count[strat] += 1
        if nst >= args.min_stages:
            kept_by_domain[dom].append(
                (d.get("pmid") or d.get("arxiv_id") or d.get("id", "?"), dict(stages)))
            strat_kept[strat] += 1

    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            n_total += 1
            add_sample(d)
            if args.mode == "sample" and n_total % 2000 == 0:
                if (seen_domains and
                    all(len(kept_by_domain.get(dom, [])) >= args.per_domain
                        for dom in seen_domains)):
                    print("[sample] cupo completo en todos los dominios vistos; cortando",
                          file=sys.stderr, flush=True)
                    break
            if args.mode == "full" and n_total % 200000 == 0:
                print(f"[{n_total} docs] kept={sum(len(v) for v in kept_by_domain.values())}",
                      file=sys.stderr, flush=True)

    if args.mode == "full":
        out_path = args.out or "data/skeleton/train_skeleton.jsonl"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        written = 0
        with open(out_path, "w") as fo:
            for dom in sorted(kept_by_domain):
                for pid, st in kept_by_domain[dom]:
                    fo.write(json.dumps({
                        "pid": pid, "domain": dom, "text": serialize(st),
                        "etapas": len([k for k in _STAGES if k in st]),
                    }) + "\n")
                    written += 1
        report_out = args.report or (args.out and args.out.rsplit(".", 1)[0] + "_report.json")
    else:
        written = sum(len(v) for v in kept_by_domain.values())
        report_out = args.report

    dge2 = sum(v for k, v in stage_hist.items() if k >= 2)
    dge3 = sum(v for k, v in stage_hist.items() if k >= 3)
    report = {
        "input": args.input, "mode": args.mode, "min_stages": args.min_stages,
        "seed": args.seed,
        "n_docs_vistos": n_total,
        "docs_ge2_etapas": dge2, "docs_ge3_etapas": dge3,
        "coverage_ge2": round(dge2 / max(n_total, 1), 4),
        "coverage_ge3": round(dge3 / max(n_total, 1), 4),
        "stage_histogram": dict(sorted(stage_hist.items())),
        "strategy_ge2": dict(strat_count),
        "strategy_kept_ge3": dict(strat_kept),
        "per_domain_kept": {k: len(v) for k, v in sorted(kept_by_domain.items())},
        "total_kept": written,
    }
    if report_out and report_out != "-":
        os.makedirs(os.path.dirname(os.path.abspath(report_out)), exist_ok=True)
        with open(report_out, "w") as fo:
            json.dump(report, fo, indent=1)
    json.dump(report, sys.stdout, indent=1)
    return 0 if written >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())