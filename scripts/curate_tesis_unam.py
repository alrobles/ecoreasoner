#!/usr/bin/env python3
"""curate_tesis_unam.py — curado del corpus de tesis doctorales UNAM (ES).

Pipeline (decisión 2026-09-13): SOLO CLASIFICAR + LIMPIAR + GUARDAR.
La traducción ES->EN se hace DESPUÉS (cluster: v4serve / gpt-oss dedicado),
cuando la descarga madure. En esta etapa el texto queda en español.

Dos productos:
  out/science/  — tesis de ciencia (bio/eco/mar/quím/biomed/sostenib/...):
                  markdown limpio (sin front-matter LFDA/Neevia/comités,
                  sin índice, sin bibliografía/apéndices) + meta jsonl.
  out/latex/    — tesis mates/física/computación o con alta densidad de
                  markup: se conservan COMO LATEX para entrenar después
                  (copia sin limpieza agresiva; solo registro).

Detalle útil: muchas tesis son "por artículos" -> capítulos ya en inglés.
El limpiador no los toca; la traducción posterior solo debe cubrir las
partes en español.

Uso:
  python3 curate_tesis_unam.py --corrida /beegfs/.../corrida_doct \
      --out /beegfs/.../tesis_curada --report /beegfs/.../curate_report.json
"""
import argparse, json, re, shutil, sys
from collections import Counter
from pathlib import Path

# --- clasificación por metadata (checkpoint.jsonl) ---
# `area` es un vocabulario controlado DGDU UNAM:
#   "Ciencias Biológicas, Químicas y de la Salud" -> science
#   "Ciencias Físico - Matemáticas y de las Ingenierías" -> latex (aparte)
# Fallback sobre `degree`/`disciplina` si area falta.
SCIENCE_RE = re.compile(
    r"biol[oó]gic|biom[eé]dic|bioqu[ií]mic|biolog[ií]a|"
    r"mar y limnolog|oceanograf|ecolog|ambiental|sostenibilidad|"
    r"qu[ií]mic|genomic|microbiolog|bot[aá]nica|zoolog|fisiolog|"
    r"neurocien|agronom|veterinar|biotecnol|inmunolog|farmaco|"
    r"ciencias de la tierra|geolog|geof[ií]s|geograf|salud", re.I)
LATEX_PROG_RE = re.compile(
    r"f[ií]sico[\s-]*matem[aá]tic|matem[aá]tic|f[ií]sica|computaci|"
    r"ingenier|estad[ií]stic|actuar|materiales|electr[oó]nic", re.I)

# densidad de matemáticas en el markdown YA convertido: los \comandos LaTeX
# se pierden al pasar a md; lo que sobrevive son $...$, $$..$$, \begin{...}
# y símbolos matemáticos unicode.
MATH_TOK_RE = re.compile(
    r"\$[^$\n]+\$|\$\$|\\begin\{|\\end\{|\\frac|\\sum|\\int|"
    r"[∀∃∈∉⊂⊆∪∩∂∇∑∏∫√∞≈≠≤≥⊗⊕→←↔⇒⇔α-ωΑ-Ω]")
LATEX_DENS_MAX = 8  # tokens matemáticos por KB (empírico sobre el md)

# --- limpieza ---
JUNK_LINE_RE = re.compile(
    r"neevia|docconverter|derechos reservados|prohibida su reproducci[oó]n|"
    r"ley federal del derecho de autor|lfda|comit[eé] tutoral|"
    r"director de tesis|comite de|jurado|acto de grado|"
    r"proyecto conacyt|conahcyt|registro de becario|"
    r"posgrado en ciencias|instituto de biolog[ií]a.{0,20}unam|"
    r"est[aá] protegido por la ley", re.I)

# primer título "real": capítulo/sección numerada o introducción
FIRST_HEAD_RE = re.compile(
    r"^#{1,2}\s+\**\s*(?:cap[ií]tulo\s+\d|\d+\s*[.\)]\s*\w|introducci[oó]n|"
    r"resumen|abstract|chapter\s+\d)",
    re.I | re.M)
# sección de tabla de contenido: su presencia indica que lo anterior y la propia
# tabla son front-matter
TOC_HEAD_RE = re.compile(
    r"^#{1,3}\s+\**\s*(?:[ií]ndice(?:\s+general|\s+de contenido|"
    r"\s+anal[ií]tico)?|contenido|contents|table of contents)\s*\**\s*$",
    re.I | re.M)
# cortes de cola: bibliografía/referencias/apéndices/agradecimientos finales
TAIL_CUT_RE = re.compile(
    r"^#{1,3}\s+\**\s*(?:\d+\s*[.\)]?\s*)?(?:bibliograf|referencias|"
    r"references|ap[eé]ndice|appendix|anexos|agradecimientos|"
    r"acknowledg|glosario|list of)",
    re.I | re.M)


def latex_density(text):
    kb = max(1, len(text) // 1024)
    return len(MATH_TOK_RE.findall(text)) / kb


def classify(info, text):
    """info: dict del checkpoint (area/degree/disciplina)."""
    area = (info.get("area") or "")
    deg = (info.get("degree") or "") + " " + (info.get("disciplina") or "")
    if SCIENCE_RE.search(area):
        return "science"
    if LATEX_PROG_RE.search(area):
        return "latex"
    if SCIENCE_RE.search(deg):
        return "science"
    if LATEX_PROG_RE.search(deg):
        return "latex"
    return "latex" if latex_density(text) > LATEX_DENS_MAX else "science"


def clean(text):
    """Devuelve markdown limpio o '' si queda irrecuperable/corto."""
    lines = []
    for l in text.splitlines():
        if JUNK_LINE_RE.search(l):
            continue
        lines.append(l)
    t = "\n".join(lines)
    # cortar front-matter: el conversor marca las ENTRADAS del índice como
    # encabezados -> no basta el primer match. Regla: el primer encabezado
    # real es aquel seguido de PROSA (>150 chars no-# hasta el proximo
    # encabezado). Se busca a partir del final de la sección Índice si existe.
    base = 0
    tm = TOC_HEAD_RE.search(t)
    if tm:
        base = tm.end()
    heads = list(re.finditer(r"^#{1,3} ", t, re.M))
    first_reales = [m for m in FIRST_HEAD_RE.finditer(t) if m.start() >= base]
    start = None
    for m in first_reales:
        nxt = next((h.start() for h in heads if h.start() > m.end()), len(t))
        block_lines = [l.strip() for l in t[m.end():nxt].splitlines()
                       if l.strip()]
        # prosa real: una linea larga (parrafo) o varias lineas densas;
        # las entradas del indice son cortas y enumeradas
        if (any(len(l) >= 200 for l in block_lines) or
                sum(len(l) >= 100 for l in block_lines) >= 3):
            start = m.start()
            break
    if start is None and first_reales:
        start = first_reales[0].start()
    if start:
        t = t[start:]
    # cortar cola: bibliografía/apéndices (primero que aparezca tras el 40%)
    for tm in TAIL_CUT_RE.finditer(t):
        if tm.start() > len(t) * 0.4:
            t = t[:tm.start()]
            break
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrida", required=True, help="dir con checkpoint.jsonl + md/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--min-chars", type=int, default=20000,
                    help="mínimo de chars limpios para conservar en science/")
    args = ap.parse_args()

    corrida = Path(args.corrida)
    meta = {}
    ck = corrida / "checkpoint.jsonl"
    if ck.exists():
        for l in ck.read_text(errors="ignore").splitlines():
            try:
                d = json.loads(l)
                # el filename usa los primeros 8 chars del uuid
                meta[d["uuid"][:8]] = d
            except Exception:
                continue

    out_sci = Path(args.out) / "science"
    out_lat = Path(args.out) / "latex"
    out_sci.mkdir(parents=True, exist_ok=True)
    out_lat.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    meta_out = []
    for fp in sorted((corrida / "md").glob("*.md")):
        uuid = fp.name.split("_", 1)[0]
        info = meta.get(uuid, {})
        raw = fp.read_text(errors="ignore")
        degree = info.get("degree", "")
        grp = classify(info, raw)
        stats[f"grp_{grp}"] += 1
        if grp == "latex":
            shutil.copy2(fp, out_lat / fp.name)  # se guarda tal cual (latex)
            meta_out.append({"uuid": uuid, "group": "latex", "degree": degree,
                             "area": info.get("area", ""),
                             "disciplina": info.get("disciplina", ""),
                             "title": info.get("title", ""),
                             "file": f"latex/{fp.name}", "chars_raw": len(raw)})
            continue
        t = clean(raw)
        if len(t) < args.min_chars:
            stats["science_dropped_short"] += 1
            continue
        (out_sci / fp.name).write_text(t)
        meta_out.append({"uuid": uuid, "group": "science", "degree": degree,
                         "area": info.get("area", ""),
                         "disciplina": info.get("disciplina", ""),
                         "title": info.get("title", ""),
                         "date": info.get("date", ""),
                         "file": f"science/{fp.name}",
                         "chars_raw": len(raw), "chars_clean": len(t)})
        stats["science_kept"] += 1

    (Path(args.out) / "meta.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in meta_out) + "\n")
    rep = {"corrida": str(corrida), "n_md": stats["grp_science"] + stats["grp_latex"],
           **dict(stats)}
    print(json.dumps(rep, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
