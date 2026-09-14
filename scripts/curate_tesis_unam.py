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

  # cosecha multi-worker (flota/w*/md) — incremental + vigilante:
  python3 curate_tesis_unam.py --md-dirs flota/w*/md \
      --meta-jsonl flota/slice*.jsonl \
      --out curada_v2 --loop 1800
"""
import argparse, glob, json, re, shutil, sys, time
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


def slugify_title(title):
    """Replica el slug de descargar_convertir.py para mapear md -> metadata."""
    return re.sub(r"[^\w\-]+", "_", (title or "")[:70]).strip("_")


def run_once(args):
    """Una pasada de curado; devuelve el dict de reporte."""
    # --- fuentes de md ---
    md_files = []
    if args.corrida:
        md_files += sorted(Path(args.corrida).glob("md/*.md"))
    for pat in (args.md_dirs or []):
        md_files += sorted(Path(p) for p in glob.glob(pat + "/*.md")
                           if Path(p).is_file())

    # --- mapas de metadata ---
    meta_uuid = {}   # uuid8 -> info (checkpoint estilo corrida: uuid presente)
    meta_slug = {}   # slug(titulo) -> info (slices: uuid=null hasta resolver)
    meta_files = list(args.meta_jsonl or [])
    if args.corrida:
        meta_files.append(str(Path(args.corrida) / "checkpoint.jsonl"))
    for mj in meta_files:
        mjp = Path(mj)
        if not mjp.exists():
            continue
        for l in mjp.read_text(errors="ignore").splitlines():
            try:
                d = json.loads(l)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            if d.get("uuid"):
                meta_uuid[d["uuid"][:8]] = d
            s = slugify_title(d.get("title", ""))
            if s:
                meta_slug.setdefault(s, d)

    out_sci = Path(args.out) / "science"
    out_lat = Path(args.out) / "latex"
    out_sci.mkdir(parents=True, exist_ok=True)
    out_lat.mkdir(parents=True, exist_ok=True)

    # --- incremental: entradas ya curadas se conservan sin reprocesar ---
    prev = {}
    mjson = Path(args.out) / "meta.jsonl"
    if mjson.exists():
        for l in mjson.read_text(errors="ignore").splitlines():
            try:
                d = json.loads(l)
                prev[d["file"]] = d
            except Exception:
                continue

    stats = Counter()
    meta_out = []
    for fp in md_files:
        stem = fp.stem
        uuid8, _, slug = stem.partition("_")
        sci_rel, lat_rel = f"science/{fp.name}", f"latex/{fp.name}"
        # skip incremental: salida y meta previas existen -> reutilizar
        if sci_rel in prev and (out_sci / fp.name).exists():
            meta_out.append(prev[sci_rel]); stats["grp_science"] += 1
            stats["science_kept"] += 1
            continue
        if lat_rel in prev and (out_lat / fp.name).exists():
            meta_out.append(prev[lat_rel]); stats["grp_latex"] += 1
            continue
        info = meta_uuid.get(uuid8) or meta_slug.get(slug) or {}
        raw = fp.read_text(errors="ignore")
        degree = info.get("degree", "")
        grp = classify(info, raw)
        stats[f"grp_{grp}"] += 1
        common = {"uuid": uuid8, "degree": degree,
                  "area": info.get("area", ""),
                  "disciplina": info.get("disciplina", ""),
                  "facultad": info.get("facultad", ""),
                  "title": info.get("title", ""),
                  "date": info.get("date", "")}
        if grp == "latex":
            shutil.copy2(fp, out_lat / fp.name)  # se guarda tal cual (latex)
            meta_out.append({**common, "group": "latex",
                             "file": lat_rel, "chars_raw": len(raw)})
            continue
        t = clean(raw)
        if len(t) < args.min_chars:
            stats["science_dropped_short"] += 1
            continue
        (out_sci / fp.name).write_text(t)
        meta_out.append({**common, "group": "science", "file": sci_rel,
                         "chars_raw": len(raw), "chars_clean": len(t)})
        stats["science_kept"] += 1

    mjson.write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in meta_out) + "\n")
    rep = {"n_md_seen": len(md_files),
           **dict(stats)}
    print(json.dumps(rep, indent=2), flush=True)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep, indent=2) + "\n")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrida", default=None, help="dir con checkpoint.jsonl + md/")
    ap.add_argument("--md-dirs", nargs="*", default=None,
                    help="globs de dirs md (p.ej. flota/w*/md); combinable con --corrida")
    ap.add_argument("--meta-jsonl", nargs="*", default=None,
                    help="jsonls de metadata (slices); el join es por slug del titulo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--min-chars", type=int, default=20000,
                    help="mínimo de chars limpios para conservar en science/")
    ap.add_argument("--loop", type=int, default=0,
                    help="si >0, re-corre el curado cada N segundos")
    args = ap.parse_args()
    if not args.corrida and not args.md_dirs:
        ap.error("se requiere --corrida o --md-dirs")
    if args.loop <= 0:
        run_once(args)
        return 0
    while True:
        rep = run_once(args)
        print(f"[loop] {rep.get('n_md_seen', 0)} md vistos; "
              f"re-curando en {args.loop}s", flush=True)
        time.sleep(args.loop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
