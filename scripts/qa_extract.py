"""qa_extract.py — extrae pasajes legibles de corpus_v5_pdb.jsonl para
generación de QA con teacher.

El corpus papers_db es LaTeX crudo. Este script:
  1. quita comentarios/comandos LaTeX y preámbulo,
  2. parte en párrafos y filtra los que son mayormente markup,
  3. selecciona párrafos "claim-bearing" (números, verbos de hallazgo,
     conectivas) — el mismo criterio de rol que usa el masking,
  4. empaqueta 1-3 párrafos contiguos en ventanas ~280-520 palabras,
  5. emite JSONL {"pid": "doc_i.seg_i", "text": pasaje} — pid único por
     segmento (un doc puede emitir varios pasajes).

Salida alimenta qa_gen.py (vLLM). Resume-friendly: --start/--limit por doc.

Uso:
  python3 qa_extract.py --corpus data/corpus_v5_pdb.jsonl \
      --out data/qa_passages.jsonl --limit 200000 --per-doc 2
"""
import argparse
import json
import re
import sys

_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?")
_COMMENT = re.compile(r"(?<!\\)%.*")
_ENV = re.compile(r"\\(begin|end)\{[^{}]*\}")
_INLINE_MATH = re.compile(r"\$\$([^$]*)\$\$|\$([^$]*)\$")
_MULTISPACE = re.compile(r"\s+")
_URL = re.compile(r"\\?(url|href|cite|ref|label|eqref)\b[^{}]*\{[^{}]*\}?")
# Señales de claim: verbos de hallazgo, conectivas causales, números/unidades
_FIND = re.compile(
    r"\b(show|find|found|demonstrat|reveal|observ|report|suggest|indicat|"
    r"conclud|result|increase|decrease|correlat|affect|caus|reduc|predict|"
    r"significant|however|therefore|because|whereas|although)\w*",
    re.I)
_NUM = re.compile(r"\d")
_ALPHA = re.compile(r"[A-Za-z]")


def clean_latex(text):
    """Latex crudo -> texto plano legible (aprox; no perfecto pero basta
    para QA gen: el teacher tolera residuos menores)."""
    text = _COMMENT.sub(" ", text)
    # cortar preámbulo
    m = re.search(r"\\begin\{document\}", text)
    if m:
        text = text[m.end():]
    text = _ENV.sub(" ", text)
    # conservar contenido de math inline (números/unidades importan para QA);
    # solo se quitan los delimitadores $
    text = _INLINE_MATH.sub(lambda m: m.group(1) or m.group(2) or " ", text)
    text = _URL.sub(" ", text)
    text = _LATEX_CMD.sub(" ", text)
    text = text.replace("~", " ").replace("\\&", "&").replace("\\%", "%")
    return re.sub(r"[ \t]+", " ", text)   # preserva \n: separa párrafos


def paragraphs(text, min_words=30):
    for p in re.split(r"\n+", text):
        p = _MULTISPACE.sub(" ", p.strip())
        w = p.split()
        if len(w) < min_words or len(w) > 400:
            continue
        frac_alpha = sum(1 for c in p if c.isalpha()) / max(1, len(p))
        if frac_alpha < 0.55:          # mayormente markup/matemáticas
            continue
        yield p


def claim_score(p):
    return len(_FIND.findall(p)) + 1.5 * len(_NUM.findall(p[:200]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="docs a leer (0=todos)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--per-doc", type=int, default=2,
                    help="ventanas por doc")
    ap.add_argument("--min-score", type=float, default=2.0,
                    help="score mínimo de claim para retener un párrafo")
    a = ap.parse_args()

    n_doc = n_out = 0
    with open(a.corpus, encoding="utf-8", errors="replace") as fin, \
            open(a.out, "w", encoding="utf-8") as fout:
        for di, line in enumerate(fin):
            if di < a.start:
                continue
            if a.limit and di >= a.start + a.limit:
                break
            n_doc += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = clean_latex(rec.get("text") or rec.get("content") or "")
            paras = [p for p in paragraphs(text) if claim_score(p) >= a.min_score]
            # ventanas de 1-3 párrafos contiguos ~280-520 palabras
            made, i = 0, 0
            while made < a.per_doc and i < len(paras):
                win, j = [], i
                while j < len(paras) and sum(len(x.split()) for x in win) < 280:
                    win.append(paras[j]); j += 1
                if not win:
                    break
                fout.write(json.dumps({"pid": f"{di}.{made}",
                                       "text": " ".join(win)},
                                      ensure_ascii=False) + "\n")
                n_out += 1; made += 1; i = j + 1   # gap de 1 párrafo
            if n_doc % 20000 == 0:
                print(f"[extract] {n_doc} docs -> {n_out} pasajes", flush=True)
    print(f"[extract] DONE docs={n_doc} pasajes={n_out} -> {a.out}")


if __name__ == "__main__":
    main()
