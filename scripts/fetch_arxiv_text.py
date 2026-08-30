#!/usr/bin/env python3
r"""
fetch_arxiv_text.py — descarga el source LaTeX de arXiv (e-print gzip) por shard
y extrae texto plano. SBATCH array friendly (igual que fetch_pmc_shard).

Uso (slurm array):
  sbatch --array=0-7 fetch_arxiv_array.slurm
  -> cada tarea procesa SU slice de _arxiv_selected.jsonl

Escribe: data/arxiv/fulltext_c{id}.jsonl  (partial, se unen luego)
  {text, arxiv_id, title, abstract, license, year, domain}

Texto: source LaTeX (gzip o tar.gz) -> limpieza con stdlib (sin pandoc)
  - quita comentarios %, comandos \cmd{...} (conserva el interior),
    entornos equation/figure/table/algorithm/verbatim, math $..$,
    convierte \section/emph/textbf al texto interior, \n~ etc.
  - fallback: si el e-print viene como PDF (raros sin source) -> descartar.
"""
import argparse, gzip, io, json, os, re, sys, tarfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EPRINT = "https://export.arxiv.org/e-print/{arxiv_id}"

def strip_latex(src: str) -> str:
    s = src
    # comentarios fuera de \verb
    s = re.sub(r"(?<!\\)%.*", "", s)
    # verbatim / lstlisting / equation etc -> quitar contenido de algunos, texto de otros
    for env in ["figure", "table", "algorithm", "lstlisting", "verbatim",
                "tikzpicture", "align", "equation", "eqnarray", "gather",
                "multline", "itemize"]:
        s = re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (env, env), " ", s, flags=re.S)
    # citas/refs/labels
    s = re.sub(r"\\(cite|citep|citet|ref|label|eqref)\{[^}]*\}", " ", s)
    # comandos con {..} -> dejar el interior (textbf, emph, section, url...)
    s = re.sub(r"\\(bf|textbf|textit|emph|textit|texttt|section|subsection|subsubsection|"
               r"paragraph|title|author|affiliation|email|url|href)\*?\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\(item|itemsep|vspace|hspace|[a-zA-Z]+\*)", " ", s)
    # math $..$ y \[..\]
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"\\\[.*?\\\]", " ", s, flags=re.S)
    # escapes y chars
    s = s.replace("\\&", "&").replace("\\%", "%").replace("~", " ").replace("\\ ", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()

def get_source(arxiv_id: str, timeout=40) -> str:
    """Descarga el e-print (gzip o tar.gz) y devuelve el texto plano."""
    url = EPRINT.format(arxiv_id=arxiv_id)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
        # puede ser gzip de un .tex, o tar.gz con varios archivos
        if data[:2] == b"\x1f\x8b":
            try:
                raw = gzip.decompress(data)
            except Exception:
                return ""
            if raw[:5] == b"!<arc" or raw[:5] == b"<arx":  # overfull
                return ""
        else:
            raw = data
        text = ""
        if tarfile.is_tarfile(io.BytesIO(raw)):
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile(): continue
                    if m.name.endswith((".tex", ".txt")):
                        try:
                            t = tf.extractfile(m).read().decode("utf-8", "replace")
                        except Exception:
                            continue
                        if len(t) > len(text): text = t
        else:
            # .tex directo (gzip) o texto plano
            text = raw.decode("utf-8", "replace")
        return strip_latex(text)
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", default="/beegfs/a474r867/ecoreasoner/data/arxiv/_arxiv_selected.jsonl")
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--n_chunks", type=int, default=8)
    ap.add_argument("--out_dir", default="/beegfs/a474r867/ecoreasoner/data/arxiv/fulltext")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--min-text", type=int, default=1500, help="min chars del texto extraido")
    a = ap.parse_args()

    sel = [json.loads(l) for l in open(a.sel, encoding="utf-8") if l.strip()]
    total = len(sel)
    start = (total * a.chunk_id) // a.n_chunks
    end = (total * (a.chunk_id + 1)) // a.n_chunks
    mine = sel[start:end]
    print(f"[fetch-arxiv] chunk {a.chunk_id}: {len(mine)} papers (of {total}, slice {start}:{end})", flush=True)

    out_path = Path(a.out_dir) / f"fulltext_c{a.chunk_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); n_ok = n_short = n_fail = 0
    with open(out_path, "w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(get_source, d["id"]): d for d in mine}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                txt = fut.result()
            except Exception:
                txt = ""
            if txt and len(txt) > a.min_text:
                rec = {"text": txt, "arxiv_id": d["id"], "title": d.get("title",""),
                       "abstract": d.get("abstract",""), "license": d.get("license",""),
                       "year": d.get("year"), "domain": d.get("domain","")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
            elif txt:
                n_short += 1
            else:
                n_fail += 1
            if (n_ok+n_fail+n_short) % 500 == 0:
                print(f"  {n_ok+n_fail+n_short}/{len(mine)} (ok {n_ok} short {n_short} fail {n_fail}) {time.time()-t0:.0f}s", flush=True)
    print(f"[fetch-arxiv] chunk {a.chunk_id} DONE: ok={n_ok} short={n_short} fail={n_fail} -> {out_path}", flush=True)

if __name__ == "__main__":
    main()