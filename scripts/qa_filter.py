"""qa_filter.py — valida pares QA crudos de qa_gen.py y emite el JSONL
de SFT con el formato establecido de build_sft_chat.py:

    {"prompt": "[USER] <q>\n[ASSISTANT]", "response": " <a>", "src": "qa-pdb"}

Filtros:
  1. schema: q/a no vacíos, longitudes acotadas (q<=400c, a 8..512c)
  2. groundedness: recall de content-words de la respuesta en el pasaje
     >= --min-recall (default 0.5). Las respuestas NO soportadas por el
     pasaje se descartan (anti-alucinación del teacher).
  3. números: todo número de la respuesta debe aparecer (normalizado) en
     el pasaje — protege el binding numérico que evalúa la battery.
  4. dedup por pregunta normalizada.

Entrada: qa_raw_shard*.jsonl {"pid","passage","qa":[{q,a}]}
Salida:  JSONL listo para sft_pretok.py.

Uso:
  python3 qa_filter.py --indir data/qa_raw --out data/qa_sft_v1.jsonl
"""
import argparse
import glob
import json
import os
import re

USER_M = "[USER]"
ASST_M = "[ASSISTANT]"
_STOP = set("""a an the of in on at to for and or but is are was were be been
it its this that these those with by from as we i you they he she not no do
does did have has had can could will would should may might than then so such
""".split())
_WORD = re.compile(r"[a-záéíóúñü][a-záéíóúñü\-']*", re.I)
_NUM = re.compile(r"\d[\d.,]*")


def content_words(text):
    return [w.lower() for w in _WORD.findall(text)
            if len(w) > 3 and w.lower() not in _STOP]


def norm_num(s):
    return s.replace(",", "").rstrip("0.")


def recall(answer, passage_set):
    aw = content_words(answer)
    if not aw:
        return 0.0
    return sum(1 for w in aw if w in passage_set) / len(aw)


def numbers_ok(answer, passage_nums):
    for n in _NUM.findall(answer):
        if norm_num(n) not in passage_nums:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="dir con qa_raw_shard*.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-recall", type=float, default=0.5)
    ap.add_argument("--max-q", type=int, default=400)
    ap.add_argument("--min-a", type=int, default=8)
    ap.add_argument("--max-a", type=int, default=512)
    a = ap.parse_args()

    seen_q = set()
    n_in = n_kept = n_dup = n_recall = n_num = n_len = 0
    files = sorted(glob.glob(os.path.join(a.indir, "qa_raw_shard*.jsonl")))
    if not files and os.path.isfile(a.indir):
        files = [a.indir]
    with open(a.out, "w", encoding="utf-8") as fout:
        for f in files:
            for line in open(f, encoding="utf-8"):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ptext = rec.get("passage", "")
                pset = set(content_words(ptext))
                pnums = {norm_num(n) for n in _NUM.findall(ptext)}
                for pair in rec.get("qa", []):
                    n_in += 1
                    q, an = pair["q"].strip(), pair["a"].strip()
                    if len(q) > a.max_q or not (a.min_a <= len(an) <= a.max_a):
                        n_len += 1
                        continue
                    key = " ".join(q.lower().split())
                    if key in seen_q:
                        n_dup += 1
                        continue
                    if recall(an, pset) < a.min_recall:
                        n_recall += 1
                        continue
                    if not numbers_ok(an, pnums):
                        n_num += 1
                        continue
                    seen_q.add(key)
                    fout.write(json.dumps({
                        "prompt": f"{USER_M} {q}\n{ASST_M}",
                        "response": f" {an}",
                        "src": "qa-pdb"}, ensure_ascii=False) + "\n")
                    n_kept += 1
    print(f"[filter] in={n_in} kept={n_kept} "
          f"drop: len={n_len} dup={n_dup} recall={n_recall} nums={n_num}")
    print(f"[filter] -> {a.out}")


if __name__ == "__main__":
    main()
