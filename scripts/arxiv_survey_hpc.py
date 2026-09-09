#!/usr/bin/env python3
"""arxiv_survey_hpc.py — barrido representativo de arXiv en HPC.

Recorre las categorias indicadas:
 - totalResults (volumen)
 - muestreo sistematico de N papers dispersos (inicio incremental) y scrape de licencia
   en /abs/{id}, reportando la distribucion estimada de licencias y DOI.

Respeta rate-limit de arXiv (~1 req/3.3s). Pensado para correr en background/HPC.

Uso:
  python3 arxiv_survey_hpc.py --categories "q-bio.PE q-bio.QM" --sample 120 --out FILE
"""
import argparse, json, re, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter

NS = {"a": "http://www.w3.org/2005/Atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/",
      "ax": "http://arxiv.org/schemas/atom"}
API = "https://export.arxiv.org/api/query?"
USER = {"User-Agent": "Mozilla/5.0 ecoreasoner-survey-hpc/1.0"}

def http_get(url, retries=5):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=USER)
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 8 * (a + 1); time.sleep(wait); continue
            return f"HTTP {e.code}"
        except Exception as e:
            time.sleep(4); continue
    return "429 persistent"

def q_total(cat):
    xml = http_get(API + urllib.parse.urlencode({"search_query": f"cat:{cat}", "max_results": 1}))
    try:
        root = ET.fromstring(xml)
        t = root.find("os:totalResults", NS)
        return int(t.text) if t is not None else None
    except Exception:
        return None

def q_batch(cat, start, n=50):
    xml = http_get(API + urllib.parse.urlencode({
        "search_query": f"cat:{cat}", "start": start, "max_results": n,
        "sortBy": "submittedDate", "sortOrder": "descending"}))
    out = []
    try:
        root = ET.fromstring(xml)
        for e in root.findall("a:entry", NS):
            aid = e.find("a:id", NS).text.strip().split("/abs/")[-1]
            title = (e.find("a:title", NS).text or "").strip().replace("\n", " ")
            published = (e.find("a:published", NS).text or "")[:10]
            d = e.find("ax:doi", NS)
            doi = d.text.strip() if d is not None and d.text else None
            out.append({"arxiv_id": aid, "title": title[:150],
                        "published": published, "doi": doi})
    except Exception:
        pass
    return out

def classify_url(u):
    if not u: return "NO-LICENCIA"
    if "creativecommons.org/licenses/" in u:
        m = re.search(r'licenses/([^/]+)/([^/"#]+)', u)
        return f"CC-{m.group(1)}-{m.group(2)}" if m else u
    if "arxiv.org" in u:
        return "arxiv-no-exclusiva"
    if "creativecommons.org/publicdomain" in u or "cc0" in u.lower():
        return "CC0"
    return u

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="q-bio.PE q-bio.QM")
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/arxiv_survey.jsonl")
    args = ap.parse_args()

    all_rows = []
    for cat in args.categories.split():
        total = q_total(cat)
        print(f"[{time.strftime('%H:%M:%S')}] {cat}: totalResults={total}", flush=True)
        time.sleep(3.3)
        N = total if isinstance(total, int) else args.sample
        step = max(1, N // args.sample)  # dispersion sistematica
        lic_counter = Counter()
        rows = []
        seen = set()
        for start in range(0, min(N, args.sample * step), step):
            batch = q_batch(cat, start, 1)
            time.sleep(3.3)
            if not batch: continue
            p = batch[0]
            if p["arxiv_id"] in seen: continue
            seen.add(p["arxiv_id"])
            # scrape licencia
            parts = p["arxiv_id"].split("/")[-1]
            html = http_get(f"https://arxiv.org/abs/{parts}")
            if html and not html.startswith(("HTTP", "ERR", "429")):
                m = re.search(r'abs-license[^>]*>\s*<a href="([^"]+)"', html)
                lu = m.group(1) if m else None
                dm = re.search(r'doi[^0-9]{0,30}(10\.\d{4,}/[^\s"<]+)', html)
                p["license_url"] = lu
                p["license"] = classify_url(lu)
                p["page_doi"] = dm.group(1) if dm else None
            else:
                p["license_url"] = None
                p["license"] = "FETCH_FAIL"
                p["page_doi"] = None
            lic_counter[p["license"]] += 1
            rows.append({**p, "category": cat})
            time.sleep(3.3)
        all_rows.extend(rows)
        print(f"[{time.strftime('%H:%M:%S')}] {cat}: muestra {len(rows)}, licencias:")
        for k, v in lic_counter.most_common():
            print(f"    {k}: {v}")
        print("", flush=True)

    with open(args.out, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"SAVED {len(all_rows)} -> {args.out}", flush=True)

if __name__ == "__main__":
    main()