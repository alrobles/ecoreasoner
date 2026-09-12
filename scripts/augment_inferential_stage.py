#!/usr/bin/env python3
"""
augment_inferential_stage.py — B2: rellenar la etapa inferencial del corpus.

Para cada esqueleto OBS→EVID→CONC (PREDICCION=0%, HIPOTESIS=2.4% en el corpus),
pide al teacher (DeepSeek-V4-Flash via ollama, resolución dinámica de nodo)
las dos etapas que faltan:

    [HIPOTESIS]  1 frase: hipótesis del mecanismo que conecta OBS->EVID.
    [PREDICCION] 1 frase: predicción derivada, específica y testeable.

Reconstruye el doc en orden canónico OBS->HIP->PRED->EVID->CONC.

Filtros (RSD: trazas cortas y simples para estudiantes pequeños):
  - <=60 palabras por etapa, sin etiquetas de etapa dentro del contenido
  - jaccard(contenido, CONCLUSION) < 0.7 (no parafrasear la conclusión)
  - debe producir AMBAS etapas o se descarta el doc aumentado

Resume: append a --out + sidecar .done (pids completados).

Uso:
  python3 augment_inferential_stage.py --input train_skeleton_train.jsonl \
      --n 40000 --out skeleton_aug_v1.jsonl --seed 731
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request

STAGE_RE = re.compile(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]")
OUT_RE = re.compile(
    r"(?:\[|\*\*)?\s*(HIP[OÓ]T[ÉE]SIS|HYPOTHESIS|PREDICCI[OÓ]N|PREDICTION)"
    r"[\]\*:]*\s*([^\n\[*]+)", re.I)

SYSTEM = """You complete scientific argument skeletons. Given OBSERVATION, EVIDENCE and CONCLUSION, write the two missing inferential stages.

Reply with EXACTLY two lines, like this example:
[HIPOTESIS] Drought reduces soil water availability, which inhibits seed germination and seedling survival.
[PREDICCION] Irrigated plots will show higher recruitment of the species than unirrigated plots.

Rules: <=30 words each, plain English, no new entities, do not restate the conclusion."""

USER_TMPL = """[OBSERVACION] {obs}
[EVIDENCIA] {evid}
[CONCLUSION] {conc}"""


def resolve_ollama_url():
    """Localiza el job ollama-v4serve RUNNING y extrae host:port del log."""
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "a474r867"),
             "-n", "ollama-v4serve", "-t", "R", "-h", "-o", "%i|%N"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        if not out:
            return None
        jobid, node = out.split("|")[0], out.split("|")[1].strip()
        workdir = os.path.expanduser("~/work/ollama")
        pf = os.path.join(workdir, f"v4flash-serve-output-{jobid}")
        if os.path.exists(pf):
            port = host = None
            for line in open(pf, errors="ignore"):
                if line.startswith("Port:"):
                    port = line.split(":", 1)[1].strip()
                elif line.startswith("Node:"):
                    host = line.split(":", 1)[1].strip()
            if port and host:
                return f"http://{host}:{port}/v1/chat/completions"
    except Exception:
        pass
    return None


def call_teacher(url, obs, evid, conc, model, retries=3, timeout=240):
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(obs=obs, evid=evid, conc=conc)},
        ],
        "temperature": 0.3, "max_tokens": 1200, "think": False,
    }).encode()
    last = "?"
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                r = json.loads(resp.read().decode())
            return r["choices"][0]["message"]["content"]
        except Exception as e:
            last = str(e)[:120]
            time.sleep(3)
    return None


def jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(1, len(sa | sb))


def parse_out(text):
    parts = {}
    for m in OUT_RE.finditer(text):
        k = "HIPOTESIS" if m.group(1).upper().startswith("H") else "PREDICCION"
        parts[k] = m.group(2).strip()
    return parts.get("HIPOTESIS"), parts.get("PREDICCION")


def valid(hip, pred, conc):
    if not hip or not pred:
        return False
    for seg in (hip, pred):
        if STAGE_RE.search(seg) or len(seg.split()) > 60 or len(seg.split()) < 4:
            return False
        if jaccard(seg, conc) >= 0.7:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=731)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--max-chars", type=int, default=1800,
                    help="truncar cada etapa del prompt a N chars")
    ap.add_argument("--every", type=int, default=1, help="procesar 1 de cada N docs")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    docs = []
    for i, line in enumerate(open(args.input)):
        if i % args.every:
            continue
        r = json.loads(line)
        if "HIPOTESIS" in r["text"] or "PREDICCION" in r["text"]:
            continue  # ya tiene la etapa (2.4%)
        m = list(STAGE_RE.finditer(r["text"]))
        if len(m) < 3:
            continue
        docs.append(r)
    rng.shuffle(docs)
    docs = [d for i, d in enumerate(docs) if i % args.nshards == args.shard]
    print(f"[aug] candidatos={len(docs)}", flush=True)

    done_path = args.out + ".done"
    done = set()
    if os.path.exists(done_path):
        done = set(open(done_path).read().split())
    f_out = open(args.out, "a")
    f_done = open(done_path, "a")

    url = resolve_ollama_url()
    if not url:
        print("[aug] FATAL: no hay ollama-v4serve RUNNING", flush=True)
        sys.exit(2)
    print(f"[aug] teacher={args.model} url={url}", flush=True)

    n_ok = n_fail = 0
    t0 = time.time()
    for k, r in enumerate(docs):
        pid = r.get("pid", f"doc{k}")
        if pid in done:
            continue
        segs = {}
        m = list(STAGE_RE.finditer(r["text"]))
        for j, mm in enumerate(m):
            end = m[j + 1].start() if j + 1 < len(m) else len(r["text"])
            segs[mm.group(1)] = r["text"][mm.end():end].strip()
        obs, evid, conc = (segs.get(s, "")[:args.max_chars]
                           for s in ("OBSERVACION", "EVIDENCIA", "CONCLUSION"))
        if not obs or not evid or not conc:
            f_done.write(pid + "\n"); f_done.flush()
            continue
        raw = call_teacher(url, obs, evid, conc, args.model)
        hip, pred = parse_out(raw or "")
        if valid(hip, pred, conc):
            new_text = (f"[OBSERVACION] {segs['OBSERVACION']}\n"
                        f"[HIPOTESIS] {hip}\n"
                        f"[PREDICCION] {pred}\n"
                        f"[EVIDENCIA] {segs['EVIDENCIA']}\n"
                        f"[CONCLUSION] {segs['CONCLUSION']}")
            f_out.write(json.dumps({**r, "text": new_text, "aug": "b2_v1"}) + "\n")
            f_out.flush()
            n_ok += 1
            if n_ok >= args.n:
                break
        else:
            n_fail += 1
            if n_fail <= 15:
                print(f"[aug-fail] pid={pid} raw={repr((raw or '')[:400])}",
                      flush=True)
        f_done.write(pid + "\n"); f_done.flush()
        if (k + 1) % 25 == 0:
            el = time.time() - t0
            rate = (k + 1) / el if el else 0
            eta = (len(docs) - k - 1) / rate / 3600 if rate else -1
            print(f"[aug] {k+1}/{len(docs)} ok={n_ok} fail={n_fail} "
                  f"{rate:.2f} docs/s eta={eta:.1f}h", flush=True)
    status = "COMPLETE" if n_ok >= args.n else "EXHAUSTED"
    print(f"[aug] {status} ok={n_ok} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
