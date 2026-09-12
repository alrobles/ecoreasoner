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
import argparse, glob, json, os, re, subprocess, sys, time, urllib.error, urllib.request

STAGE_RE = re.compile(r"\[(OBSERVACION|HIPOTESIS|PREDICCION|EVIDENCIA|CONCLUSION)\]")
OUT_MARK = re.compile(
    r"(?:^|\n)[ \t>*_\-•#]*[\[\(\*_]*"
    r"(HIP[OÓ]T[ÉE]SIS|HYPOTHESIS|PREDICCI[OÓ]N|PREDICTION)"
    r"[\]\)\*:.\-]+[ \t]*", re.I)

SYSTEM = """You complete scientific argument skeletons. Given OBSERVATION, EVIDENCE and CONCLUSION, write the two missing inferential stages.

Your reply must be EXACTLY two lines and nothing else, like this example:
[HIPOTESIS] Drought reduces soil water availability, which inhibits seed germination and seedling survival.
[PREDICCION] Irrigated plots will show higher recruitment of the species than unirrigated plots.

Rules: <=30 words each, plain English, no new entities, do not restate the conclusion, no preamble or explanation."""

USER_TMPL = """[OBSERVACION] {obs}
[EVIDENCIA] {evid}
[CONCLUSION] {conc}

Reply with the two lines [HIPOTESIS] and [PREDICCION] only."""


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


def call_teacher(url, obs, evid, conc, model, retries=3, timeout=240,
                 api_key=None, models=None, max_tokens=1200, pace=0.0,
                 _last=[0.0]):
    """Devuelve (texto, estado). estado: None=ok, "net"=red/serve caido,
    "rate"=429 (no reintentar en caliente: las llamadas fallidas tambien
    descuentan la cuota diaria de OpenRouter)."""
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(obs=obs, evid=evid, conc=conc)},
        ],
        "temperature": 0.3, "max_tokens": max_tokens,
    }
    if models:  # openrouter: fallback server-side entre variantes :free
        body["models"] = models
        body["reasoning"] = {"enabled": False}  # el reasoning consume
        # max_tokens antes de emitir content -> respuesta vacia/truncada
    else:
        body["model"] = model
        body["think"] = False  # param ollama-only
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-Title"] = "ecoreasoner-b2-aug"
    payload = json.dumps(body).encode()
    for _ in range(retries):
        if pace:
            dt = time.time() - _last[0]
            if dt < pace:
                time.sleep(pace - dt)
        try:
            req = urllib.request.Request(url, data=payload, headers=headers,
                                         method="POST")
            _last[0] = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                r = json.loads(resp.read().decode())
            return r["choices"][0]["message"]["content"], None
        except urllib.error.HTTPError as e:
            ebody = e.read()[:200]
            print(f"[aug-err] HTTP {e.code} {ebody!r}", flush=True)
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                return None, ("rate", ra)
            if e.code in (400, 401, 403, 404, 422):
                # 4xx permanente (config/modelo): reintentar no lo arregla y un
                # resubmit-loop quemaria cuota en puros rechazos -> "fatal"
                return None, ("fatal", e.code)
            time.sleep(3)
        except Exception as e:
            print(f"[aug-err] {type(e).__name__} {str(e)[:200]}", flush=True)
            time.sleep(3)
    return None, ("net", None)


def openrouter_quota(api_key):
    """GET /api/v1/key -> usage diario restante (best-effort)."""
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            d = json.loads(resp.read().decode())["data"]
        return d.get("usage_daily"), d.get("limit")
    except Exception:
        return None, None


def jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(1, len(sa | sb))


def parse_out(text):
    """Marcador + contenido en la misma línea; si la línea queda vacía,
    el contenido es el bloque siguiente hasta línea en blanco u otro marcador."""
    parts = {}
    ms = list(OUT_MARK.finditer(text))
    for i, m in enumerate(ms):
        k = "HIPOTESIS" if m.group(1).upper().startswith("H") else "PREDICCION"
        if k in parts:
            continue
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        seg = text[m.end():end]
        nl = seg.find("\n")
        same_line = (seg[:nl] if nl != -1 else seg).strip()
        if same_line:
            parts.setdefault(k, same_line)
        else:
            block = seg.strip().split("\n\n", 1)[0]
            lines = [l for l in block.splitlines() if not STAGE_RE.match(l.strip())]
            val = "\n".join(lines).strip()
            if val:
                parts.setdefault(k, val)
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
    ap.add_argument("--backend", default=os.environ.get("BACKEND", "ollama"),
                    choices=["ollama", "openrouter"])
    ap.add_argument("--models", default=os.environ.get("OR_MODELS", ""),
                    help="CSV de slugs openrouter (fallback server-side)")
    ap.add_argument("--reverse", action="store_true",
                    default=bool(int(os.environ.get("REVERSE", "0"))),
                    help="recorrer candidatos al reves (workers secundarios)")
    ap.add_argument("--pace", type=float,
                    default=float(os.environ.get("PACE", "0")),
                    help="seg minimos entre llamadas (rate limit client-side)")
    ap.add_argument("--max-tokens", type=int,
                    default=int(os.environ.get("MAX_TOKENS", "1200")))
    ap.add_argument("--claims", default=os.environ.get("CLAIMS", ""),
                    help="dir de claims compartidos entre workers (dedup)")
    ap.add_argument("--done-glob", default=os.environ.get("DONE_GLOB", ""),
                    help="glob de .done hermanos a respetar (dedup cross-worker)")
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
    if args.reverse:
        docs = docs[::-1]
    print(f"[aug] candidatos={len(docs)}", flush=True)

    done_path = args.out + ".done"
    done = set()
    for f in glob.glob(args.done_glob or done_path):
        if os.path.exists(f):
            done.update(open(f).read().split())
    f_out = open(args.out, "a")
    f_done = open(done_path, "a")

    claim_dir = args.claims or (args.out + ".claims")
    os.makedirs(claim_dir, exist_ok=True)

    if args.backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            kf = os.environ.get("OPENROUTER_KEY_FILE",
                                os.path.expanduser("~/.openrouter-key"))
            try:
                api_key = open(kf).read().strip()
            except OSError:
                pass
        if not api_key:
            print("[aug] FATAL: sin OPENROUTER_API_KEY ni key-file", flush=True)
            sys.exit(2)
        url = "https://openrouter.ai/api/v1/chat/completions"
        # OR: fallback array admite max 3 modelos (400 si no)
        models = [m.strip() for m in args.models.split(",") if m.strip()][:3]
        used, lim = openrouter_quota(api_key)
        print(f"[aug] openrouter usage_daily={used} limit={lim} "
              f"models={models}", flush=True)
    else:
        url = resolve_ollama_url()
        if not url:
            print("[aug] FATAL: no hay ollama-v4serve RUNNING", flush=True)
            sys.exit(2)
        api_key, models = None, None
    print(f"[aug] teacher={args.model} backend={args.backend} url={url}",
          flush=True)

    n_ok = n_fail = consec_net = n_rate = 0
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
        # claim atomico: otro worker (otro shard/backend) no lo repite
        cpath = os.path.join(claim_dir, pid.replace("/", "_"))
        try:
            fd = os.open(cpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue

        raw, status = call_teacher(url, obs, evid, conc, args.model,
                                   api_key=api_key, models=models,
                                   max_tokens=args.max_tokens, pace=args.pace)
        if status is not None:
            try:
                os.unlink(cpath)  # sin intento real: liberar para otro worker
            except OSError:
                pass
            kind, ra = status
            if kind == "fatal":
                # EXHAUSTED en el mensaje: el grep del slurm frena el resubmit
                print(f"[aug] CONFIG_EXHAUSTED: HTTP {ra} permanente — "
                      "fin sin resubmit", flush=True)
                sys.exit(3)
            if kind == "rate":
                n_rate += 1
                wait = min(float(ra), 300.0) if ra else 60.0
                print(f"[aug] 429 x{n_rate} retry-after={wait:.0f}s", flush=True)
                if n_rate >= 4:
                    # QUOTA_EXHAUSTED: el grep del slurm lo toma como fin
                    # limpio -> NO resubmit (reintentar quemaria cuota diaria)
                    print("[aug] QUOTA_EXHAUSTED 429 x4 -> fin sin resubmit",
                          flush=True)
                    sys.exit(3)
                time.sleep(wait)
                continue
            consec_net += 1
            if consec_net >= 5:
                print(f"[aug] FATAL: teacher inalcanzable x{consec_net}, "
                      "salir para resubmit (pids NO marcados)", flush=True)
                sys.exit(2)
            continue
        consec_net = n_rate = 0
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
