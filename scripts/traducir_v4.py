#!/usr/bin/env python3
"""Traducción masiva ES->EN de curada_v2/science contra endpoints ollama (túneles).

Diseño:
- Chunking por párrafos (~CHUNK chars). Detección de idioma por chunk con
  stopwords: los chunks ya en inglés pasan intactos (tesis por artículos).
- Un único pool de hilos (--workers) despacha chunks de una ola de docs
  (--wave), round-robin entre endpoints. Concurrencia real = workers.
- Resume seguro: .en.md existente y no vacío = hecho. Escritura atómica
  (tmp + rename). Chunk que falla tras retry => doc no se escribe
  (reintentable en la próxima corrida).
- Orden: docs de menor a mayor tamaño (maximiza #docs terminados por ventana).
"""
import argparse, concurrent.futures as cf, glob, json, os, re, sys, threading, time, urllib.request

ES_STOP = set("""el la los las de del que en y a un una es son fue fueron para con por se su sus al lo
como más pero o no sí este esta estos estas entre sobre también desde donde cuando porque hasta hay
sin durante mediante cada tanto muy ya ser está están han sido tiene tienen puede pueden así mismo
misma otros otras parte años forma caso ellos ellas ni ante bajo según hacia debe deben presenta
presentan muestra muestran resultados obtenidos estudio especie especies muestras análisis
datos nivel niveles diferentes significativo significativa mayor menor aumento disminución efecto
efectos grupo grupos tratamiento control figura tabla capítulo objetivos metodología discusión
conclusiones introducción revisión material métodos""".split())

EN_STOP = set("""the of and in to was were is are for with on by as an be has have had it its that from
at or this these those between over also where when because until there without during each such very
already been can may thus same other others part years form case we our they their not no but which
who whom whose what all any both more most some only own than then so if will would could should
results obtained study species samples analysis data level levels different significant higher lower
increase decrease effect effects group groups treatment figure table chapter objectives methodology
discussion conclusions introduction review material methods however therefore furthermore""".split())

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")

def lang_of(text):
    toks = WORD_RE.findall(text.lower())
    if not toks:
        return "en"
    es = sum(1 for t in toks if t in ES_STOP)
    en = sum(1 for t in toks if t in EN_STOP)
    es += 2 * len(re.findall(r"[¿¡ñ]", text))  # caracteres inequívocos
    return "es" if es > en else "en"

def para_chunks(text, target):
    paras = re.split(r"\n\s*\n", text)
    out, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) + 2 > target:
            out.append(buf)
            buf = p
        else:
            buf = p if not buf else buf + "\n\n" + p
        while len(buf) > target:  # párrafo gigante: corte duro por línea
            cut = buf.rfind("\n", 0, target)
            if cut < target // 2:
                cut = target
            out.append(buf[:cut])
            buf = buf[cut:]
    if buf.strip():
        out.append(buf)
    return out

SYS = ("You are a scientific translator Spanish->English. Translate into natural academic English. "
       "Preserve the Markdown structure exactly (headers, lists, tables, citations, math), scientific "
       "names (italics), numbers, and references. Do NOT add comments or explanations. "
       "Return ONLY the translation of the given text.")

class Ep:
    def __init__(self, base, model, key=None):
        self.base, self.model, self.key = base, model, key
        self.is_or = base.startswith("https://openrouter.ai")
        self.consec_fail = 0
        self.dead_until = 0.0

class Pool:
    def __init__(self, spec, key_file=None):
        # spec: "url1=model1,url2=model2,or:model3" — modelo por endpoint
        # (un Q6000 no puede servir el 284B; cada serve anuncia el suyo).
        # "or:<model>" = OpenRouter (api/v1/chat/completions, key por archivo).
        # Repetir el mismo endpoint N veces lo pondera xN en el round-robin.
        self.eps = []
        key = None
        for item in spec.split(","):
            item = item.strip()
            if item.startswith("or:"):
                if key is None:
                    key = open(os.path.expanduser(key_file)).read().strip()
                self.eps.append(Ep("https://openrouter.ai/api/v1/chat/completions",
                                   item[3:], key))
            else:
                b, m = item.rsplit("=", 1)
                self.eps.append(Ep(b.strip(), m.strip()))
        self._i = 0
        self._lock = threading.Lock()
        self.toks = 0

    def _pick(self):
        # round-robin entre endpoints vivos; un endpoint con >=4 fallos
        # seguidos se aparca 10 min (murió el job slurm -> no quemar retries)
        with self._lock:
            now = time.time()
            live = [e for e in self.eps if e.dead_until < now]
            if not live:
                live = self.eps  # todos "muertos": reintentar igual
            ep = live[self._i % len(live)]
            self._i += 1
            return ep

    def _report(self, ep, ok):
        with self._lock:
            if ok:
                ep.consec_fail = 0
            else:
                ep.consec_fail += 1
                if ep.consec_fail >= 4:
                    ep.dead_until = time.time() + 600
                    print(f"[pool] {ep.base} apartado 10min ({ep.consec_fail} fallos)", flush=True)

    def chat(self, text, max_tokens):
        ep = self._pick()
        if ep.is_or:
            # OpenRouter: reasoning OFF (los reasoners queman max_tokens
            # sin emitir content — lección B2). usage.completion_tokens.
            body = json.dumps({
                "model": ep.model,
                "messages": [{"role": "system", "content": SYS},
                             {"role": "user", "content": text}],
                "max_tokens": max_tokens, "temperature": 0.1,
                "reasoning": {"enabled": False},
            }).encode()
            req = urllib.request.Request(ep.base, data=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {ep.key}",
            })
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    d = json.loads(r.read())
            except Exception:
                self._report(ep, False)
                raise
            self._report(ep, True)
            with self._lock:
                self.toks += d.get("usage", {}).get("completion_tokens", 0)
            return d["choices"][0]["message"]["content"]
        body = json.dumps({
            "model": ep.model,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": text}],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.1},
        }).encode()
        req = urllib.request.Request(f"{ep.base}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
        except Exception:
            self._report(ep, False)
            raise
        self._report(ep, True)
        with self._lock:
            self.toks += d.get("eval_count", 0)
        return d["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", required=True,
                    help="url1=model1,url2=model2,or:model3 (repetir pondera)")
    ap.add_argument("--or-key-file", default="~/.openrouter-key")
    ap.add_argument("--indir", default="curada_v2/science")
    ap.add_argument("--outdir", default="curada_v2/md_en")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--wave", type=int, default=24, help="docs por ola")
    ap.add_argument("--reverse", action="store_true",
                    help="orden descendente (para 2o worker anti-colisión)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", default="curada_v2/translate_report.jsonl")
    a = ap.parse_args()

    pool = Pool(a.endpoints, a.or_key_file)
    os.makedirs(a.outdir, exist_ok=True)

    docs = sorted(glob.glob(os.path.join(a.indir, "*.md")), key=os.path.getsize,
                  reverse=a.reverse)
    todo = []
    for p in docs:
        out = os.path.join(a.outdir, os.path.basename(p)[:-3] + ".en.md")
        if not (os.path.exists(out) and os.path.getsize(out) > 100):
            todo.append((p, out))
    if a.limit:
        todo = todo[: a.limit]
    print(f"[init] {len(docs)} docs, {len(todo)} pendientes -> {a.outdir}", flush=True)

    rep = open(a.report, "a", encoding="utf-8")
    stats = {"done": 0, "fail": 0, "es": 0, "en": 0}
    t_start = time.time()
    ex = cf.ThreadPoolExecutor(max_workers=a.workers)
    write_lock = threading.Lock()

    def tr_chunk(text):
        err = ""
        for k in range(2):
            try:
                mt = max(1024, int(len(text) * 1.5 / 3.5))
                return pool.chat(text, mt), None
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                time.sleep(2 * (k + 1))
        return None, err

    def finish_doc(st):
        # st = dict(path,out,cs,res,remaining,failed,n); escribe si completo
        base = os.path.basename(st["path"])[:-3]
        rec = {"doc": base, "chars": st["n"], "es_chunks": st["nes"],
               "en_chunks": len(st["cs"]) - st["nes"],
               "t": round(time.time() - t_start, 1)}
        if st["failed"]:
            stats["fail"] += 1
            rec["err"] = "chunk_failed"
        else:
            parts = [st["res"].get(i, st["cs"][i]) for i in range(len(st["cs"]))]
            tmp = st["out"] + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fo:
                fo.write("\n\n".join(parts))
            os.rename(tmp, st["out"])
            stats["done"] += 1
            stats["es"] += st["nes"]
            stats["en"] += len(st["cs"]) - st["nes"]
        rep.write(json.dumps(rec) + "\n")
        rep.flush()

    pending = a.wave * 4  # docs con plan abierto simultáneo
    idx = 0
    open_docs = {}

    def submit_next():
        nonlocal idx
        if idx >= len(todo):
            return False
        p, out = todo[idx]
        idx += 1
        if os.path.exists(out) and os.path.getsize(out) > 100:
            return True  # otro worker lo terminó desde el arranque
        texto = open(p, encoding="utf-8").read()
        cs = para_chunks(texto, a.chunk)
        es_idx = [i for i, c in enumerate(cs) if lang_of(c) == "es"]
        st = {"path": p, "out": out, "cs": cs, "res": {}, "nes": len(es_idx),
              "remaining": len(es_idx), "failed": False, "n": len(texto)}
        open_docs[id(st)] = st
        if not es_idx:
            with write_lock:
                finish_doc(st)
            del open_docs[id(st)]
            return True
        for i in es_idx:
            f = ex.submit(tr_chunk, cs[i])
            f.add_done_callback(lambda fut, st=st, i=i: on_chunk(fut, st, i))
        return True

    def on_chunk(fut, st, i):
        r, err = fut.result()
        with write_lock:
            if err:
                st["failed"] = True
                base = os.path.basename(st["path"])[:-3]
                print(f"[fail] {base} chunk{i}: {err}", flush=True)
            else:
                st["res"][i] = r
            st["remaining"] -= 1
            if st["remaining"] == 0:
                finish_doc(st)
                del open_docs[id(st)]

    # siembra inicial
    for _ in range(min(pending, len(todo))):
        submit_next()
    # mantener el embudo lleno mientras haya docs abiertos
    while open_docs:
        time.sleep(2)
        while len(open_docs) < pending and idx < len(todo):
            submit_next()
        el = time.time() - t_start
        rate = pool.toks / el if el else 0
        done, fail = stats["done"], stats["fail"]
        eta_h = (len(todo) - done - fail) / max(done, 1) * el / 3600
        print(f"[prog] {done}+{fail}f/{len(todo)} open={len(open_docs)} | "
              f"{rate:.0f} tok/s agg | ETA {eta_h:.1f}h", flush=True)

    el = time.time() - t_start
    print(f"[fin] done={stats['done']} fail={stats['fail']} es={stats['es']} "
          f"en_passthrough={stats['en']} tok={pool.toks} en {el/3600:.2f}h", flush=True)

if __name__ == "__main__":
    main()
