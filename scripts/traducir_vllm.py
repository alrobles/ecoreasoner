#!/usr/bin/env python3
"""Traducción ES->EN por vLLM (HY-MT1.5) — batching continuo en GPU de cluster.

Misma lógica de corpus que traducir_batch.py (para_chunks + lang_of + clean +
resume por .en.md) pero usando vllm.LLM offline: todos los chunks ES de todos
los docs pendientes del shard se meten a la cola de una vez y vLLM los agenda.
"""
import argparse, glob, json, os, re, sys, time

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
    es += 2 * len(re.findall(r"[¿¡ñ]", text))
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
        while len(buf) > target:
            cut = buf.rfind("\n", 0, target)
            if cut < target // 2:
                cut = target
            out.append(buf[:cut])
            buf = buf[cut:]
    if buf.strip():
        out.append(buf)
    return out

LICENSE_RE = re.compile(
    r"(?is)^.*?(direcci[oó]n general de bibliotecas|general directorate of libraries|"
    r"uso y reproducci[oó]n|usage restrictions|all rights reserved\.?.*?reservados).*?$",
    re.M)
PAGENOISE_RE = re.compile(r"(?m)^\s*(\d{1,4}|\w{0,4}\d{1,4})\s*$")

def clean(text):
    text = LICENSE_RE.sub("", text)
    text = PAGENOISE_RE.sub("", text)
    return text

PROMPT = ("Translate the following segment into English, without additional "
          "explanation.\n\n{}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--model", default="tencent/HY-MT1.5-7B")
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--maxlen", type=int, default=8192)
    ap.add_argument("--gmu", type=float, default=0.92)
    ap.add_argument("--maxdocs", type=int, default=64,
                    help="docs por oleada de generate() (controla RAM/host)")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    os.makedirs(a.outdir, exist_ok=True)
    docs = sorted(glob.glob(os.path.join(a.indir, "*.md")), key=os.path.getsize)
    mine = [p for i, p in enumerate(docs) if i % a.nshards == a.shard]
    todo = []
    for p in mine:
        out = os.path.join(a.outdir, os.path.basename(p)[:-3] + ".en.md")
        if not (os.path.exists(out) and os.path.getsize(out) > 100):
            todo.append((p, out))
    if a.limit:
        todo = todo[: a.limit]
    print(f"[init] shard {a.shard}/{a.nshards}: {len(mine)} docs míos, "
          f"{len(todo)} pendientes", flush=True)
    if not todo:
        return

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.maxlen,
              gpu_memory_utilization=a.gmu, enforce_eager=False)
    tok = llm.get_tokenizer()

    rep = open(a.report, "a", encoding="utf-8") if a.report else None
    t_start = time.time()
    out_toks = 0
    ndone = nfail = 0

    for w0 in range(0, len(todo), a.maxdocs):
        wave = todo[w0: w0 + a.maxdocs]
        # construir prompts de todos los chunks ES de la ola
        jobs = []  # (doc_i, chunk_i, n_prompt_tokens, max_new)
        texts = {}  # doc_i -> chunks
        for di, (p, _) in enumerate(wave):
            try:
                texto = clean(open(p, encoding="utf-8").read())
            except Exception as e:
                print(f"[fail-read] {os.path.basename(p)}: {e}", flush=True)
                continue
            cs = para_chunks(texto, a.chunk)
            texts[di] = cs
            for ci, c in enumerate(cs):
                if lang_of(c) == "es":
                    jobs.append((di, ci, c))
        if jobs:
            prompts = [[{"role": "user", "content": PROMPT.format(c)}]
                       for _, _, c in jobs]
            lens = [len(tok.encode(PROMPT.format(c))) for _, _, c in jobs]
            sp = [SamplingParams(temperature=0.7, top_k=20, top_p=0.6,
                                 repetition_penalty=1.05,
                                 max_tokens=min(int(n * 1.7 / 3.5) + 128, 6000))
                  for n in lens]
            t0 = time.time()
            outs = llm.chat(prompts, sp)
            el = time.time() - t0
            for (di, ci, _), o in zip(jobs, outs):
                n = len(o.outputs[0].token_ids)
                out_toks += n
                texts[di][ci] = o.outputs[0].text.strip()
            print(f"[wave] {len(jobs)} chunks en {el:.0f}s "
                  f"({out_toks / max(time.time() - t_start, 1):.0f} tok/s agg)",
                  flush=True)
        for di, (p, out) in enumerate(wave):
            base = os.path.basename(p)[:-3]
            if di not in texts:
                nfail += 1
                continue
            try:
                tmp = out + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(texts[di]))
                os.rename(tmp, out)
                ndone += 1
                print(f"[ok] {base[:48]} ({ndone}/{len(todo)})", flush=True)
                if rep:
                    rep.write(json.dumps({"doc": base, "shard": a.shard}) + "\n")
                    rep.flush()
            except Exception as e:
                nfail += 1
                print(f"[fail-w] {base}: {e}", flush=True)

    el = time.time() - t_start
    print(f"[fin] shard{a.shard} done={ndone} fail={nfail} "
          f"out_toks={out_toks} {out_toks / max(el, 1):.0f} tok/s "
          f"en {el / 3600:.2f}h", flush=True)

if __name__ == "__main__":
    main()
