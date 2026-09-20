#!/usr/bin/env python3
# DEPRECATED 2026-09-20: dependia de un modelo de origen chino vetado
# (Qwen / Tencent HY-MT / LLaDA weights / DeepSeek). Los pesos fueron
# eliminados del cluster; este script ya no corre. El tokenizer LLaDA
# (vocab 126080) sigue vivo SOLO como tokenizer de nuestro modelo.
"""Traducción ES->EN por BATCH en GPU de cluster (HY-MT1.5), sin serve/túneles.

- Cada job slurm toma un shard: docs[i] con i % nshards == shard.
- Reutiliza la lógica de traducir_v4: para_chunks + lang_of (chunks ya-EN
  pasan intactos — tesis por artículos) + limpieza de boilerplate UNAM.
- Inferencia: transformers, bf16, chat template del model card
  ("Translate the following segment into English, without additional
  explanation."), sampling params recomendados (t=0.7, top_k=20, top_p=0.6,
  rep_pen=1.05). Batching con left-padding.
- Resume seguro: .en.md existente no vacío = hecho. Escritura atómica.
- Estado progreso en report.jsonl + stdout por doc.
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

# boilerplate UNAM que NO aporta (licencia DGDU, portadas, furniture)
LICENSE_RE = re.compile(
    r"(?is)^.*?(direcci[oó]n general de bibliotecas|general directorate of libraries|"
    r"uso y reproducci[oó]n|usage restrictions|all rights reserved\.?.*?reservados).*?$",
    re.M)
PAGENOISE_RE = re.compile(r"(?m)^\s*(\d{1,4}|\w{0,4}\d{1,4})\s*$")  # folios sueltos

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
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tokbudget", type=int, default=6000)
    ap.add_argument("--nopenalty", action="store_true")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(a.model)
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dt).cuda().eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

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

    rep = open(a.report, "a", encoding="utf-8") if a.report else None
    stats = {"done": 0, "fail": 0, "es": 0, "en": 0}
    t_start = time.time()
    out_toks = 0

    def tr_batch(texts):
        """Traduce chunks en micro-batches por presupuesto de tokens de prompt
        (los logits fp32 del prefill ~ posiciones x vocab x 4B dominan la VRAM)."""
        nonlocal out_toks
        prompts = [PROMPT.format(t) for t in texts]
        encs = [tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True, add_generation_prompt=False) for p in prompts]
        def run(idxs):
            nonlocal out_toks
            sub = [encs[i] for i in idxs]
            mx = max(len(s) for s in sub)
            pad = tok.pad_token_id
            ids = torch.tensor(
                [[pad] * (mx - len(s)) + s for s in sub],
                dtype=torch.long, device=model.device)
            mask = torch.tensor(
                [[0] * (mx - len(s)) + [1] * len(s) for s in sub],
                dtype=torch.long, device=model.device)
            mnt = int(max(len(texts[i]) for i in idxs) * 1.6 / 3.5) + 128
            kw = {} if a.nopenalty else {"repetition_penalty": 1.05}
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids, attention_mask=mask, max_new_tokens=mnt,
                    do_sample=True, temperature=0.7, top_k=20, top_p=0.6,
                    pad_token_id=pad, **kw)
            gen = out[:, mx:]
            for j, i in zip(range(len(idxs)), idxs):
                res[i] = tok.decode(gen[j], skip_special_tokens=True).strip()
            out_toks += int((gen != pad).sum())

        order = sorted(range(len(encs)), key=lambda i: len(encs[i]))
        res = [None] * len(texts)
        buf, buf_tok = [], 0
        for i in order:
            n = len(encs[i])
            if buf and (buf_tok + n > a.tokbudget or len(buf) >= a.batch):
                run(buf)
                buf, buf_tok = [], 0
            buf.append(i)
            buf_tok += n
        if buf:
            run(buf)
        return res

    for p, out in todo:
        base = os.path.basename(p)[:-3]
        t0 = time.time()
        try:
            texto = clean(open(p, encoding="utf-8").read())
            cs = para_chunks(texto, a.chunk)
            es_idx = [i for i, c in enumerate(cs) if lang_of(c) == "es"]
            outs = tr_batch([cs[i] for i in es_idx])
            res = dict(zip(es_idx, outs))
            parts = [res.get(i, cs[i]) for i in range(len(cs))]
            tmp = out + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n\n".join(parts))
            os.rename(tmp, out)
            stats["done"] += 1
            stats["es"] += len(es_idx)
            stats["en"] += len(cs) - len(es_idx)
            el = time.time() - t_start
            rate = out_toks / max(time.time() - t_start, 1)
            print(f"[ok] {base[:48]} {len(es_idx)}es/{len(cs)-len(es_idx)}en "
                  f"{el:.0f}s | agg {rate:.0f} tok/s", flush=True)
            if rep:
                rep.write(json.dumps({"doc": base, "chars": len(texto),
                                      "es_chunks": len(es_idx),
                                      "en_chunks": len(cs) - len(es_idx),
                                      "t": round(time.time() - t_start, 1)}) + "\n")
                rep.flush()
        except Exception as e:
            stats["fail"] += 1
            print(f"[fail] {base}: {type(e).__name__}: {e}", flush=True)
            if rep:
                rep.write(json.dumps({"doc": base, "err": str(e)[:200]}) + "\n")
                rep.flush()

    el = time.time() - t_start
    print(f"[fin] shard{a.shard} done={stats['done']} fail={stats['fail']} "
          f"out_toks={out_toks} {out_toks/max(el,1):.0f} tok/s en {el/3600:.2f}h",
          flush=True)

if __name__ == "__main__":
    main()
