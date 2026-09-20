"""qa_gen_mt.py — variante MULTI-TURN de qa_gen.py.

Genera un dialogo de N turnos por pasaje: cada turno posterior depende
de las respuestas anteriores (follow-ups), para SFT multi-turn con
spans de respuesta multiples (sft_pretok emite resp_ends/span_counts).

Salida: qa_mt_shard{K}.jsonl {"pid","passage","turns":[{"q","a"}...]}
El filtrado reusa qa_filter.py por turno (groundedness por respuesta).

Uso: igual que qa_gen.py pero --k = numero de turnos (default 3).
"""
import argparse
import glob
import json
import os
import re
import time

PROMPT_TMPL = """You are given a passage from a scientific paper. Write a {k}-turn dialogue that tests comprehension of THIS passage only.

Requirements:
- Turn 1: a question answerable from the passage.
- Turns 2..{k}: follow-up questions that DEPEND on the previous turns — they must use "it", "that value", "the method you mentioned", etc., or otherwise require the dialogue context to make sense.
- Every answer must be fully supported by the passage — no outside knowledge.
- Answers: 1-3 concise sentences.
- Output STRICT JSON only, no other text, no markdown fences.

Example of the required output format:
[{{"q": "What mechanism mediates the observed warming?", "a": "It is mediated by increased latent heat flux."}}, {{"q": "By how much did that flux increase?", "a": "It increased by 12 W/m2 over the study period."}}]

Passage:
\"\"\"
{passage}
\"\"\"

JSON array:"""

_JSON_BLOCK = re.compile(r"\[.*\]", re.S)


def load_passages(indir, shard, nshards, limit):
    """Lee passages.jsonl (o dir de shards) y toma la rebanada del shard."""
    if os.path.isdir(indir):
        files = sorted(glob.glob(os.path.join(indir, "*.jsonl")))
        recs = []
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for l in fh:
                    recs.append(l)
    else:
        with open(indir, encoding="utf-8") as fh:
            recs = fh.readlines()
    mine = [l for i, l in enumerate(recs) if i % nshards == shard]
    if limit:
        mine = mine[:limit]
    out = []
    for l in mine:
        try:
            r = json.loads(l)
            if r.get("text"):
                out.append(r)
        except json.JSONDecodeError:
            continue
    return out


def parse_qa(raw):
    """Extrae el primer bloque JSON [...] y valida schema mínimo q/a."""
    m = _JSON_BLOCK.search(raw)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    qa = []
    for it in arr if isinstance(arr, list) else []:
        q = str(it.get("q", "")).strip() if isinstance(it, dict) else ""
        a_ = str(it.get("a", "")).strip() if isinstance(it, dict) else ""
        if len(q) >= 10 and len(a_) >= 5:
            qa.append({"q": q, "a": a_})
    return qa or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="passages.jsonl o dir")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--model", default="allenai/OLMo-2-1124-13B-Instruct")
    ap.add_argument("--k", type=int, default=3, help="turnos por dialogo")
    ap.add_argument("--maxdocs", type=int, default=256,
                    help="pasajes por oleada de generate()")
    ap.add_argument("--maxlen", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--gmu", type=float, default=0.90)
    ap.add_argument("--dtype", default="bfloat16",
                    help="bfloat16 (cc>=8.0) | float16 (Turing q8000)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    out_path = os.path.join(a.outdir, f"qa_mt_shard{a.shard}.jsonl")
    recs = load_passages(a.indir, a.shard, a.nshards, a.limit)
    # resume: saltar pids ya escritos
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for l in fh:
                try:
                    done.add(json.loads(l)["pid"])
                except Exception:
                    pass
    todo = [r for r in recs if r["pid"] not in done]
    print(f"[gen] shard {a.shard}/{a.nshards}: {len(recs)} pasajes, "
          f"{len(done)} ya hechos, {len(todo)} pendientes", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=a.maxlen,
              gpu_memory_utilization=a.gmu)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=a.temp, top_p=0.9,
                        max_tokens=a.max_new)

    fout = open(out_path, "a", encoding="utf-8")
    t0 = time.time()
    ndone = nqa = 0
    for w0 in range(0, len(todo), a.maxdocs):
        wave = todo[w0: w0 + a.maxdocs]
        prompts = []
        for r in wave:
            msgs = [{"role": "user", "content":
                     PROMPT_TMPL.format(k=a.k, passage=r["text"][:6000])}]
            prompts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        outs = llm.generate(prompts, sp)
        for r, o in zip(wave, outs):
            qa = parse_qa(o.outputs[0].text)
            if qa:
                fout.write(json.dumps(
                    {"pid": r["pid"], "passage": r["text"], "turns": qa},
                    ensure_ascii=False) + "\n")
                nqa += len(qa)
            ndone += 1
        fout.flush()
        dt = time.time() - t0
        print(f"[gen] {ndone}/{len(todo)} pasajes, {nqa} pares, "
              f"{dt:.0f}s ({ndone/dt:.1f} pas/s)", flush=True)
    print(f"[gen] DONE shard {a.shard}: {ndone} pasajes -> {nqa} pares")


if __name__ == "__main__":
    main()
