#!/usr/bin/env python3
"""gen_distill_data.py — genera datos de destilación para el dLLM-MoE agentic.

Toma los prompts canónicos de tool-call + los expande con variantes, y para cada
uno hace UN loop cerrado con el teacher (v4-flash): el teacher decide las tool
calls, nosotros devolvemos un tool-result simulado (JSON), y el teacher da la
respuesta final. Guardamos la trayectoria completa en formato listo para
entrenamiento (secuencia de mensajes con tool-calls).

Output: distill_data.jsonl
  {"prompt": str, "trajectory": [{role, content, tool_calls?}, ...], "model": str}

Uso: python3 gen_distill_data.py [--endpoint http://127.0.0.1:20006]
      [--prompts data/prompts_toolcall_canonical.jsonl] [--out [...]]
      [--variants N] [--max_rounds 3]
"""
import argparse, json, time, urllib.request, random, sys, re

DEFAULT_TOOLS = [
    {"type": "function", "function": {"name": "gbif_occurrence",
     "description": "Busca registros de presencia de una especie en GBIF (occurrence search).",
     "parameters": {"type": "object", "properties": {"species": {"type": "string"},
        "region": {"type": "string"}}, "required": ["species"]}}},
    {"type": "function", "function": {"name": "bioclim_download",
     "description": "Descarga variables climaticas Bioclim/ERA5 para una region y periodo.",
     "parameters": {"type": "object", "properties": {"region": {"type": "string"},
        "year": {"type": "string"}}, "required": ["region"]}}},
    {"type": "function", "function": {"name": "maxent_train",
     "description": "Entrena un modelo de distribucion de especies MaxEnt dado ocurrencias y variables.",
     "parameters": {"type": "object", "properties": {"species": {"type": "string"},
        "layers": {"type": "string"}}, "required": ["species"]}}},
]

SYSTEM = ("Eres un agente cientifico ecologo que razona paso a paso y usa "
          "herramientas. Cuando necesites datos, llama a la funcion adecuada "
          "con argumentos JSON validos. Al final da una conclusion cientifica "
          "breve. Responde en espanol.")

def call_teacher(endpoint, messages, tools):
    payload = {
        "model": "deepseek-v4-flash:latest",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "stream": False,
    }
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def simulate_tool_result(tc):
    """Devuelve un resultado simulado para una tool_call (cerrado, sin ejecutar nada)."""
    name = tc.get("function", {}).get("name", "tool")
    args = tc.get("function", {}).get("arguments", "{}")
    try:
        a = json.loads(args); sp = a.get("species") or a.get("region") or "especie"
    except Exception:
        sp = "especie"
    content = {
        "gbif_occurrence": f"GBIF: 1,024 registros de {sp} (lat,lon) listos.",
        "bioclim_download": f"Bioclim/ERA5 descargado para {sp} (19 vars).",
        "maxent_train": f"MaxEnt entrenado para {sp} (AUC 0.81).",
    }.get(name, f"Resultado simulado de {name}: OK")
    return content

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:20006/v1/chat/completions")
    ap.add_argument("--prompts", default="/beegfs/a474r867/ecoreasoner/data/prompts_toolcall_canonical.jsonl")
    ap.add_argument("--out", default="/beegfs/a474r867/ecoreasoner/data/distill_data.jsonl")
    ap.add_argument("--variants", type=int, default=2, help="variaciones por prompt")
    ap.add_argument("--max_rounds", type=int, default=3)
    args = ap.parse_args()

    prompts = [json.loads(l)["prompt"] for l in open(args.prompts) if l.strip()]
    print(f"[{time.strftime('%H:%M:%S')}] {len(prompts)} prompts base; {args.variants} variantes c/u", flush=True)

    seen = set()
    count = 0
    with open(args.out, "w") as f:
        for i, p in enumerate(prompts):
            for v in range(args.variants):
                # expansion ligera: aniadir contexto/region aleatoria si no existe
                query = p
                if v > 0:
                    query = f"{p} Usa como referencia la region {'neotropico' if v==1 else 'paleartico'} y explica el metodo."
                if query in seen:
                    continue
                seen.add(query)
                # loop cerrado
                messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
                trajectory = [{"role": "user", "content": query}]
                final = ""
                try:
                    for rnd in range(args.max_rounds):
                        out = call_teacher(args.endpoint, messages, DEFAULT_TOOLS)
                        msg = out["choices"][0]["message"]
                        content = msg.get("content") or ""
                        tool_calls = msg.get("tool_calls") or []
                        step = {"role": "assistant", "content": content, "tool_calls": tool_calls}
                        trajectory.append(step)
                        if not tool_calls:
                            final = content; break
                        # process tool calls, append tool results
                        for tc in tool_calls:
                            tr = simulate_tool_result(tc)
                            messages.append({"role": "assistant", "content": content, "tool_calls": [tc]})
                            messages.append({"role": "tool", "tool_call_id": tc.get("id","t"), "content": tr})
                            trajectory.append({"role": "tool", "tool_call_id": tc.get("id","t"), "content": tr})
                        if rnd == args.max_rounds-1:
                            final = content
                except Exception as e:
                    print(f"  [warn] prompt {i} v{v}: {e}", flush=True); continue
                if final:
                    rec = {"prompt": query, "trajectory": trajectory, "final": final}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 5 == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] {count} trayectorias", flush=True)
    print(f"DONE {count} trayectorias -> {args.out}", flush=True)

if __name__ == "__main__":
    main()