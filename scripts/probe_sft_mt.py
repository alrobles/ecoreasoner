"""probe_sft_mt.py — sonda multi-turno para ckpt SFT (default sft-v4 g16000).

Encadena K dialogos de 4 turnos: la respuesta generada se reinyecta como
contexto (formato entrenamiento: "[USER] q\\n[ASSISTANT] a\\n" por turno).
Mide disciplina de frontera de turno:
  - 'eos'        : el modelo emitio EOS dentro de la ventana (cierre limpio)
  - 'next_turn'  : genero "\\n[USER]" -> intento abrir el siguiente turno
                   (lo cortamos nosotros; boundary aprendido a medias)
  - 'max_new'    : no paro -> sobre-generacion (mala frontera)
  - leak_user    : "[USER]" aparece DENTRO de la respuesta extraida
Reutiliza loader+gen por confianza de probe_chat_v5.py.

Uso (cluster): sbatch --export=ALL,CKPT=...,N_DIALOGS=6 probe_sft_mt.slurm
"""
import os, sys, json
from pathlib import Path
import torch

BASE = Path("/beegfs/a474r867/ecoreasoner")
os.environ.setdefault("CKPT", str(BASE/"runs/sft-v4/checkpoint-g16000/ema_model.pt"))
sys.path.insert(0, str(BASE/"scripts"))
import probe_chat_v5 as P

MAX_NEW = int(os.environ.get("MAX_NEW", "128"))
STEPS = int(os.environ.get("STEPS", "96"))
TEMP = float(os.environ.get("TEMP", "0.0"))
N_DIALOGS = int(os.environ.get("N_DIALOGS", "6"))

# Cadenas de 4 turnos estilo QA-cientifico (dominio de sft_ids_v4_mt).
DIALOGS = [
    ["What is a species distribution model?",
     "Which algorithm is most commonly used to fit them?",
     "What kind of occurrence data do they need?",
     "Name one important limitation of these models."],
    ["What is the main purpose of a photosynthesis experiment?",
     "Which variables are usually measured in such an experiment?",
     "How is light intensity typically manipulated?",
     "What result would you expect at very low light?"],
    ["What does a Bayesian hierarchical model add over a simple regression?",
     "What is the role of the prior distribution?",
     "How are hyperparameters estimated?",
     "When would you prefer a non-hierarchical model?"],
    ["What is a metapopulation?",
     "How does habitat fragmentation affect metapopulation dynamics?",
     "What is the rescue effect?",
     "Give an example of a species living in a metapopulation."],
    ["Why do researchers use null models in community ecology?",
     "What pattern are they usually testing against?",
     "Name one commonly used null model.",
     "What is a criticism of the null model approach?"],
    ["Que es el nicho ecologico de una especie?",
     "Cual es la diferencia entre nicho fundamental y nicho realizado?",
     "Que factores reducen el nicho realizado?",
     "Como se estima el nicho a partir de registros de presencia?"],
]

def run_turn(model, tok, context_ids, dev):
    """Genera la respuesta de un turno. Devuelve (texto_cortado, info)."""
    ids = context_ids[-(P.SEQ_LEN - MAX_NEW):]
    out = P.gen(model, ids, MAX_NEW, STEPS, TEMP, dev)
    new = [t for t in out[len(ids):] if t != P.MASK_ID]
    raw = tok.decode(new)
    # corte: \n[USER (siguiente turno) | EOS/token 0 | fin de ventana
    stop_ids = {0, tok.eos_token_id, tok.pad_token_id} - {None}
    eos_at = next((j for j, t in enumerate(new) if t in stop_ids), None)
    user_at = raw.find("\n[USER")
    info = {"n_tok": len(new), "boundary": "max_new"}
    if eos_at is not None:
        info["boundary"] = "eos"
        answer = tok.decode(new[:eos_at])
    elif user_at >= 0:
        info["boundary"] = "next_turn"
        answer = raw[:user_at]
    else:
        answer = raw
    info["leak_user"] = "[USER]" in answer
    info["answer"] = answer.strip()
    return info["answer"].strip(), info

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = P.load_model(dev)
    n_par = sum(p.numel() for p in model.parameters())/1e9
    print(f"[probe-mt] ckpt={P.CKPT}\n[probe-mt] params={n_par:.2f}B dev={dev}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["TOK"], trust_remote_code=True)

    stats = {"eos": 0, "next_turn": 0, "max_new": 0, "leaks": 0, "turns": 0}
    for di, dialog in enumerate(DIALOGS[:N_DIALOGS]):
        print(f"\n########## DIALOG {di} ##########")
        ctx_ids = []
        for ti, q in enumerate(dialog):
            seg = f"[USER] {q}\n[ASSISTANT]"
            ctx_ids += tok(seg, add_special_tokens=False)["input_ids"]
            ans, info = run_turn(model, tok, ctx_ids, dev)
            stats["turns"] += 1
            stats[info["boundary"]] += 1
            stats["leaks"] += int(info["leak_user"])
            print(f"--- turno {ti} [{info['boundary']}] tok={info['n_tok']} "
                  f"leak={info['leak_user']} ---\nQ: {q}\nA: {info['answer']}")
            # reinjectar igual que en entrenamiento: " <a>\n"
            ctx_ids += tok(f" {ans}\n", add_special_tokens=False)["input_ids"]
    t = max(stats["turns"], 1)
    print(f"\n========== RESUMEN ==========\n"
          f"turns={t} eos={stats['eos']} ({stats['eos']/t:.0%}) "
          f"next_turn={stats['next_turn']} ({stats['next_turn']/t:.0%}) "
          f"max_new={stats['max_new']} ({stats['max_new']/t:.0%}) "
          f"leaks={stats['leaks']}")

if __name__ == "__main__":
    main()
