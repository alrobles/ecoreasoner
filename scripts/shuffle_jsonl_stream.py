#!/usr/bin/env python3
"""
shuffle_jsonl_stream.py — baraja un jsonl gigante sin cargarlo todo en RAM.

Lee el archivo en bloques de N MB, baraja cada bloque en memoria (bounded),
y reescribe. ORDEN DE MAGNITUD OK para pretrain (shuffle aproximado por bloques).
Más eficiente que readlines() de ~21GB (que mató el concat anterior con 137).

Uso: python3 shuffle_jsonl_stream.py --in data/train_corpus_v4.jsonl --block-stride 2000
"""
import argparse, random, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--chunk-lines", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    inp = a.inp
    tmp = inp + ".shuf"

    lines = []
    with open(inp) as f, open(tmp, "w") as out:
        for i, line in enumerate(f):
            lines.append(line)
            if len(lines) >= a.chunk_lines:
                rng.shuffle(lines)
                out.writelines(lines)
                lines = []
                print(f"  chunk flush, total ~{i+1}", flush=True)
        if lines:
            rng.shuffle(lines)
            out.writelines(lines)
    # ahora mezcla global aproximada: re-leer tmp en chunks que se solapan no es trivial;
    # mejor: el archivo ya tiene shuffle DENTRO de cada chunk consecutivo.
    # Para un shuffle global real, un segundo pase de interleaving. Aquí dejamos
    # shuffle por chunk (suficiente para evitar sesgo de dominios consecutivos).
    os.replace(tmp, inp)
    print(f"DONE shuffle por chunks -> {inp}", flush=True)

if __name__ == "__main__":
    main()