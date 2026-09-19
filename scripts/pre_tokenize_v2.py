#!/usr/bin/env python3
"""pre_tokenize_v2.py — tokeniza corpus con EOS y guarda .npz para packing v2.

Lee un corpus JSONL, tokeniza con el tokenizer LLaDA, añade EOS al final de
cada documento y guarda un .npz con:
  - ids:      array int32 plano con todos los tokens (incluyendo EOS).
  - lengths:  array int32 con la longitud de cada documento (incluyendo EOS).
  - eos_id:   token id del EOS.

Esto permite a train_mdlm_moe_v2.py hacer packing con fronteras de documento.

Uso:
  python3 pre_tokenize_v2.py --input data/train_skeleton.jsonl \
      --tokenizer <dir> --out data/train_ids_skeleton_v2.npz [--workers 16]
"""
import argparse, json, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer


def _clamp(ids, vocab, eos_id):
    """Asegura que todos los ids esten dentro del vocab del tokenizer."""
    if eos_id is not None and eos_id >= vocab:
        eos_id = 0
    out = []
    for x in ids:
        if x >= vocab:
            out.append(0)
        else:
            out.append(x)
    return out, eos_id


def encode_batch(args):
    lines, tok_path, max_len, field, split_long = args
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True,
                                        local_files_only=True)
    vocab = tok.vocab_size
    eos_id = tok.eos_token_id
    if eos_id is None:
        eos_id = tok.pad_token_id
    ids_all = []
    lengths = []
    clipped = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line).get(field, "")
        except Exception:
            continue
        ids = tok.encode(t, add_special_tokens=False)
        if split_long and len(ids) > max_len - 1:
            # ventanas consecutivas de max_len-1 + EOS: preserva TODOS los
            # tokens del doc (fulltexts no se truncan). Cada ventana es un
            # "doc" para el packing; la cola <4 tok se descarta.
            wins = [ids[i:i + max_len - 1] for i in range(0, len(ids), max_len - 1)]
            for w in wins:
                if len(w) >= 4:
                    w, eos_id = _clamp(w, vocab, eos_id)
                    w.append(eos_id)
                    ids_all.extend(w)
                    lengths.append(len(w))
                else:
                    clipped += 1
            continue
        ids = ids[:max_len - 1]
        if len(ids) >= 4:
            ids, eos_id = _clamp(ids, vocab, eos_id)
            ids.append(eos_id)
            ids_all.extend(ids)
            lengths.append(len(ids))
        else:
            clipped += 1
    return ids_all, lengths, clipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=768)
    ap.add_argument("--field", default="text", help="campo JSON con el texto")
    ap.add_argument("--split-long", action="store_true",
                    help="docs >seq_len se cortan en ventanas (preserva tokens)")
    ap.add_argument("--read-block", type=int, default=200000,
                    help="lineas por bloque de lectura (streaming, limita RAM)")
    args = ap.parse_args()

    t0 = time.time()
    clipped = 0
    n_lines = 0
    n_tok = 0
    # Los IDs se vuelcan a .npy parciales en disco (una por bloque) en vez de
    # acumular una lista de Python (~28B/int -> ~400GB para 15B tok).
    parts_dir = args.out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    blk_i = 0
    def save_part(ids, lens):
        nonlocal blk_i, n_tok
        np.save(os.path.join(parts_dir, f"ids_{blk_i:05d}.npy"),
                np.asarray(ids, dtype=np.int32))
        np.save(os.path.join(parts_dir, f"lens_{blk_i:05d}.npy"),
                np.asarray(lens, dtype=np.int32))
        n_tok += len(ids)
        blk_i += 1
    # streaming por bloques: el corpus puede ser ~40GB, no cabe en RAM entero
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(args.input) as f:
        pending = []
        block = []
        def flush():
            nonlocal block
            if not block:
                return
            chunk = max(1, len(block) // args.workers)
            for i in range(0, len(block), chunk):
                pending.append(ex.submit(
                    encode_batch,
                    (block[i:i + chunk], args.tokenizer, args.seq_len,
                     args.field, args.split_long)))
            block = []
        for line in f:
            block.append(line)
            n_lines += 1
            if len(block) >= args.read_block:
                flush()
                # drena los más viejos para acotar RAM
                while len(pending) > args.workers * 2:
                    ids, lens, cl = pending.pop(0).result()
                    save_part(ids, lens); clipped += cl
                print(f"[{time.strftime('%H:%M:%S')}] {n_lines} lineas, "
                      f"{n_tok/1e6:.0f}M tok acum", flush=True)
        flush()
        for fu in pending:
            ids, lens, cl = fu.result()
            save_part(ids, lens); clipped += cl
    print(f"[{time.strftime('%H:%M:%S')}] leidas {n_lines} lineas de {args.input}",
          flush=True)

    # concatena partes via memmap (RAM acotada) -> ids.npy/lengths.npy finales
    import glob as _glob
    id_parts = sorted(_glob.glob(os.path.join(parts_dir, "ids_*.npy")))
    ln_parts = sorted(_glob.glob(os.path.join(parts_dir, "lens_*.npy")))
    def concat_parts(parts, dst):
        total = sum(int(np.load(p, mmap_mode="r").shape[0]) for p in parts)
        out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.int32,
                                        shape=(total,))
        off = 0
        for p in parts:
            a = np.load(p, mmap_mode="r")
            out[off:off + a.shape[0]] = a
            off += a.shape[0]
        out.flush()
        return dst
    ids_npy = concat_parts(id_parts, os.path.join(parts_dir, "_ids_final.npy"))
    ln_npy = concat_parts(ln_parts, os.path.join(parts_dir, "_lens_final.npy"))
    arr = np.load(ids_npy, mmap_mode="r")
    lengths = np.load(ln_npy, mmap_mode="r")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True,
                                        local_files_only=True)
    vocab = tok.vocab_size
    eos_id = tok.eos_token_id
    if eos_id is None:
        eos_id = tok.pad_token_id
    if eos_id is not None and eos_id >= vocab:
        eos_id = 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, ids=arr, lengths=lengths, vocab_size=int(vocab),
             eos_id=int(eos_id))
    # limpia las partes (quedan solo el npz + meta)
    import shutil
    shutil.rmtree(parts_dir, ignore_errors=True)

    meta = {
        "n_tokens": int(arr.size),
        "n_docs_usable": int(lengths.shape[0]),
        "n_docs_clipped": int(clipped),
        "seq_len": args.seq_len,
        "vocab_size": tok.vocab_size,
        "eos_id": int(eos_id),
        "input": args.input,
        "field": args.field,
        "split_long": bool(args.split_long),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = args.out + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    el = time.time() - t0
    print(f"DONE {n_lines} lineas -> {len(lengths)} docs, {arr.size/1e9:.3f}B tokens "
          f"(incl. EOS) en {el/60:.1f}min", flush=True)
    print(f"guardado: {args.out} + {meta_path}", flush=True)


if __name__ == "__main__":
    main()
