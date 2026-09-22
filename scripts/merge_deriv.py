#!/usr/bin/env python3
"""merge_deriv.py — mezcla B2: subsample v5pdb + deriv upsampled a share de tokens.

Lee el cache crudo de v5pdb (train_ids_v5pdb.npz.ids.npy / .lengths.npy,
memmap) en UN solo pase secuencial (evita I/O aleatoria en lustre) y escribe:

  --base-out  train_ids_b2base.npz   : subsample seedeado de v5pdb
  --mix-out   train_ids_b2deriv.npz  : base + deriv ×K (share objetivo)

El brazo control entrena con base-out; el brazo deriv con mix-out. Misma
base, mismos docs base — la única diferencia es la presencia de deriv.

Uso:
  python3 merge_deriv.py --v5pdb data/train_ids_v5pdb.npz \
      --deriv data/train_ids_deriv.npz \
      --base-tokens 400000000 --deriv-share 0.20 --seed 7 \
      --base-out data/train_ids_b2base.npz \
      --mix-out data/train_ids_b2deriv.npz
"""
import argparse, json, os, time
import numpy as np


def raw_paths(npz_path):
    """train_ids_v5pdb.npz -> (ids.npy, lengths.npy) crudos memmap."""
    ids = npz_path + ".ids.npy"
    lens = npz_path + ".lengths.npy"
    if os.path.exists(ids) and os.path.exists(lens):
        return ids, lens
    return None, None


def load_npz_or_memmap(path):
    """Devuelve (ids, lengths) — crudo memmap si existe, si no el npz."""
    ids_p, lens_p = raw_paths(path)
    if ids_p:
        return np.load(ids_p, mmap_mode="r"), np.load(lens_p, mmap_mode="r")
    z = np.load(path)
    return z["ids"], z["lengths"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5pdb", required=True)
    ap.add_argument("--deriv", required=True)
    ap.add_argument("--base-tokens", type=int, default=400_000_000)
    ap.add_argument("--deriv-share", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--base-out", required=True)
    ap.add_argument("--mix-out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    ids, lens = load_npz_or_memmap(args.v5pdb)
    n_docs = int(lens.size)
    print(f"v5pdb: {n_docs} docs, {int(np.asarray(lens).sum())/1e9:.2f}B tok",
          flush=True)
    lens = np.asarray(lens)
    off = np.zeros(n_docs + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])

    rng = np.random.default_rng(args.seed)
    # docs necesarios para ~base-tokens: muestrear hasta cubrir el cupo
    est_mean = off[-1] / n_docs
    n_pick = min(n_docs, int(args.base_tokens / est_mean * 1.05) + 100)
    sel = np.sort(rng.choice(n_docs, size=n_pick, replace=False))
    sel_tok = int(lens[sel].sum())
    print(f"base subsample: {len(sel)} docs ~{sel_tok/1e6:.0f}M tok", flush=True)

    # pase secuencial: recolectar ids de los docs seleccionados
    base_ids = np.empty(sel_tok, dtype=np.int32)
    base_lens = lens[sel].astype(np.int32)
    w = 0
    cursor = 0  # offset actual en ids
    for j, d in enumerate(sel):
        ln = int(lens[d])
        s, e = int(off[d]), int(off[d]) + ln
        base_ids[w:w + ln] = ids[s:e]
        w += ln
        cursor = e
        if j % 50000 == 0:
            print(f"  gather {j}/{len(sel)} ({w/1e6:.0f}M tok)", flush=True)
    base_ids = base_ids[:w]
    print(f"base recolectada: {w/1e6:.1f}M tok en {time.time()-t0:.0f}s", flush=True)

    d_ids, d_lens = load_npz_or_memmap(args.deriv)
    d_ids = np.asarray(d_ids); d_lens = np.asarray(d_lens).astype(np.int32)
    d_tok = int(d_ids.size)
    # share = deriv_tok / (base + deriv) -> deriv_necesario = s/(1-s)*base
    need = int(args.deriv_share / (1 - args.deriv_share) * w)
    k = int(np.ceil(need / d_tok))
    print(f"deriv: {d_tok/1e6:.1f}M tok x{k} = {k*d_tok/1e6:.0f}M "
          f"(share {k*d_tok/(w + k*d_tok):.1%} >= {args.deriv_share:.0%})",
          flush=True)

    meta_common = {"vocab_size": 126080, "eos_id": 0}
    np.savez(args.base_out, ids=base_ids, lengths=base_lens, **meta_common)
    with open(args.base_out + ".meta.json", "w") as f:
        json.dump({"n_tokens": int(w), "n_docs": int(base_lens.size),
                   "src": "v5pdb_subsample", "seed": args.seed,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)

    # mezcla: base + deriv xK, docs intercalados (bloques aleatorios de origen)
    mix_ids = np.concatenate([base_ids] + [d_ids] * k)
    mix_lens = np.concatenate([base_lens] + [d_lens] * k)
    perm = rng.permutation(mix_lens.size)
    # reordenar a nivel doc: ids se reconstruye por doc
    off2 = np.zeros(mix_lens.size + 1, dtype=np.int64)
    np.cumsum(mix_lens, out=off2[1:])
    out_ids = np.empty(mix_ids.size, dtype=np.int32)
    out_lens = mix_lens[perm]
    w2 = 0
    for d in perm:
        ln = int(mix_lens[d])
        out_ids[w2:w2 + ln] = mix_ids[off2[d]:off2[d] + ln]
        w2 += ln
    np.savez(args.mix_out, ids=out_ids, lengths=out_lens, **meta_common)
    with open(args.mix_out + ".meta.json", "w") as f:
        json.dump({"n_tokens": int(w2), "n_docs": int(out_lens.size),
                   "base_docs": int(base_lens.size), "deriv_docs_x": k,
                   "deriv_share": float(k * d_tok / w2), "seed": args.seed,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
    print(f"MIX OK: base {w/1e6:.0f}M + deriv {k*d_tok/1e6:.0f}M = {w2/1e6:.0f}M "
          f"tok -> {args.mix_out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
