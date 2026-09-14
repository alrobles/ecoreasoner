#!/usr/bin/env python3
"""corpus_audit.py — auditoria del corpus por embeddings.

Sobre el emb dir de embed_docs.py:
  1. dedup: kNN auto-similitud; pares > dedup-thr -> union-find, keep primero
  2. leakage: sim max doc<->holdout; flags > leak-thr (lista a excluir)
  3. kNN graph k=16 -> edges.tsv (para hard-negative mining posterior)
  4. report.json con distribuciones

Uso:
  python3 corpus_audit.py --emb-dir EMB_SKELETON --holdout-emb-dir EMB_HO \
      --out audit_v3/ [--dedup-thr 0.95] [--leak-thr 0.90] [--knn-k 16]
"""
import argparse, json, os
import numpy as np


def topk_chunked(A, B, k, device, exclude_self=False):
    """max-sim + argmax por chunks; si exclude_self, ignora i==i (misma matriz)."""
    import torch
    At = torch.from_numpy(np.asarray(A, dtype=np.float32)).to(device)
    Bt = torch.from_numpy(np.asarray(B, dtype=np.float32)).to(device)
    N = At.shape[0]
    sims = torch.empty(N, k, dtype=torch.float32)
    idxs = torch.empty(N, k, dtype=torch.int64)
    CH = 8192
    for i0 in range(0, N, CH):
        i1 = min(i0 + CH, N)
        S = At[i0:i1] @ Bt.T
        if exclude_self:
            S[torch.arange(i1 - i0), torch.arange(i0, i1)] = -2.0
        v, ix = S.topk(k, dim=1)
        sims[i0:i1] = v.cpu(); idxs[i0:i1] = ix.cpu()
        del S
    return sims.numpy(), idxs.numpy()


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--holdout-emb-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dedup-thr", type=float, default=0.95)
    ap.add_argument("--leak-thr", type=float, default=0.90)
    ap.add_argument("--knn-k", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    emb = np.memmap(os.path.join(args.emb_dir, "emb.fp16.npy"),
                    dtype=np.float16, mode="r")
    N, dim = emb.shape[0], emb.shape[1]
    emb = emb.reshape(N, dim)
    print(f"[audit] corpus N={N} dim={dim}", flush=True)

    report = {"n_docs": int(N), "dedup_thr": args.dedup_thr,
              "leak_thr": args.leak_thr}

    # ---------- self kNN: dedup + graph ----------
    kq = args.knn_k + 1
    sims, idxs = topk_chunked(emb, emb, kq, args.device, exclude_self=True)
    # dedup via union-find sobre pares > thr
    dsu = DSU(N)
    n_pairs = 0
    for i in range(N):
        for j in range(kq):
            if idxs[i, j] >= 0 and sims[i, j] > args.dedup_thr:
                dsu.union(i, int(idxs[i, j])); n_pairs += 1
    keep, drop = [], []
    seen_root = {}
    for i in range(N):
        r = dsu.find(i)
        if r in seen_root:
            drop.append(i)
        else:
            seen_root[r] = i; keep.append(i)
    report["dedup_pairs"] = int(n_pairs)
    report["dedup_drop"] = len(drop)
    report["dedup_keep"] = len(keep)
    np.save(os.path.join(args.out, "keep_idx.npy"), np.array(keep, dtype=np.int64))
    np.save(os.path.join(args.out, "drop_idx.npy"), np.array(drop, dtype=np.int64))
    print(f"[audit] dedup: {n_pairs} pares>{args.dedup_thr}, "
          f"drop {len(drop)}, keep {len(keep)}", flush=True)

    # ---------- kNN graph (k vecinos, sin self) ----------
    edges = os.path.join(args.out, "knn_edges.tsv")
    with open(edges, "w") as f:
        f.write("src\tdst\tsim\n")
        for i in range(N):
            for j in range(min(args.knn_k, kq)):
                if idxs[i, j] >= 0:
                    f.write(f"{i}\t{int(idxs[i,j])}\t{sims[i,j]:.4f}\n")
    print(f"[audit] kNN graph -> {edges}", flush=True)

    # ---------- leakage vs holdout ----------
    if args.holdout_emb_dir:
        he = np.memmap(os.path.join(args.holdout_emb_dir, "emb.fp16.npy"),
                       dtype=np.float16, mode="r")
        he = he.reshape(he.shape[0], dim)
        hsims, hidxs = topk_chunked(emb, he, 1, args.device)
        max_per_doc = hsims[:, 0]
        leaks = np.where(max_per_doc > args.leak_thr)[0]
        report["holdout_n"] = int(he.shape[0])
        report["leak_docs"] = int(len(leaks))
        qs = np.percentile(max_per_doc, [50, 90, 99, 99.9]).tolist()
        report["holdout_sim_percentiles"] = qs
        np.save(os.path.join(args.out, "leak_idx.npy"), leaks.astype(np.int64))
        # peores ofensores con idx para inspeccion
        worst = np.argsort(-max_per_doc)[:50]
        with open(os.path.join(args.out, "leak_top.tsv"), "w") as f:
            f.write("doc_idx\tholdout_idx\tsim\n")
            for i in worst:
                f.write(f"{int(i)}\t{int(hidxs[i,0])}\t{max_per_doc[i]:.4f}\n")
        print(f"[audit] leakage: {len(leaks)} docs >{args.leak_thr}; "
              f"p99={qs[2]:.3f}", flush=True)

    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("[audit] DONE", flush=True)


if __name__ == "__main__":
    main()
