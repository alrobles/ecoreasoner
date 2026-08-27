#!/usr/bin/env python3
"""
dataset_catalog.py — AUC titulado del registro de datasets de EcoReasoner.

Recorre el directorio de datos real en el cluster /beegfs/a474r867/ecoreasoner/data/
(o el directorio dado con --data) y lista los datasets con su código canónico, tamaño,
nº de docs/líneas y etapa. Ayuda a no mezclar datasets cuando se corre en paralelo.

Modo de uso (en el cluster, donde están los datos):
    python3 dataset_catalog.py                       # escanea el data/ por defecto
    python3 dataset_catalog.py --data /beegfs/a474r867/ecoreasoner/data
    python3 dataset_catalog.py --group               # agrupa por etapa

NO modifica nada: es solo un auditor/helper del registro docs/dataset-registry.md.
"""
import argparse, os, gzip, re

# (suen: estos se usan como presentación, no sustituyen al doc)
PREFIX_STAGE_CHAR = {
    "1": "ETAPA 1 - FUENTES (ingesta)",
    "2": "ETAPA 2 - PRETRAIN (input Fase A)",
    "3": "ETAPA 3 - DESTILACIÓN (input Fase B/C)",
    "4": "ETAPA 4 - EVAL/BENCHMARK",
    "P": "PROMPTS",
    "T": "PROMPTS toolcall",
    "X": "AUX/inventario/provenance",
}

APPROX_TYPES = {
    ".jsonl": ("jsonl", "lineas"),
    ".npy": ("npy", "tokens (cabecera)"),
    ".meta.json": ("meta", "bytes"),
    ".inv.gz": ("inventarioS3", "lineas.gz"),
    ".pkl": ("pickle", "bytes"),
    ".txt": ("texto-aux", "lineas"),
    ".log": ("log", "lineas"),
}


def file_len(path, gz=False):
    """Cuenta líneas de forma rápida."""
    try:
        op = gzip.open(path, "rt", errors="ignore") if gz else open(path, "r", errors="ignore")
        n = sum(1 for _ in op)
        op.close()
        return n
    except Exception:
        return None


def canonical_code(fname):
    """Deriva un código canónico del nombre de archivo, o None."""
    low = fname.lower()
    # ETAPA 3 logs
    if "distill_teacher" in low and low.endswith(".log"):
        return "X_log"
    if fname.startswith("_") or low.endswith(".log") or low.endswith(".pkl"):
        return "X_aux"
    # intentar descriptores ya canónicos
    m = re.match(r"^(\d|T|P|X)_", fname)
    if m:
        return fname
    # reglas de escalado hacia códigos estables
    base = fname
    low = fname.lower()
    # ETAPA 2
    if low.startswith("train_corpus_v"):
        m2 = re.match(r"train_corpus_v(\d+)\.jsonl", low)
        n = m2.group(1) if m2 else "?"
        return f"2_pretrain_{n}"
    if low.startswith("train_ids_v"):
        m2 = re.match(r"train_ids_v(\d+)\.npy", low)
        n = m2.group(1) if m2 else "?"
        return f"2_ids_{n}"
    # ETAPA 3
    if low.startswith("distill_v4_round"):
        m2 = re.match(r"distill_v4_round(\d+)\.jsonl", low)
        n = m2.group(1) if m2 else "?"
        return f"3_distill_r{n}"
    if low.startswith("distill_data"):
        return "3_distill_0"
    if low.startswith("prompts_toolcall_canonical"):
        return "T_toolcall"
    if low.startswith("sci_v"):
        return "4_eval_sci"
    if low.startswith("ecoevorxiv_fulltext_c"):
        return "1_evorxiv"
    if "fulltext_corpus" in low:
        return "1_pmc_full"
    if low.startswith("eco_corpus_v"):
        m2 = re.match(r"eco_corpus_v(\d+)\.jsonl", low)
        n = m2.group(1) if m2 else "?"
        return f"1_pub_v{n}"
    if low.startswith("provenance") or "license_map" in low or "osf_map" in low:
        return "X_prov"
    if low.endswith(".inv.gz"):
        return "X_inv"
    return "?"


def human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/beegfs/a474r867/ecoreasoner/data")
    ap.add_argument("--group", action="store_true", help="agrupa por ETAPA")
    a = ap.parse_args()

    if not os.path.isdir(a.data):
        print(f"[catalog] ruta no existe: {a.data}")
        return

    rows = []
    for f in sorted(os.listdir(a.data)):
        p = os.path.join(a.data, f)
        if not os.path.isfile(p):
            continue
        gz = f.endswith(".gz")
        size = os.path.getsize(p)
        binary = f.endswith((".npy", ".pkl"))
        nlin = None if binary else file_len(p, gz=gz)
        code = canonical_code(f)
        rows.append((code, f, size, nlin))

    if a.group:
        by_stage = {}
        for r in rows:
            stage = r[0][0] if r[0] else "?"
            by_stage.setdefault(stage, []).append(r)
        for stage in sorted(by_stage):
            print(f"\n== {PREFIX_STAGE_CHAR.get(stage, 'ETAPA ?')} ==")
            for code, f, size, n in by_stage[stage]:
                extra = f"{human(n)} lineas" if n else ""
                print(f"  {code:<18} {f:<38} {human(size):>10}  {extra}")
    else:
        print(f"== {a.data} ==")
        for code, f, size, n in rows:
            ln = f"{n:,} lines" if n else "?"
            print(f"  {code:<18} {f:<38} {human(size):>10}  {ln}")


if __name__ == "__main__":
    main()