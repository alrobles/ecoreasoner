#!/usr/bin/env python3
"""gen_overseer.py — supervisor de una generacion evolutiva.

Para cada tag:
  DONE   — existe battery_verdict_dense.json
  ALIVE  — train.log fresco (< stale_min) o un job suyo esta en squeue
  DEAD   — sin verdict, log viejo y sin job -> con --write-dead se escribe
           un verdict terminal (diverged) para que watcher/leaderboard cierren.

Deteccion de job vivo: nombres en squeue matcheando el tag, O logs
g0_*.out recientes cuyo contenido mencione "Lanzando G0 <tag>" con un
jobid que siga en squeue (los re-encolados renombran a g0).

Uso:
  python3 gen_overseer.py --runs /beegfs/.../runs --tags "g4-a g4-b" \
      --stale-min 30 [--write-dead] [--user a474r867]
"""
import argparse, glob, json, os, re, subprocess, sys, time

VERDICT = "battery_verdict_dense.json"


def squeue_entries(user):
    """Devuelve (set_de_jobnames, set_de_jobids) vivos."""
    try:
        out = subprocess.run(
            ["squeue", "-u", user, "-h", "-o", "%i %j"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return set(), set()
    names, ids = set(), set()
    for l in out.splitlines():
        p = l.split(None, 1)
        if len(p) == 2:
            ids.add(p[0]); names.add(p[1])
    return names, ids


def tag_has_live_job(tag, names, ids, logs_glob, since_s=86400):
    """True si algun job vivo parece pertenecer a tag."""
    if tag in names:
        return True
    # re-encolados: el job se llama g0; buscar logs recientes que lo mencionen
    now = time.time()
    for lf in sorted(glob.glob(logs_glob), key=os.path.getmtime, reverse=True):
        if now - os.path.getmtime(lf) > since_s:
            break
        try:
            head = open(lf, errors="ignore").read(4096)
        except Exception:
            continue
        if f"Lanzando G0 {tag} " not in head and f"Lanzando G0 {tag}(" not in head \
                and f"G0 {tag}" not in head:
            continue
        m = re.search(r"g0_(\d+)\.out", lf)
        if m and m.group(1) in ids:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--logs", default="/beegfs/a474r867/ecoreasoner/logs/g0_*.out")
    ap.add_argument("--stale-min", type=int, default=30)
    ap.add_argument("--user", default="a474r867")
    ap.add_argument("--write-dead", action="store_true",
                    help="escribe verdict terminal para corridas muertas")
    a = ap.parse_args()

    names, ids = squeue_entries(a.user)
    now = time.time()
    done, alive, dead = [], [], []
    for tag in a.tags.split():
        d = os.path.join(a.runs, tag)
        if os.path.exists(os.path.join(d, VERDICT)):
            done.append(tag); continue
        log = os.path.join(d, "train.log")
        fresh = os.path.exists(log) and (now - os.path.getmtime(log)) < a.stale_min * 60
        nan = False
        if os.path.exists(log) and not fresh:
            try:
                tail = open(log, errors="ignore").read()[-20000:]
                nan = "nan" in tail.lower()
            except Exception:
                pass
        if fresh or tag_has_live_job(tag, names, ids, a.logs):
            alive.append(tag)
        else:
            dead.append(tag)
            if a.write_dead:
                why = "NaN/divergencia en train.log" if nan else \
                    "sin job vivo y log estale >%dmin" % a.stale_min
                v = {"error": f"overseer: corrida terminal sin verdict ({why})",
                     "diverged": True, "levels": []}
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, VERDICT), "w") as fh:
                    fh.write(json.dumps(v))
                done.append(tag)
                print(f"[overseer] verdict terminal escrito: {tag} ({why})")

    print(json.dumps({"done": done, "alive": alive, "dead": dead,
                      "n_done": len(done), "n_alive": len(alive),
                      "n_dead": len(dead)}, indent=2))
    return 0 if not dead else 2


if __name__ == "__main__":
    sys.exit(main())
