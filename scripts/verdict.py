#!/usr/bin/env python3
"""verdict.py — veredicto estadístico para evaluaciones de discriminación.

Sucesor de verdict_f2.py (que se conserva). Corrige el defecto estadístico que
el postmortem de F2 documentó: la curva se miró 69 veces y el "pico" 0.566 era
indistinguible del máximo esperado de un modelo al azar (Monte Carlo: P(max >=
0.566 | azar, 69 miradas, n=256) ~= 0.74).

Reglas de medición v2 (pre-registrables):
  - la DECISIÓN usa únicamente el punto final pre-registrado; la curva queda
    como diagnóstico (si hay varios puntos de decisión, corregir alpha por
    Bonferroni: alpha / n_puntos).
  - reportar SIEMPRE: n, IC de Wilson 95%, p binomial exacto unilateral vs
    H0: p = 0.5 (azar).
  - tamaño muestral: para detectar +5 p.p. sobre azar con potencia 80% y
    alpha=0.05 se necesitan n ~= 620 pares; n=256 solo resuelve ~+-6 p.p.
  - modo --battery: agrega los evals por nivel L0-L3 y traduce el patrón
    (L0/L1 = coherencia temática, L2 = orden de etapas, L3 = inferencia).

Uso:
  python3 scripts/verdict.py --curve runs/f2-spanes/eval_curve.jsonl \
      --n-pairs 256
  python3 scripts/verdict.py --report runs/pairs_hard/eval_L3.json
  python3 scripts/verdict.py --battery runs/f2-spanes/battery_L0L3 \
      --out runs/f2-spanes/battery_verdict.json
  python3 scripts/verdict.py --selftest
"""
import argparse, glob, json, math, sys
from pathlib import Path


# ---------- estadística (stdlib, sin scipy) ----------

def wilson_ci(k, n, z=1.959964):
    """Intervalo de Wilson para proporción k/n."""
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - half) / den), min(1.0, (centre + half) / den))


def binom_sf(k, n, p0=0.5):
    """P(X >= k | Bin(n, p0)) exacto, en log-espacio (robusto para n grande)."""
    if n <= 0 or k <= 0:
        return 1.0
    lnp = math.log(p0)
    lnq = math.log(1 - p0)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * lnp + (n - i) * lnq for i in range(k, n + 1)]
    mx = max(terms)
    return min(1.0, math.exp(mx) * sum(math.exp(t - mx) for t in terms))


def min_n_for_effect(delta, alpha=0.05, power=0.8):
    """n aproximado para detectar acc = 0.5 + delta sobre azar (unilateral)."""
    z_a = _inv_norm(1 - alpha)
    z_b = _inv_norm(power)
    return math.ceil(((z_a + z_b) ** 2) * 0.25 / (delta * delta))


def _inv_norm(p):
    """Aproximación de Acklam a la normal inversa (suficiente para potencia)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def sig_stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def eval_point(acc, n, go, hit, ref, slope, alpha, is_final):
    """Veredicto de UN punto con estadística. Devuelve (label, dict)."""
    k = int(round(acc * n))
    lo, hi = wilson_ci(k, n)
    p = binom_sf(k, n)
    significant = p < alpha
    if is_final is False:
        label = "DIAGNOSTIC"
    elif acc >= hit and significant:
        label = "HIT"
    elif acc > ref and slope > 0 and significant:
        label = "EXTEND"
    elif acc <= go and not significant:
        label = "FALSIFY"
    elif acc <= go and significant:
        label = "NO-GO"          # bajo el umbral pero distinto del azar (paradójico)
    else:
        label = "NO-GO"
    return label, {
        "n": n, "acc": acc, "k_wins": k,
        "ci95_wilson": [round(lo, 4), round(hi, 4)],
        "p_binom_vs_chance": round(p, 6), "significant": significant,
        "slope_last3": slope,
    }


def _slope_last3(rows):
    xs = [r["step"] for r in rows[-3:]]
    ys = [r["pairwise_acc"] for r in rows[-3:]]
    if len(xs) < 3:
        return None
    xm, ym = sum(xs) / 3, sum(ys) / 3
    den = sum((x - xm) ** 2 for x in xs)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den if den else 0.0


def load_curve(path):
    by_step = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("pairwise_acc") is None:
            continue  # filas error
        by_step[r["step"]] = r   # última escritura gana (dedupe)
    return [by_step[k] for k in sorted(by_step)]


def curve_verdict(curve_path, n_pairs, args):
    rows = load_curve(curve_path)
    if not rows:
        print("curva sin puntos válidos")
        return 2
    slope = _slope_last3(rows)
    final = rows[-1]
    label, stats = eval_point(final["pairwise_acc"], n_pairs,
                            args.go, args.hit, args.ref, slope or 0.0,
                            args.alpha, is_final=True)
    # contexto: ¿el máximo de la curva sería plausible bajo azar?
    peak = max(r["pairwise_acc"] for r in rows)
    n_looks = len(rows)
    p_peak_single = binom_sf(int(round(peak * n_pairs)), n_pairs)
    # aprox. sidak: P(max >= peak) ~ 1 - (1 - p_single)^looks (indep. aprox)
    p_peak_multi = 1 - (1 - p_peak_single) ** n_looks
    out = {
        "verdict": label, "final_step": final["step"], **stats,
        "curve": {"n_points": n_looks, "peak_acc": peak,
                  "p_peak_under_chance_multi_look": round(p_peak_multi, 4)},
        "rule": "decisión = punto final pre-registrado; la curva es diagnóstico",
        "thresholds": {"go": args.go, "hit": args.hit, "ref": args.ref,
                       "alpha": args.alpha},
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


def battery_report(batt_dir, args):
    """Agrega evals por nivel: archivos que contengan {pairwise_acc, n_pairs}
    con nombre *_L{0..3}*.json (p.ej. battery_L0.json) o un único report por
    nivel. Emite tabla + interpretación + battery_verdict.json."""
    files = sorted(glob.glob(str(Path(batt_dir) / "*_L*.json")))
    rows = []
    for fp in files:
        lvl = Path(fp).stem.rsplit("_L", 1)[-1]
        if lvl not in ("0", "1", "2", "3"):
            continue
        try:
            d = json.load(open(fp))
        except json.JSONDecodeError:
            continue
        acc, n = d.get("pairwise_acc"), d.get("n_pairs") or args.n_pairs
        if acc is None:
            continue
        k = int(round(acc * n))
        lo, hi = wilson_ci(k, n)
        p = binom_sf(k, n)
        rows.append({"level": f"L{lvl}", "file": Path(fp).name, "n": n,
                     "acc": acc, "ci95": [round(lo, 4), round(hi, 4)],
                     "p": round(p, 6), "sig": sig_stars(p)})
    if not rows:
        print(f"sin evals de nivel en {batt_dir} (esperaba *_L{{0..3}}*.json)")
        return 2
    print(f"{'lvl':<4} {'n':>5} {'acc':>7} {'IC95':>15} {'p':>9}  sig")
    for r in rows:
        print(f"{r['level']:<4} {r['n']:>5} {r['acc']:>7.4f} "
              f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] {r['p']:>9.5f}  {r['sig']}")
    interp = {
        "L0": "coherencia temática (baseline débil)",
        "L1": "tópico controlado a dominio",
        "L2": "orden/rol de etapa (mismo doc)",
        "L3": "inferencia de contenido (payload mutado)",
    }
    sig = {r["level"]: r["p"] < args.alpha for r in rows}
    reading = []
    if sig.get("L3"):
        reading.append("SEÑAL INFERENCIAL: discrimina el payload mutado -> "
                       "la línea no está falsada; la métrica L0 era demasiado fácil.")
    elif sig.get("L2"):
        reading.append("ESTRUCTURA SIN INFERENCIA: distingue el orden de etapas "
                       "pero no el contenido -> gramática aprendida, inferencia no.")
    elif sig.get("L1") or sig.get("L0"):
        reading.append("SOLO COHERENCIA TEMÁTICA: el atajo documentado "
                       "(NSP/overlap) explica cualquier señal previa.")
    else:
        reading.append("FALSACIÓN FUERTE: ni siquiera orden de etapas sobre azar.")
    out = {"levels": rows, "level_meaning": interp, "reading": reading,
           "alpha": args.alpha}
    print("\n".join(reading))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


def _selftest():
    # n=256: acc 0.5273 (F2 final) debe ser ns; acc 0.60 debe ser **;
    # n=1000: acc 0.55 debe ser claramente significativo
    for acc, n in ((0.5273, 256), (0.60, 256), (0.55, 1000), (0.50, 256)):
        k = int(round(acc * n))
        p = binom_sf(k, n)
        lo, hi = wilson_ci(k, n)
        print(f"acc={acc} n={n}: p={p:.5f} {sig_stars(p)} IC=[{lo:.3f},{hi:.3f}]")
    assert binom_sf(int(round(0.5273 * 256)), 256) > 0.05
    assert binom_sf(int(round(0.60 * 256)), 256) < 0.01
    assert 600 <= min_n_for_effect(0.05) <= 650
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", help="eval_curve.jsonl de un run")
    ap.add_argument("--report", help="report.json de suite_smoke (un punto)")
    ap.add_argument("--battery", metavar="DIR", help="directorio con *_L*.json")
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--step", type=int, default=None,
                    help="punto de decisión (default: último step válido)")
    ap.add_argument("--go", type=float, default=0.55)
    ap.add_argument("--hit", type=float, default=0.60)
    ap.add_argument("--ref", type=float, default=0.535)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", help="escribir veredicto JSON")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.battery:
        return battery_report(a.battery, a)
    if a.report:
        d = json.load(open(a.report))
        n = d.get("n_pairs") or a.n_pairs
        label, stats = eval_point(d["pairwise_acc"], n, a.go, a.hit, a.ref,
                                  0.0, a.alpha, is_final=True)
        out = {"verdict": label, **stats}
        print(json.dumps(out, indent=2))
        if a.out:
            Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
        return 0
    if a.curve:
        return curve_verdict(a.curve, a.n_pairs, a)
    ap.error("indica --curve, --report o --battery")
    return 2


if __name__ == "__main__":
    sys.exit(main())
