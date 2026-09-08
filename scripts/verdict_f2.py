#!/usr/bin/env python3
"""verdict_f2.py — herramienta de veredicto para el run f2-spanes.

Lee <run-dir>/eval_curve.jsonl (deduplica por step, conserva la última fila buena
e ignora filas con error), estima la tendencia de los últimos 3 puntos y emite
un veredicto:

  HIT      : last acc >= 0.60
  FALSIFY  : last acc <= 0.55 y por debajo de la referencia de partida (0.535)
  EXTEND   : 0.535 < last acc < 0.60 y pendiente last-3 > 0
  NO-GO    : el resto (no supera la referencia o no tiene tendencia positiva)
  INSUFFICIENT: menos de 3 puntos evaluados

Uso:
  python3 scripts/verdict_f2.py --run-dir runs/f2-spanes

Solo stdlib (json, argparse, pathlib, sys). No requiere GPU.
"""
import argparse
import json
import sys
from pathlib import Path

START_REF = 0.535
HIT_THRESHOLD = 0.60
FALSIFY_THRESHOLD = 0.55
MIN_POINTS_TREND = 3

RECOMMENDATIONS = {
    "HIT": "Archive as positive result.",
    "EXTEND": "Launch f2 to 100K steps.",
    "FALSIFY": "Archive the line as falsified and move to the controller/verificator architecture.",
    "NO-GO": "Archive the line as falsified and move to the controller/verificator architecture.",
}


def load_curve(jsonl_path):
    """Carga eval_curve.jsonl, deduplica por step, ignora errores.

    Devuelve una lista de tuplas (step, pairwise_acc) ordenadas por step.
    Si hay múltiples filas para el mismo step, gana la última del archivo.
    """
    by_step = {}
    if not jsonl_path.exists():
        return []
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = row.get("step")
        if step is None:
            continue
        # Ignorar filas de error o sin medida de discriminación
        if "error" in row or row.get("pairwise_acc") is None:
            continue
        by_step[step] = row
    points = [(int(s), float(by_step[s]["pairwise_acc"])) for s in sorted(by_step)]
    return points


def slope(points):
    """Pendiente de la regresión lineal simple y ~ x usando los puntos dados."""
    n = len(points)
    if n < MIN_POINTS_TREND:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0.0:
        return 0.0
    return cov / var


def mean_acc(points):
    if not points:
        return None
    return sum(p[1] for p in points) / len(points)


def compute_verdict(points):
    n = len(points)
    if n < MIN_POINTS_TREND:
        return "INSUFFICIENT", None, None, None, None

    last_step, last_acc = points[-1]
    last3 = points[-3:]
    first3 = points[:3]
    s = slope(last3)
    mean_last = mean_acc(last3)
    mean_first = mean_acc(first3)
    delta = mean_last - mean_first if (mean_last is not None and mean_first is not None) else None

    if last_acc >= HIT_THRESHOLD:
        verdict = "HIT"
    elif last_acc <= FALSIFY_THRESHOLD and last_acc < START_REF:
        verdict = "FALSIFY"
    elif last_acc > START_REF and last_acc < HIT_THRESHOLD and s is not None and s > 0.0:
        verdict = "EXTEND"
    else:
        verdict = "NO-GO"

    return verdict, last_step, last_acc, s, delta


def build_report(verdict, last_step, last_acc, slope, delta, points):
    if verdict == "INSUFFICIENT":
        needed = max(0, MIN_POINTS_TREND - len(points))
        recommendation = (
            f"Need at least {MIN_POINTS_TREND} evaluated checkpoints to estimate trend "
            f"({len(points)} found, need {needed} more)."
        )
    else:
        recommendation = RECOMMENDATIONS.get(verdict, "")

    report = {
        "verdict": verdict,
        "last_step": last_step,
        "last_acc": round(last_acc, 4) if last_acc is not None else None,
        "start_ref": START_REF,
        "hit_threshold": HIT_THRESHOLD,
        "falsify_threshold": FALSIFY_THRESHOLD,
        "slope_last3": round(slope, 6) if slope is not None else None,
        "delta_first3_last3": round(delta, 6) if delta is not None else None,
        "n_points": len(points),
        "points": [[p[0], p[1]] for p in points],
        "recommendation": recommendation,
    }
    return report


def print_block(report):
    print("=" * 62)
    print(f"VERDICT: {report['verdict']}")
    print("-" * 62)
    print(f"Last evaluated step : {report['last_step'] if report['last_step'] is not None else 'N/A'}")
    print(f"Last pairwise_acc   : {f'{report['last_acc']:.4f}' if report['last_acc'] is not None else 'N/A'}")
    print(f"Start reference     : {report['start_ref']:.3f}")
    print(f"Hit threshold       : {report['hit_threshold']:.2f}")
    print(f"Falsify threshold   : {report['falsify_threshold']:.2f}")
    print(f"Slope (last 3)      : {f'{report['slope_last3']:.6f}' if report['slope_last3'] is not None else 'N/A'}")
    print(f"Delta (last3-first3): {f'{report['delta_first3_last3']:+.4f}' if report['delta_first3_last3'] is not None else 'N/A'}")
    print(f"N points            : {report['n_points']}")
    print("-" * 62)
    print("Trajectory:")
    print(f"  {'step':>8} {'pairwise_acc':>12}")
    for step, acc in report["points"]:
        print(f"  {step:>8} {acc:>12.4f}")
    print("-" * 62)
    print(f"Recommendation: {report['recommendation']}")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs/f2-spanes",
                    help="directorio del run (default: runs/f2-spanes)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    jsonl_path = run_dir / "eval_curve.jsonl"
    out_json = run_dir / "verdict_f2.json"

    points = load_curve(jsonl_path)
    verdict, last_step, last_acc, s, delta = compute_verdict(points)
    report = build_report(verdict, last_step, last_acc, s, delta, points)
    print_block(report)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[verdict_f2] written {out_json}")


if __name__ == "__main__":
    main()
