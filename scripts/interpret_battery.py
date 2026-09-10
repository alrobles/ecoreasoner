#!/usr/bin/env python3
"""interpret_battery.py — interpreta el battery_verdict.json de un piloto v2.

Aplica las reglas de lectura de la Fase 3 -new (design doc §13.4):
  - L0/L1 > azar, L2/L3 ~ azar  -> atajo topico/tematico
  - L2 > azar, L3 ~ azar         -> estructura/orden de etapas, sin inferencia
  - L3 > azar                    -> señal inferencial genuina (no falsada)
  - ningun nivel > azar          -> falsacion fuerte

Uso:
  python3 scripts/interpret_battery.py \
      --verdict runs/f0-span-esqueleto-v2-piloto/battery_verdict.json \
      --out runs/f0-span-esqueleto-v2-piloto/interpretation.json
"""
import argparse, json, sys
from pathlib import Path


THRESH = 0.55
ALPHA = 0.05


def load_verdict(path):
    d = json.loads(Path(path).read_text())
    levels = {r["level"]: r for r in d.get("levels", [])}
    return levels, d


def sig(r):
    return r["p"] < ALPHA


def interpret(levels):
    L0 = levels.get("L0", {})
    L1 = levels.get("L1", {})
    L2 = levels.get("L2", {})
    L3 = levels.get("L3", {})

    l3_acc = L3.get("acc", 0.5)
    l3_p = L3.get("p", 1.0)
    l2_acc = L2.get("acc", 0.5)
    l2_p = L2.get("p", 1.0)
    l1_acc = L1.get("acc", 0.5)
    l0_acc = L0.get("acc", 0.5)

    interpretation = {}
    recommendation = None

    if l3_acc > THRESH and l3_p < ALPHA:
        verdict = "INFERENTIAL_SIGNAL"
        interpretation = {
            "label": "SEÑAL INFERENCIAL (L3 > 0.55 y significativo)",
            "meaning": "El modelo discrimina payload mutado de etapas correctas; la línea dLLM no está falsada.",
            "note": "Replicar con ablaciones para descartar artefacto; continuar escalando dLLM.",
        }
        recommendation = "continuar_dLLM"
    elif l2_acc > THRESH and l2_p < ALPHA and (l3_acc <= THRESH or l3_p >= ALPHA):
        verdict = "STAGE_GRAMMAR"
        interpretation = {
            "label": "ESTRUCTURA SIN INFERENCIA (L2 > 0.55, L3 no)",
            "meaning": "El modelo aprendió orden/rol de etapas, pero no el contenido inferencial.",
            "note": "Posible mejora con más datos, arquitectura, o harder negatives; considerar ablaciones.",
        }
        recommendation = "ablate_or_scale"
    elif (l0_acc > THRESH or l1_acc > THRESH):
        verdict = "TOPICAL_SHORTCUT"
        interpretation = {
            "label": "SOLO COHERENCIA TEMÁTICA (L0/L1 > 0.55, L2/L3 no)",
            "meaning": "El modelo se apoya en tópico/lexical overlap, no en la estructura argumentativa.",
            "note": "Igual patrón que F2; falsación de la línea inferencial. Evaluar controller/verificator.",
        }
        recommendation = "pivot_controller"
    else:
        verdict = "STRONG_FALSIFY"
        interpretation = {
            "label": "FALSACIÓN FUERTE (ningún nivel sobre azar)",
            "meaning": "Ni siquiera coherencia tópica; modelo básicamente aleatorio en la batería.",
            "note": "Revisar entrenamiento (underfit, mask, datos) o descartar dLLM para esta escala.",
        }
        recommendation = "debug_or_pivot"

    return {
        "verdict": verdict,
        "thresholds": {"acc": THRESH, "alpha": ALPHA},
        "L0": L0, "L1": L1, "L2": L2, "L3": L3,
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


def main():
    global THRESH, ALPHA
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", required=True, help="battery_verdict.json")
    ap.add_argument("--out", default=None, help="interpretation.json")
    ap.add_argument("--thresh", type=float, default=THRESH)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    THRESH = args.thresh
    ALPHA = args.alpha

    levels, _ = load_verdict(args.verdict)
    if not levels:
        print("[fatal] no levels in verdict", file=sys.stderr)
        sys.exit(2)

    result = interpret(levels)
    print(json.dumps(result, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
