#!/usr/bin/env python3
"""Probe: detecta dominios emergentes en una muestra del corpus v5 (solo lectura)."""
import json, re, sys
from collections import Counter
DOMAINS = {
    "eco": ["ecolog", "biodivers", "species distribution", "maxent", "niche", "occupancy", "habitat", "conservation", "invasion", "community ecology", "population dynamic", "ecosystem"],
    "phylo": ["phylogen", "phylogeograph", "evolut", "natural selection", "adaptation", "speciation", "divergence", "comparative method", "biogeograph", "molecular evolution"],
    "genom": ["genom", "transcriptom", "population genetics", "gwas", "genome assembl", "gene expression", "metagenom", "whole-genome", "exome", "sequencing"],
    "bioc": ["bioinformatic", "machine learning", "deep learning", "neural network", "computational", "modeling", "simulation", "statistical model", "bayesian", "algorithm"],
    "microbio": ["microbi", "bacteria", "archaea", "virus", "pathogen", "infection", "antimicrobial", "probiotic", "gut flora", "microorganism"],
    "evolution-anthro": ["human evolution", "archaeolog", "paleo", "fossil", "hominin", "anthropolog", "primate evolution", "paleoecolog", "quaternary"],
    "climate": ["climate change", "global warming", "temperature", "precipitation", "carbon", "co2", "greenhouse", "climate model", "warming", "sea level"],
    "genetics-medical": ["disease", "clinical", "medical genetics", "variant", "mutation", "hereditary", "cancer", "genetic disorder", "therapeutic", "patient"],
    "conservation-mgmt": ["protected area", "management", "restoration", "endangered", "threatened", "policy", "sustainable", "natural resource", "wildlife management"],
    "plant": ["plant", "botan", "flora", "crop", "agricultur", "forest", "tree", "pollination", "vegetation", "herbivor"],
}
def score(text, pats):
    t = text.lower()
    return sum(1 for p in pats if p in t)
c = Counter()
n = 0
for line in sys.stdin:
    try: d = json.loads(line)
    except: continue
    n += 1
    text = d.get("text","")
    best, bs = None, 0
    for dom, pats in DOMAINS.items():
        s = score(text, pats)
        if s > bs: best, bs = dom, s
    c[best if bs>0 else "otro"] += 1
    if n >= 200000: break
print(f"docs={n}")
for dom, cnt in c.most_common():
    print(f"  {dom}: {cnt} ({100*cnt/n:.1f}%)")
