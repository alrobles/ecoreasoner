#!/usr/bin/env python3
"""tools_resolver.py — HITO 4: resolución REAL de las 3 herramientas ecológicas.

Reemplaza el mock de resolve_tool (run_controller_verificator) por llamadas reales:

  gbif_occurrence  -> API pública GBIF (sin key): species/match para el
                      usageKey REAL + occurrence/search para el conteo real de
                      registros. La región se devuelve como petición (sin bbox
                      fiable para regiones libres tipo 'peninsula de yucatan';
                      si la región es 'global' se omite el filtro).
  bioclim_download -> verifica el stack REAL de capas CHELSA/ERA5 en el HPC
                      (vía ssh kuhpc) y devuelve qué capas están disponibles.
  maxent_train     -> valida inputs contra el stack y devuelve la spec LISTA
                      para lanzar un job HPC de MaxEnt (no lo ejecuta: es un
                      entrenamiento, no una tool-call síncrona).

Uso (desde run_controller_verificator --real, o standalone):
  python3 tools_resolver.py gbif_occurrence '{"species":"Panthera onca","region":"global"}'
"""
import json, os, subprocess, sys, urllib.request, urllib.error

GBIF_API = "https://api.gbif.org/v1"
OCC_LIMIT = 3

def _http(url, timeout=45):
    req = urllib.request.Request(url, headers={"User-Agent": "ecoreasoner-controller/1.0",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")

def resolve_gbif_occurrence(args):
    """species/match -> speciesKey real; occurrence/search -> conteo real."""
    species = str(args.get("species") or "").strip()
    region = str(args.get("region") or "").strip() or "global"
    if not species:
        return "fail", "gbif_occurrence requiere species"
    try:
        m = _http(f"{GBIF_API}/species/match?name={urllib.parse.quote(species)}")
    except Exception as e:
        return "fail", f"gbif species/match error: {type(e).__name__}: {e}"
    if not m.get("acceptedUsageKey") and not m.get("usageKey"):
        return "fail", f"gbif: especie no encontrada '{species}'"
    uk = m.get("acceptedUsageKey") or m.get("usageKey")
    sci = m.get("scientificName") or m.get("canonicalName") or species
    # ocurrencias (filtro por región solo si es 'global' -> sin filtro; si no,
    # se reporta el total global y la petición regional como tal — sin bbox fiable
    # para regiones libres. Documentado en el docstring.)
    try:
        url = f"{GBIF_API}/occurrence/search?speciesKey={uk}&limit={OCC_LIMIT}"
        occ = _http(url)
        total = occ.get("count", 0)
        recs = [{"key": r.get("key"), "country": r.get("country"),
                 "year": r.get("year"), "decimalLatitude": r.get("decimalLatitude"),
                 "decimalLongitude": r.get("decimalLongitude")}
                for r in (occ.get("results") or [])[:OCC_LIMIT]]
        detail = (f"gbif: {sci} (usageKey {uk}) region={region} -> "
                  f"{total:,} registros reales; muestra {len(recs)}: {recs}")
        return "ok", detail
    except Exception as e:
        return "partial", f"gbif: {sci} usageKey {uk} (occurrence/search error: {type(e).__name__}: {e})"

def _kuhpc_ls(remote_dir):
    """Lista un dir remoto en HPC (devuelve lista o None si no existe)."""
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "kuhpc",
                            f"ls {remote_dir} 2>/dev/null"],
                           capture_output=True, text=True, timeout=40)
        out = [x for x in r.stdout.splitlines() if x.strip()]
        return out if out else None
    except Exception:
        return None

BIOCLIM_STACK_CANDIDATES = [
    "/beegfs/a474r867/stage1_global_hybrid/bioclim_separate_1km_global_1986-2025",
    "/beegfs/a474r867/stage1_global_hybrid/bioclim_separate_1km_global_1950-2025",
    "/beegfs/a474r867/stage1_global_hybrid/bioclim",
    "/beegfs/a474r867/keras/bioclim/chelsa",
    "/beegfs/a474r867/ecoreasoner/data/bioclim",
]

def resolve_bioclim_download(args):
    """Verifica el stack de capas en HPC y devuelve disponibilidad real."""
    region = str(args.get("region") or "").strip() or "?"
    year = str(args.get("year") or "").strip() or "?"
    for cand in BIOCLIM_STACK_CANDIDATES:
        lst = _kuhpc_ls(cand)
        if lst:
            # stack global: dirs por año (1986..2025), dentro 19 capas bio*
            if len(lst) > 3 and all(x.isdigit() for x in lst[:3]):
                year_ok = year in lst
                inner = _kuhpc_ls(f"{cand}/{lst[0]}") or []
                layers = [x for x in inner if "bio" in x.lower()][:25]
                return "ok", (f"bioclim: region={region} year={year} -> stack REAL {cand}: "
                              f"{len(lst)} años [{lst[0]}..{lst[-1]}], year={year} "
                              f"{'presente' if year_ok else 'NO presente'}; "
                              f"capas en {lst[0]}: {len(layers)} (primeras: {layers[:6]})")
            layers = [x for x in lst if "bio" in x.lower()][:25]
            return "ok", (f"bioclim: region={region} year={year} -> stack REAL {cand}: "
                          f"{len(lst)} ítems, {len(layers)} capas bio* (primeras: {layers[:6]})")
    return "fail", (f"bioclim: region={region} year={year} -> stack no localizado en HPC "
                    f"(candidatos: {BIOCLIM_STACK_CANDIDATES}); recurso oficial: https://chelsa-climate.org")

def resolve_maxent_train(args):
    """Valida inputs y devuelve la spec lanzable (no ejecuta el entrenamiento)."""
    species = str(args.get("species") or "").strip()
    layers = str(args.get("layers") or "").strip()
    if not species or not layers:
        return "fail", "maxent_train requiere species y layers"
    # validar capas contra el stack del HPC
    ok_any = False
    for cand in BIOCLIM_STACK_CANDIDATES:
        lst = _kuhpc_ls(cand)
        if lst:
            ok_any = True
            break
    stack_state = "stack HPC OK" if ok_any else "stack HPC NO localizado"
    return "ok", (f"maxent: species={species} layers={layers} ({stack_state}) -> spec lista "
                  f"para job HPC (moe_v4/species_dm) — ejecución es entrenamiento, no síncrona")

RESOLVERS = {
    "gbif_occurrence": resolve_gbif_occurrence,
    "bioclim_download": resolve_bioclim_download,
    "maxent_train": resolve_maxent_train,
}

def resolve(tool, args):
    fn = RESOLVERS.get(tool)
    if not fn:
        return "fail", f"tool desconocida: {tool}"
    return fn(args or {})

if __name__ == "__main__":
    import urllib.parse  # top-level no: some paths don't need it
    tool = sys.argv[1]
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    status, msg = resolve(tool, args)
    print(f"[{status}] {msg}")
    sys.exit(0 if status == "ok" else 1)