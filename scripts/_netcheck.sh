#!/bin/bash
apptainer exec --nv /beegfs/a474r867/ecoreasoner/pytorch-cuda.sif bash -lc '
python - <<PY
import urllib.request
try:
    r = urllib.request.urlopen("https://pypi.org/simple/torch/", timeout=25)
    print("NET_OK status", r.status)
except Exception as e:
    print("NET_FAIL", repr(e))
PY
'