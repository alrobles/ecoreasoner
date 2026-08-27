#!/usr/bin/env python3
"""
watchdog_v4teacher.py — Vigila el teacher v4 y la destilación.

Ciclo:
1. Detecta el job ollama-v4serve actual (squeue) y su nodo.
2. Extrae el puerto del endpoint (del output del job: "Port: N").
3. Comprueba que el endpoint responde /v1/models.
4. Si el teacher murió o el endpoint no responde: busca el NUEVO job v4serve
   (relanzado por cron), extrae su puerto, y registra el cambio.
5. Reporta estado (a pantalla/log) para que la destilación use el endpoint correcto.

Uso: python3 watchdog_v4teacher.py --interval 60  (loop infinito)

El job de destilación (distill-v4) se relanza/redirige con el nuevo endpoint.
"""
import subprocess, re, time, sys, json, urllib.request

SSH=["ssh","kuhpc"]
def run(cmd, timeout=40):
    r=subprocess.run(SSH+[cmd],capture_output=True,text=True,timeout=timeout)
    return r.stdout + r.stderr

def get_v4serve_job():
    out=run("squeue -u a474r867 --format='%.12i %.20j %.8T %.12N' --noheader")
    for line in out.splitlines():
        p=line.split()
        if len(p)>=4 and "v4serve" in line and p[2]=="RUNNING":
            return {"job":p[0],"node":p[3]}
    # puede estar PENDING
    for line in out.splitlines():
        p=line.split()
        if len(p)>=3 and "v4serve" in line:
            return {"job":p[0],"node":p[3] if len(p)>3 else "?"}
    return None

def get_port(job):
    # output file: v4flash-serve-output-<job>
    out=run(f"grep -a 'Port:' /home/a474r867/work/ollama/v4flash-serve-output-{job} 2>/dev/null | tail -1")
    m=re.search(r"Port:\s*(\d+)",out)
    return m.group(1) if m else None

def check_endpoint(node,port):
    if not node or not port: return False
    # consultar DESDE el cluster (el login node alcanza el nodo compute; local no)
    out=run(f"curl -s --max-time 10 http://{node}:{port}/v1/models 2>&1 | head -c 80", timeout=30)
    return "deepseek" in out or "model" in out.lower() and "error" not in out.lower()

def main():
    args=[a for a in sys.argv[1:] if a!="--interval"]
    interval=int(args[0]) if args else 60
    last_endpoint=None
    while True:
        job=get_v4serve_job()
        state=[]
        if job:
            port=get_port(job.get("job"))
            node=job.get("node")
            alive=check_endpoint(node,port) if (node and port) else False
            ep=f"http://{node}:{port}" if (node and port) else "?"
            state.append(f"v4serve job={job.get('job')} node={node} port={port} alive={alive}")
            if alive and ep!=last_endpoint:
                print(f"[watchdog] TEACHER V4: {ep}",flush=True)
                last_endpoint=ep
            if not alive:
                print(f"[watchdog] teacher no responde ({ep}) — esperando relanzado cron",flush=True)
        else:
            print("[watchdog] NO v4serve en cola — teacher caído/entre rotaciones",flush=True)
        # estado destilación
        dout=run("squeue -u a474r867 --format='%.12i %.20j %.8T' --noheader 2>/dev/null | grep -iE 'distill'")
        state.append("distill: "+ (dout.strip() or "no en cola"))
        print("[watchdog] "+ " | ".join(state),flush=True)
        time.sleep(interval)

if __name__=="__main__":
    main()