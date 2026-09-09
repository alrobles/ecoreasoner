import json
from collections import Counter
n=0; with_mesh=0; by_src=Counter()
for i,line in enumerate(open("/beegfs/a474r867/ecoreasoner/data/train_corpus_v5.jsonl")):
    if i>=30000: break
    try: d=json.loads(line)
    except: continue
    n+=1
    if d.get("mesh_terms"): with_mesh+=1
    by_src[d.get("fine_source","?")]+=1
print("muestra %d: con_mesh=%d (%.0f%%)" % (n, with_mesh, 100*with_mesh/n))
print("fine_source:", dict(by_src))
print("Ejemplos sdm:")
cnt=0
for line in open("/beegfs/a474r867/ecoreasoner/data/train_corpus_v5.jsonl"):
    try: d=json.loads(line)
    except: continue
    if d.get("domain_fine")=="sdm" and cnt<4:
        t=(d.get("text") or "").replace("\n"," ")[:110]
        print("  [%s] %s" % (d.get("pmid"), t))
        cnt+=1
    if cnt>=4: break
