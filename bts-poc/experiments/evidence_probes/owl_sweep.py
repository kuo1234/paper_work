"""Tune OWLv2 evidence: per-candidate INDEPENDENT query (cleanest evidence def).
For each (frame, candidate): query that one name, take detection box nearest the candidate's
projected pixel -> record (dist, score). Post-process sweeps RAD (max dist to accept) and reports
top-1/top-2 target-selection per frame. Sweeps 3 prompt templates."""
import os, json, numpy as np, re
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
from PIL import Image
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.camera_utils as CU
SIZE=256
def stem(n): return re.sub(r"_\d+$","",n)
META=Path.home()/"bts-poc/experiments/runs/libero_object_evidence_v0/metadata.jsonl"
rows=[json.loads(l) for l in META.read_text().splitlines() if l.strip()]
dev="cuda"
model=Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
proc=Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
bm=benchmark.get_benchmark_dict()["libero_object"]()
Tc={}
def get_T(t):
    if t in Tc: return Tc[t]
    task=bm.get_task(t); bddl=os.path.join(get_libero_path("bddl_files"),task.problem_folder,task.bddl_file)
    e=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=SIZE,camera_widths=SIZE); e.seed(0); e.reset()
    for _ in range(10): e.step([0,0,0,0,0,0,-1])
    T=CU.get_camera_transform_matrix(e.env.sim,"agentview",SIZE,SIZE); e.close(); Tc[t]=T; return T
def proj(p,T):
    rc=CU.project_points_from_world_to_camera(np.asarray(p)[None,:],T,SIZE,SIZE)[0]
    return (SIZE-1-float(rc[1])),(SIZE-1-float(rc[0]))
imgdir=Path.home()/"bts-poc/experiments/runs/libero_object_evidence_v0"
TEMPLATES={"bare":"{n}","a":"a {n}","photo":"a photo of a {n}"}
ts=torch.tensor([[SIZE,SIZE]],device=dev)

# collect per (template, frame): for each candidate -> (dist_nearest_box, score_nearest_box)
collected={k:[] for k in TEMPLATES}
for r in rows:
    T=get_T(r["task_id"]); img=Image.open(imgdir/r["image"]).convert("RGB")
    target=stem(r["target_key_name"])
    cand=[(stem(n),proj(p,T)) for n,p in sorted(r["object_positions"].items())]
    cand=[(n,xy) for n,xy in cand if n!=stem(r["receptacle_key_name"])]
    names=[c[0] for c in cand]; pix=[c[1] for c in cand]
    for tkey,tmpl in TEMPLATES.items():
        rec=[]
        for ci,nm in enumerate(names):
            q=[[tmpl.format(n=nm.replace("_"," "))]]
            with torch.no_grad():
                inp=proc(text=q,images=img,return_tensors="pt").to(dev)
                out=model(**inp)
                det=proc.post_process_object_detection(out,target_sizes=ts,threshold=0.0)[0]
            bx=det["boxes"].cpu().numpy(); sc=det["scores"].cpu().numpy()
            x,y=pix[ci]
            if len(sc):
                cx=(bx[:,0]+bx[:,2])/2; cy=(bx[:,1]+bx[:,3])/2
                d=np.hypot(cx-x,cy-y); j=int(np.argmin(d))
                rec.append((float(d[j]),float(sc[j])))
            else: rec.append((1e9,0.0))
        collected[tkey].append({"task":r["task_id"],"names":names,"target":target,"rec":rec})

def eval_cfg(tkey,RAD):
    per_task={}; top1=[]; top2=[]
    for fr in collected[tkey]:
        sc=np.array([s if d<=RAD else -1.0 for (d,s) in fr["rec"]])
        order=np.argsort(-sc); ranked=[fr["names"][i] for i in order]
        h1=int(ranked[0]==fr["target"]); h2=int(fr["target"] in ranked[:2])
        top1.append(h1); top2.append(h2); per_task.setdefault(fr["task"],[]).append(h1)
    return np.mean(top1),np.mean(top2),per_task

best=None
print("template  RAD   top1  top2   fixed(t1/t6/t8)")
for tkey in TEMPLATES:
    for RAD in (20,30,45,9999):
        t1,t2,pt=eval_cfg(tkey,RAD)
        ff=[round(np.mean(pt.get(t,[0])),2) for t in (1,6,8)]
        print(f"{tkey:7s} {RAD:5d}  {t1:.2f}  {t2:.2f}   {ff}")
        score=(sum(1 for x in ff if x>=0.5),t1)  # prioritize fixed-failure coverage then overall
        if best is None or score>best[0]: best=(score,tkey,RAD,t1,t2,ff)
print("\nBEST:",json.dumps({"template":best[1],"RAD":best[2],"top1":round(best[3],3),"top2":round(best[4],3),"fixed":best[5]}))
