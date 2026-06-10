"""Error attribution for OWLv2 evidence at best cfg (bare template, per-candidate independent).
For each frame classify the failure:
 - HIT: target ranked #1
 - CONFUSED: target detected (score>0) but a distractor outscored it
 - MISSED: target got no detection near its pixel (score<=0 or no box in radius)
Also dump per-object detection score stats to see which objects are invisible to OWLv2."""
import os, json, numpy as np, re
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
from PIL import Image
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.camera_utils as CU
SIZE=256; RAD=45
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
ts=torch.tensor([[SIZE,SIZE]],device=dev)
def score_obj(img,name,xy):
    q=[[name.replace("_"," ")]]
    with torch.no_grad():
        inp=proc(text=q,images=img,return_tensors="pt").to(dev)
        det=proc.post_process_object_detection(model(**inp),target_sizes=ts,threshold=0.0)[0]
    bx=det["boxes"].cpu().numpy(); sc=det["scores"].cpu().numpy()
    if not len(sc): return 0.0
    cx=(bx[:,0]+bx[:,2])/2; cy=(bx[:,1]+bx[:,3])/2; d=np.hypot(cx-xy[0],cy-xy[1])
    m=d<=RAD
    return float(sc[m].max()) if m.any() else 0.0
cats={"HIT":0,"CONFUSED":0,"MISSED":0}; per_obj={}; detail=[]
for r in rows:
    T=get_T(r["task_id"]); img=Image.open(imgdir/r["image"]).convert("RGB")
    target=stem(r["target_key_name"])
    cand=[(stem(n),proj(p,T)) for n,p in sorted(r["object_positions"].items()) if stem(n)!=stem(r["receptacle_key_name"])]
    sc={nm:score_obj(img,nm,xy) for nm,xy in cand}
    for nm in sc: per_obj.setdefault(nm,[]).append(sc[nm])
    ts_=sc[target]; best=max(sc.values()); win=max(sc,key=sc.get)
    if win==target and ts_>0: c="HIT"
    elif ts_<=1e-6: c="MISSED"
    else: c="CONFUSED"
    cats[c]+=1
    if c!="HIT": detail.append({"task":r["task_id"],"init":r["init_id"],"target":target,"cat":c,"t_score":round(ts_,3),"winner":win,"win_score":round(best,3)})
print(json.dumps({"categories":cats,
 "per_object_mean_score":{k:round(np.mean(v),3) for k,v in sorted(per_obj.items())},
 "per_object_detect_rate(>0)":{k:round(np.mean([x>0 for x in v]),2) for k,v in sorted(per_obj.items())}},indent=2))
print("=== non-hit detail (first 20) ==="); [print(json.dumps(d)) for d in detail[:20]]
