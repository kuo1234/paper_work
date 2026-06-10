"""Fix the per-object score bias found in error attribution.
Idea: OWLv2 raw score(obj, name) has a strong per-object prior. The evidence we want is
"does the TARGET name match candidate C better than candidate C's OWN identity / than other names?"
Test 3 normalization schemes for selecting the target among candidates:

 raw:        s(c) = score(image, target_name) near c            [current, 0.40]
 selfnorm:   s(c) = score(c, target_name) - score(c, c_name)    [is target a better label for c than c itself?]
 raticontr:  s(c) = score(c, target_name) / (mean_k score(c, name_k))  [softmax-like over names at box c]

All per-candidate independent queries. Report top-1 + fixed failures."""
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
def score_at(img,query_name,xy):
    q=[[query_name.replace("_"," ")]]
    with torch.no_grad():
        inp=proc(text=q,images=img,return_tensors="pt").to(dev)
        det=proc.post_process_object_detection(model(**inp),target_sizes=ts,threshold=0.0)[0]
    bx=det["boxes"].cpu().numpy(); sc=det["scores"].cpu().numpy()
    if not len(sc): return 0.0
    cx=(bx[:,0]+bx[:,2])/2; cy=(bx[:,1]+bx[:,3])/2; d=np.hypot(cx-xy[0],cy-xy[1]); m=d<=RAD
    return float(sc[m].max()) if m.any() else 0.0

schemes={"raw":[],"selfnorm":[],"ratiocontr":[]}; ptask={k:{} for k in schemes}
for r in rows:
    T=get_T(r["task_id"]); img=Image.open(imgdir/r["image"]).convert("RGB")
    target=stem(r["target_key_name"])
    cand=[(stem(n),proj(p,T)) for n,p in sorted(r["object_positions"].items()) if stem(n)!=stem(r["receptacle_key_name"])]
    names=[c[0] for c in cand]
    # score matrix M[c][k] = score at box c using name k  (only need: target col + diagonal + all for ratio)
    s_target={c:score_at(img,target,xy) for c,xy in cand}
    s_self={c:score_at(img,c,xy) for c,xy in cand}
    s_all={c:{k:score_at(img,k,xy) for k in names} for c,xy in cand}
    raw={c:s_target[c] for c,_ in cand}
    selfn={c:s_target[c]-s_self[c] for c,_ in cand}
    ratio={c:s_target[c]/(np.mean(list(s_all[c].values()))+1e-6) for c,_ in cand}
    for key,sc in (("raw",raw),("selfnorm",selfn),("ratiocontr",ratio)):
        win=max(sc,key=sc.get); h=int(win==target)
        schemes[key].append(h); ptask[key].setdefault(r["task_id"],[]).append(h)
def rt(k,t): return round(np.mean(ptask[k].get(t,[0])),2)
print(json.dumps({k:{"top1":round(np.mean(v),3),
  "fixed_t1_t6_t8":[rt(k,1),rt(k,6),rt(k,8)]} for k,v in schemes.items()},indent=2))
