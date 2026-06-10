"""Decisive test: zero-shot CLIP candidate-vs-target ranking on LIBERO-Object evidence frames.
For each frame: project every object to pixels, crop a patch, CLIP-score crop vs target name,
rank candidates. Report top-1/top-2 target-hit rate. Random baseline ~ 1/n_candidates.
Excludes basket (receptacle) optionally; reports both."""
import os, json, numpy as np
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.camera_utils as CU

SIZE=256; CROP=44  # half-size of crop box in px
META=Path.home()/"bts-poc/experiments/runs/libero_object_evidence_v0/metadata.jsonl"
rows=[json.loads(l) for l in META.read_text().splitlines() if l.strip()]

dev="cuda" if torch.cuda.is_available() else "cpu"
model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
proc=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def stem(n): 
    import re; return re.sub(r"_\d+$","",n)

# cache env per task to get camera matrix + projection (positions come from metadata)
bm=benchmark.get_benchmark_dict()["libero_object"]()
Tcache={}
def get_T(task_id):
    if task_id in Tcache: return Tcache[task_id]
    task=bm.get_task(task_id)
    bddl=os.path.join(get_libero_path("bddl_files"),task.problem_folder,task.bddl_file)
    env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=SIZE,camera_widths=SIZE)
    env.seed(0); env.reset()
    for _ in range(10): env.step([0,0,0,0,0,0,-1])
    T=CU.get_camera_transform_matrix(env.env.sim,"agentview",SIZE,SIZE)
    env.close(); Tcache[task_id]=T; return T

def proj(p,T):
    rc=CU.project_points_from_world_to_camera(np.asarray(p)[None,:],T,SIZE,SIZE)[0]
    row,col=float(rc[0]),float(rc[1])
    return (SIZE-1-col),(SIZE-1-row)  # x,y after 180 flip

imgdir=Path.home()/"bts-poc/experiments/runs/libero_object_evidence_v0"
stats={"all":[], "no_receptacle":[]}
per_task={}
for r in rows:
    T=get_T(r["task_id"])
    img=Image.open(imgdir/r["image"]).convert("RGB")
    target=stem(r["target_key_name"])
    cands=[(n,p) for n,p in sorted(r["object_positions"].items())]
    crops=[]; names=[]
    for n,p in cands:
        x,y=proj(p,T)
        box=(max(0,x-CROP),max(0,y-CROP),min(SIZE,x+CROP),min(SIZE,y+CROP))
        crops.append(img.crop(box).resize((96,96))); names.append(stem(n))
    text=f"a photo of {target.replace('_',' ')}"
    with torch.no_grad():
        inp=proc(text=[text],images=crops,return_tensors="pt",padding=True).to(dev)
        out=model(**inp)
        sims=out.logits_per_image.squeeze(1).cpu().numpy()  # [n_cands]
    order=np.argsort(-sims)
    ranked=[names[i] for i in order]
    top1=ranked[0]==target
    top2=target in ranked[:2]
    stats["all"].append((top1,top2))
    # variant excluding basket/receptacle as candidate
    keep=[i for i in range(len(names)) if names[i]!=stem(r["receptacle_key_name"])]
    sims2=sims[keep]; names2=[names[i] for i in keep]
    order2=np.argsort(-sims2); ranked2=[names2[i] for i in order2]
    stats["no_receptacle"].append((ranked2[0]==target, target in ranked2[:2]))
    per_task.setdefault(r["task_id"],[]).append(ranked2[0]==target)

def rate(lst,idx): return sum(x[idx] for x in lst)/len(lst)
print(json.dumps({
  "n_frames":len(rows),
  "all_candidates": {"top1":rate(stats["all"],0),"top2":rate(stats["all"],1)},
  "no_receptacle":  {"top1":rate(stats["no_receptacle"],0),"top2":rate(stats["no_receptacle"],1)},
  "fixed_failure_tasks_top1_no_recep": {
     "task1_cream_cheese":rate([s for r,s in zip(rows,stats["no_receptacle"]) if r["task_id"]==1],0),
     "task6_butter":rate([s for r,s in zip(rows,stats["no_receptacle"]) if r["task_id"]==6],0),
     "task8_choc_pudding":rate([s for r,s in zip(rows,stats["no_receptacle"]) if r["task_id"]==8],0),
  },
}, indent=2))
