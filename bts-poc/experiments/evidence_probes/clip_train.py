"""Decisive Phase-3 test: TRAIN a small head on CLIP features for candidate-vs-target.
Features per candidate: CLIP image embed (crop) + CLIP text embed (target name) + their cos.
Label: is_target. Eval: leave-one-task-out (10 folds). Report target-selection top-1 per frame
(argmax of predicted score among candidates) and the 3 fixed-failure tasks.
Compares against zero-shot CLIP and random."""
import os, json, numpy as np, re
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.camera_utils as CU
SIZE=256; HALF=40
def stem(n): return re.sub(r"_\d+$","",n)
META=Path.home()/"bts-poc/experiments/runs/libero_object_evidence_v0/metadata.jsonl"
rows=[json.loads(l) for l in META.read_text().splitlines() if l.strip()]
dev="cuda"
clip=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
proc=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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

# build features
samples=[]  # dict per frame: task_id, feats[n,D], labels[n], names
for r in rows:
    T=get_T(r["task_id"]); img=Image.open(imgdir/r["image"]).convert("RGB")
    target=stem(r["target_key_name"]); names=[]; crops=[]
    for n,p in sorted(r["object_positions"].items()):
        x,y=proj(p,T); b=(max(0,x-HALF),max(0,y-HALF),min(SIZE,x+HALF),min(SIZE,y+HALF))
        crops.append(img.crop(b).resize((96,96))); names.append(stem(n))
    with torch.no_grad():
        im_in=proc(images=crops,return_tensors="pt").to(dev)
        ie=clip.get_image_features(**im_in); ie=ie/ie.norm(dim=-1,keepdim=True)
        tx=proc(text=[f"a photo of {target.replace('_',' ')}"],return_tensors="pt",padding=True).to(dev)
        te=clip.get_text_features(**tx); te=te/te.norm(dim=-1,keepdim=True)
        te=te.repeat(len(names),1)
        cos=(ie*te).sum(-1,keepdim=True)
        feat=torch.cat([ie,te,cos],dim=-1).cpu().numpy()
    labels=np.array([int(nm==target) for nm in names])
    samples.append(dict(task=r["task_id"],feat=feat,lab=labels,names=names,target=target,cos=cos.cpu().numpy().ravel()))

D=samples[0]["feat"].shape[1]
def run_fold(test_task):
    tr=[s for s in samples if s["task"]!=test_task]; te=[s for s in samples if s["task"]==test_task]
    Xtr=np.concatenate([s["feat"] for s in tr]); ytr=np.concatenate([s["lab"] for s in tr])
    Xtr=torch.tensor(Xtr,dtype=torch.float32,device=dev); ytr=torch.tensor(ytr,dtype=torch.float32,device=dev)
    net=nn.Sequential(nn.Linear(D,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,1)).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    pos_w=torch.tensor([(ytr==0).sum()/max(1,(ytr==1).sum())],device=dev)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for ep in range(300):
        opt.zero_grad(); out=net(Xtr).squeeze(-1); l=lossf(out,ytr); l.backward(); opt.step()
    net.eval(); hits=[]
    with torch.no_grad():
        for s in te:
            sc=net(torch.tensor(s["feat"],dtype=torch.float32,device=dev)).squeeze(-1).cpu().numpy()
            pred=s["names"][int(np.argmax(sc))]; hits.append(int(pred==s["target"]))
    return hits

trained={}; zshot={}
for t in range(10):
    h=run_fold(t); trained[t]=np.mean(h)
for s in samples:
    zshot.setdefault(s["task"],[]).append(int(s["names"][int(np.argmax(s["cos"]))]==s["target"]))
zshot={k:np.mean(v) for k,v in zshot.items()}
all_trained=np.mean([trained[t] for t in range(10)])
all_zshot=np.mean([zshot[t] for t in range(10)])
print(json.dumps({
  "trained_head_top1_per_task":{f"t{t}":round(trained[t],3) for t in range(10)},
  "trained_head_top1_overall":round(all_trained,3),
  "zeroshot_top1_overall":round(all_zshot,3),
  "fixed_failures_trained":{"t1_cream":round(trained[1],3),"t6_butter":round(trained[6],3),"t8_choc":round(trained[8],3)},
  "fixed_failures_zeroshot":{"t1_cream":round(zshot[1],3),"t6_butter":round(zshot[6],3),"t8_choc":round(zshot[8],3)},
}, indent=2))
