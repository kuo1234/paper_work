"""Diagnostic: is the crop visually meaningful at all?
1) Self-consistency confusion: for each crop, score vs ALL candidate-name texts in that frame.
   diag_acc = crop X picks text X. Random ~ 1/n.
2) Try 3 crop sizes. 3) Dump a montage of one frame's crops for inspection."""
import os, json, numpy as np, re
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.camera_utils as CU
SIZE=256
def stem(n): return re.sub(r"_\d+$","",n)
META=Path.home()/"bts-poc/experiments/runs/libero_object_evidence_v0/metadata.jsonl"
rows=[json.loads(l) for l in META.read_text().splitlines() if l.strip()]
dev="cuda"
model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
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
for HALF in (22,34,50):
    diag=[]
    for r in rows:
        T=get_T(r["task_id"]); img=Image.open(imgdir/r["image"]).convert("RGB")
        names=[]; crops=[]
        for n,p in sorted(r["object_positions"].items()):
            x,y=proj(p,T); b=(max(0,x-HALF),max(0,y-HALF),min(SIZE,x+HALF),min(SIZE,y+HALF))
            crops.append(img.crop(b).resize((96,96))); names.append(stem(n))
        texts=[f"a photo of {nm.replace('_',' ')}" for nm in names]
        with torch.no_grad():
            inp=proc(text=texts,images=crops,return_tensors="pt",padding=True).to(dev)
            sims=model(**inp).logits_per_image.cpu().numpy()  # [n_img, n_txt]
        for i in range(len(names)):
            diag.append(int(np.argmax(sims[i])==i))
    print(f"HALF={HALF} crop={2*HALF}px  self_consistency_diag_acc={np.mean(diag):.3f}  n={len(diag)}")
# dump montage of frame 0 crops at HALF=34
r=rows[0]; T=get_T(r["task_id"]); img=Image.open(imgdir/r["image"]).convert("RGB")
HALF=34; mont=Image.new("RGB",(96*7,96+14),(0,0,0)); 
from PIL import ImageDraw; d=ImageDraw.Draw(mont); xo=0
for n,p in sorted(r["object_positions"].items()):
    x,y=proj(p,T); b=(max(0,x-HALF),max(0,y-HALF),min(SIZE,x+HALF),min(SIZE,y+HALF))
    c=img.crop(b).resize((96,96)); mont.paste(c,(xo,14)); d.text((xo,0),stem(n)[:12],fill=(255,255,0)); xo+=96
mont.save(Path.home()/"bts-poc/experiments/runs/_crop_montage.png")
print("montage saved")
