"""Decisive Phase-3 test with OWLv2 open-vocab detector.
For each frame: query = all candidate object names (natural text). Detector returns boxes+scores
per query. For each candidate, take max detection score whose box-center is nearest that
candidate's projected pixel (within radius). Rank candidates by score -> top-1/top-2 target hit.
This matches plan's evidence module: image+target name -> target_candidate + confidence."""
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
def nice(n): return stem(n).replace("_"," ")
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

# Two scoring modes:
# A) "detect_target_only": query just the target name, see if top box lands on target vs distractor
# B) "rank_all": query all names, assign each candidate the best box-score near its pixel
res_A=[]; res_B=[]; per_task_A={}; per_task_B={}
RAD=28
for r in rows:
    T=get_T(r["task_id"]); img=Image.open(imgdir/r["image"]).convert("RGB")
    target=stem(r["target_key_name"])
    cand=[(stem(n),proj(p,T)) for n,p in sorted(r["object_positions"].items())]
    cand=[(n,xy) for n,xy in cand if n!=stem(r["receptacle_key_name"])]  # drop basket
    names=[c[0] for c in cand]; pix=[c[1] for c in cand]
    # ---- mode A: target-only query ----
    q=[[f"a {nice(r['target_key_name'])}"]]
    with torch.no_grad():
        inp=proc(text=q,images=img,return_tensors="pt").to(dev)
        out=model(**inp)
        ts=torch.tensor([[SIZE,SIZE]],device=dev)
        det=proc.post_process_object_detection(out,target_sizes=ts,threshold=0.0)[0]
    boxes=det["boxes"].cpu().numpy(); scores=det["scores"].cpu().numpy()
    if len(scores):
        bi=int(np.argmax(scores)); bc=((boxes[bi][0]+boxes[bi][2])/2,(boxes[bi][1]+boxes[bi][3])/2)
        # which candidate is the top box nearest to?
        d=[np.hypot(bc[0]-x,bc[1]-y) for (x,y) in pix]
        hitA=int(names[int(np.argmin(d))]==target)
    else: hitA=0
    res_A.append(hitA); per_task_A.setdefault(r["task_id"],[]).append(hitA)
    # ---- mode B: all-names query, rank candidates ----
    q2=[[f"a {n.replace('_',' ')}" for n in names]]
    with torch.no_grad():
        inp=proc(text=q2,images=img,return_tensors="pt").to(dev)
        out=model(**inp)
        det=proc.post_process_object_detection(out,target_sizes=ts,threshold=0.0)[0]
    boxes=det["boxes"].cpu().numpy(); scores=det["scores"].cpu().numpy(); labels=det["labels"].cpu().numpy()
    cand_score=np.full(len(names),-1.0)
    for bi in range(len(scores)):
        lab=int(labels[bi]); bc=((boxes[bi][0]+boxes[bi][2])/2,(boxes[bi][1]+boxes[bi][3])/2)
        # box assigned to candidate `lab`; check its center near that candidate's pixel
        x,y=pix[lab]
        if np.hypot(bc[0]-x,bc[1]-y)<=RAD and scores[bi]>cand_score[lab]:
            cand_score[lab]=scores[bi]
    order=np.argsort(-cand_score); ranked=[names[i] for i in order]
    hitB=int(ranked[0]==target); res_B.append(hitB); per_task_B.setdefault(r["task_id"],[]).append(hitB)
def rt(d,t): return round(np.mean(d[t]),3)
print(json.dumps({
 "modeA_target_only_top1": round(np.mean(res_A),3),
 "modeB_rank_all_top1": round(np.mean(res_B),3),
 "fixed_failures_modeB":{"t1_cream":rt(per_task_B,1),"t6_butter":rt(per_task_B,6),"t8_choc":rt(per_task_B,8)},
 "fixed_failures_modeA":{"t1_cream":rt(per_task_A,1),"t6_butter":rt(per_task_A,6),"t8_choc":rt(per_task_A,8)},
 "per_task_modeB":{f"t{t}":rt(per_task_B,t) for t in range(10)},
}, indent=2))
