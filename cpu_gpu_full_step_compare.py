import os, sys, time
import numpy as np
import torch
sys.path.insert(0, "/workspace/bts/3d_diffuser_actor")

from transformers import CLIPTokenizer, CLIPTextModel
from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
from utils.common_utils import get_gripper_loc_bounds
from online_evaluation_calvin.evaluate_policy import make_env, get_sequences
from online_evaluation_calvin.evaluate_utils import prepare_visual_states, prepare_proprio_states, get_env_state_for_initial_condition, convert_action
from utils.utils_with_calvin import relative_to_absolute
import diffusers.schedulers.scheduling_ddpm as scheduling_ddpm

REPO = "/workspace/bts/3d_diffuser_actor"
DS = "/workspace/bts/calvin/dataset/task_ABC_D"
CKPT = f"{REPO}/train_logs/diffuser_actor_calvin.pth"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

env = make_env(DS, show_gui=False)
initial_state, eval_sequence = get_sequences(1)[0]
robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
obs = env.get_obs()
obs = prepare_visual_states(obs, env)
obs = prepare_proprio_states(obs, env)
lang = "turn on the led"
print("eval_sequence:", eval_sequence)

rgbs_np = np.stack([obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"]], axis=0).transpose(0,3,1,2)[:, :, 20:180, 20:180]
pcds_np = np.stack([obs["pcd_obs"]["pcd_static"], obs["pcd_obs"]["pcd_gripper"]], axis=0).transpose(0,3,1,2)[:, :, 20:180, 20:180]
proprio_np = obs["proprio"]

tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()
tokens = torch.tensor(tok(lang + ".", padding="max_length")["input_ids"], device="cuda").view(1,-1)
with torch.no_grad():
    lang_cuda = text_model(tokens).last_hidden_state.float()
lang_cpu = lang_cuda.cpu()
del text_model

glb = get_gripper_loc_bounds(f"{REPO}/tasks/calvin_rel_traj_location_bounds_task_ABC_D.json", task=None, buffer=0.01)

def make_policy(device):
    m = DiffuserActor(backbone="clip", image_size=(256,256), embedding_dim=192, num_vis_ins_attn_layers=2, use_instruction=True, fps_subsampling_factor=3, gripper_loc_bounds=glb, rotation_parametrization="6D", quaternion_format="wxyz", diffusion_timesteps=25, nhist=3, relative=True, lang_enhanced=True)
    sd = torch.load(CKPT, map_location="cpu")["weight"]
    m.load_state_dict({k[7:]: v for k,v in sd.items()})
    return m.eval().to(device)

g = torch.Generator(device="cpu").manual_seed(12345)
fixed_init = torch.randn((1,19,9), generator=g)
fixed_pos_vars = [torch.randn((1,19,3), generator=g) for _ in range(24)]
fixed_rot_vars = [torch.randn((1,19,6), generator=g) for _ in range(24)]

class FixedNoise:
    def __enter__(self):
        self.pos_i=0; self.rot_i=0; self.init_used=False
        self.orig_torch_randn=torch.randn; self.orig_sched_randn=scheduling_ddpm.randn_tensor
        def fake_torch_randn(*args, **kwargs):
            size=kwargs.get("size", args[0] if args else None); shape=tuple(size) if size is not None else None
            if shape == tuple(fixed_init.shape) and not self.init_used:
                self.init_used=True; return fixed_init.to(device=kwargs.get("device",None), dtype=kwargs.get("dtype",torch.float32))
            return self.orig_torch_randn(*args, **kwargs)
        def fake_sched_randn(shape, generator=None, device=None, dtype=None, layout=None):
            shape=tuple(shape)
            if shape == tuple(fixed_pos_vars[0].shape):
                out=fixed_pos_vars[self.pos_i]; self.pos_i+=1; return out.to(device=device, dtype=dtype or torch.float32)
            if shape == tuple(fixed_rot_vars[0].shape):
                out=fixed_rot_vars[self.rot_i]; self.rot_i+=1; return out.to(device=device, dtype=dtype or torch.float32)
            return self.orig_sched_randn(shape, generator=generator, device=device, dtype=dtype, layout=layout)
        torch.randn=fake_torch_randn; scheduling_ddpm.randn_tensor=fake_sched_randn; return self
    def __exit__(self,*args):
        torch.randn=self.orig_torch_randn; scheduling_ddpm.randn_tensor=self.orig_sched_randn
        print("noise calls:", self.init_used, self.pos_i, self.rot_i)

def run(device, lang_emb):
    m=make_policy(device)
    rgbs=torch.as_tensor(rgbs_np,device=device).unsqueeze(0).float()
    pcds=torch.as_tensor(pcds_np,device=device).unsqueeze(0).float()
    grip=torch.as_tensor(proprio_np,device=device).unsqueeze(0).float()
    fake=torch.zeros((1,19,7),device=device); mask=torch.full((1,19),False,device=device)
    with torch.no_grad(), FixedNoise(), torch.cuda.amp.autocast(enabled=False):
        raw=m(fake,mask,rgbs,pcds,lang_emb.to(device),curr_gripper=grip[...,:7],run_inference=True)
    torch.cuda.synchronize() if device=="cuda" else None
    post_traj=convert_action(raw)
    post_grip=convert_action(grip[:,[-1],:])
    abs_action=relative_to_absolute(post_traj, post_grip)
    return raw.detach().cpu(), torch.as_tensor(abs_action).detach().cpu()

print("=== GPU ===")
raw_g, abs_g = run("cuda", lang_cuda)
print("=== CPU ===")
raw_c, abs_c = run("cpu", lang_cpu)

def report(name,a,b):
    d=(a-b).abs(); rel=d/(b.abs()+1e-6)
    print(f"=== {name} ===")
    print("max_abs", float(d.max()), "mean_abs", float(d.mean()), "p95_abs", float(torch.quantile(d.flatten(),0.95)))
    print("max_rel", float(rel.max()), "mean_rel", float(rel.mean()))
    print("first gpu", np.round(a[0,0].numpy(),6))
    print("first cpu", np.round(b[0,0].numpy(),6))
    print("per_dim", [round(float(d[...,i].max()),9) for i in range(d.shape[-1])])
report("RAW", raw_g, raw_c)
report("ABS_ACTION", abs_g, abs_c)
