---
type: research-inquiry
aliases:
  - "M1b-6 external inquiry for GPT"
  - "3D-DA CALVIN low score external debug question"
tags:
  - 研究主線
  - CALVIN
  - 3D-Diffuser-Actor
  - eval
  - debug
  - 外部詢問
summary: "給 GPT/其他模型/外部研究者查詢用：3D-Diffuser-Actor released CALVIN checkpoint 在 GB10 上 ABC→D eval 遠低於論文。列出環境、結果、已排除項、剩餘假設與精確詢問。"
---

# 外部詢問文件：3D-Diffuser-Actor CALVIN ABC→D eval 分數極低，請協助查 root cause

> 目的：把本地/遠端已做過的 debug 全部濃縮成一份 **self-contained prompt**。可以直接貼給 GPT / 其他 LLM / repo maintainer / 有經驗的人查詢。  
> 專案：3D Diffuser Actor (`nickgkan/3d_diffuser_actor`) on CALVIN ABC→D online eval。  
> 問題：released CALVIN checkpoint 在我的 GB10/aarch64 環境中 online eval 分數約 **0.57 avg seq len**，遠低於論文約 **2.5–2.8**。

---

## 1. 我想請你幫忙回答什麼？

請協助判斷：

1. **3D-Diffuser-Actor public released CALVIN checkpoint (`diffuser_actor_calvin.pth` / `diffuser_actor_calvin_nohistory.pth`) 的可重現 expected score 到底是多少？**  
   - README 沒有列精確數字。
   - 論文數字約 avg seq len 2.48–2.8，但不確定是否對應 public checkpoint + public script。

2. **若 public checkpoint 應該接近論文分數，請找出我這個環境為何只有 avg 0.57 的最可能原因。**  
   我已經排除很多項（見下文），請不要重複建議已排除的 flag/checkpoint/FPS/precision 基本檢查，除非你能指出我排除方式有漏洞。

3. **是否有已知的 3D-Diffuser-Actor / CALVIN / pybullet / diffusers / transformers / Python 版本相容性問題，會造成 online eval 分數極低但不 crash？**  
   尤其請查 GitHub issues、README、commit history、成功復現環境。

4. **在只有這台 GB10 遠端機器、沒有第二張 x86 CUDA GPU 的條件下，下一個最有資訊量的 debug 實驗是什麼？**

---

## 2. 結果摘要

### checkpoint / eval 設定

- repo: `nickgkan/3d_diffuser_actor`
- branch: local patch branch `bts-gb10-calvin-smoke`
- base upstream commit: `4faf00b update model for calvin`
- local compatibility patch commit: `0c6685b fix(calvin): make eval smoke work on GB10 aarch64`
- checkpoint:
  - `diffuser_actor_calvin.pth` = README listed old / with history
  - `diffuser_actor_calvin_nohistory.pth` = README listed new / no history
- dataset:
  - CALVIN `task_ABC_D/validation`
  - extracted by HTTP Range partial extraction from official `task_ABC_D.zip`
  - validation dir has metadata, `.hydra`, language annotations, statistics
- online eval:
  - `online_evaluation_calvin/evaluate_policy.py`
  - `NUM_SEQUENCES=100` for main measurement
  - 1 GPU, `WORLD_SIZE=1`, sequence indices 0..99

### observed score

For `diffuser_actor_calvin.pth` old w/history, `num_history=3`:

```text
100-seq ABC→D validation:
1/5 : 36.0%
2/5 : 13.0%
3/5 :  4.0%
4/5 :  3.0%
5/5 :  1.0%
avg seq len = 0.57
```

Historical small runs:

```text
old 10-seq:        40 / 30 / 10 / 10 / 0
old 50-seq:        48 / 22 / 14 /  8 / 2
old 100-seq:       36 / 13 /  4 /  3 / 1
no-history 50-seq: 48 / 20 / 12 /  6 / 4
```

Important interpretation:

- The failure is **not just chaining**. Task-1 itself is only ~36–48%, while paper-level first-task success is reportedly ~90%+.
- Both old and no-history checkpoints are similarly low.

---

## 3. Exact eval flags used

These were compared against `scripts/test_trajectory_calvin.sh` and match official script except GPU count / sequence count / absolute paths:

```text
--calvin_dataset_path /workspace/bts/calvin/dataset/task_ABC_D
--calvin_model_path /workspace/bts/calvin/calvin_models
--text_encoder clip
--text_max_length 16
--tasks A B C D
--backbone clip
--gripper_loc_bounds tasks/calvin_rel_traj_location_bounds_task_ABC_D.json
--gripper_loc_bounds_buffer 0.01
--calvin_gripper_loc_bounds /workspace/bts/calvin/dataset/task_ABC_D/validation/statistics.yaml
--embedding_dim 192
--action_dim 7
--use_instruction 1
--rotation_parametrization 6D
--diffusion_timesteps 25
--interpolation_length 20
--num_history 3              # old checkpoint
--relative_action 1
--fps_subsampling_factor 3
--lang_enhanced 1
--save_video 0
--quaternion_format wxyz
--checkpoint train_logs/diffuser_actor_calvin.pth
```

For no-history checkpoint, only `--num_history 1` and checkpoint path changed, matching `scripts/train_trajectory_calvin_nohistory.sh` eval block.

---

## 4. Environment

Remote machine:

```text
host: p76141495@192.168.65.11
machine: spark-3994 / GB10 / aarch64 / NVIDIA GB10 / PCIe
container: bts_m1
repo path: /workspace/bts/3d_diffuser_actor
CALVIN path: /workspace/bts/calvin
```

Relevant packages / commits:

```text
torch:        2.7.0a0+79aa17489c.nv25.04
CUDA stack:   NVIDIA / GB10 / container NGC-like nv25.04
python:       3.12
numpy:        1.26.4
diffusers:    0.37.1
transformers: 4.44.2
dgl:          2.1.0
pybullet:     3.2.7 (aarch64 wheel/build)
gym:          0.26.2
hydra-core:   1.3.2
omegaconf:    2.3.0
calvin_env:   editable git commit 1431a46 by default
calvin_env main/origin/main: 797142c
calvin_models: editable git commit fa03f01
```

Important: README says `cd calvin_env; git checkout main`. I tested that too (see below).

---

## 5. Local patches needed just to run on GB10/aarch64

Patch commit `0c6685b` contains:

1. Camera selection by class name instead of assuming `env.cameras[0]=static`, `env.cameras[1]=gripper`.
2. DGL farthest point sampler fallback: aarch64 DGL wheel has CPU-only FPS; code moves FPS input to CPU, runs DGL FPS, moves indices back to GPU.

Site-packages shims:

```text
/usr/local/lib/python3.12/dist-packages/pyhash.py
  minimal FNV1-32 shim for CALVIN deterministic seeding
/usr/local/lib/python3.12/dist-packages/dgl/graphbolt/__init__.py
  graphbolt load try/except; warning only
```

---

## 6. What I already ruled out

Please do not suggest these again unless you identify a flaw in the evidence.

### 6.1 Flags mismatch — ruled out

Flags were compared one-by-one against `scripts/test_trajectory_calvin.sh`; no missing eval flag found. GPU count differs (1 GPU vs official 6 GPU) but `WORLD_SIZE=1` sequence slicing is clean.

### 6.2 Checkpoint load — ruled out

I instantiated `DiffuserActor` with the eval flags and loaded checkpoint manually:

```text
MISSING_KEYS: 0
UNEXPECTED_KEYS: 0
SHAPE_MISMATCH: 0
```

Checkpoint raw keys all start with `module.`, and eval code strips first 7 chars correctly. `load_state_dict` uses strict=True in normal eval and would raise on mismatch.

### 6.3 Partial validation missing files — ruled out for eval behavior

Checks:

- `lang_annotations/auto_lang_ann.npy` exists
- `lang_annotations/embeddings.npy` exists
- `.hydra/merged_config.yaml`, `statistics.yaml`, `ep_start_end_ids.npy` exist
- 1087 language-task windows referenced by `auto_lang_ann.npy`: **0 missing frame**

Important discovery: online eval initial states come from `online_evaluation_calvin/multistep_sequences.py::get_sequences()`, not from validation `.npz` frames. The validation data mostly supplies env config/statistics/language metadata.

### 6.4 FPS dtype / CPU fallback — largely ruled out

Facts:

- Upstream original code already does `.to(torch.float64)` before DGL FPS. My patch did **not** introduce float64; it only moved input to CPU and output indices back to GPU.
- On this aarch64 DGL, CUDA FPS is unavailable:
  ```text
  DGLError: Operator FarthestPointSampler does not support cuda device.
  ```
- CPU FPS deterministic.
- CPU fp64 vs fp32 FPS index overlap = 1.000 for tested shapes.
- DGL CPU FPS vs pure Python canonical FPS: exact index equality for fp32/fp64 and multiple shapes.
- Reverse rollout test with `fps_subsampling_factor=1` did **not** improve:
  ```text
  factor=3, N=20: avg 0.95
  factor=1, N=20: avg 0.55
  ```

Therefore FPS is unlikely, though I still cannot directly compare DGL CUDA FPS on a reference x86 GPU.

### 6.5 GB10 EGL rendering / point cloud — mostly ruled out

One-step geometry probe:

- live depth finite and plausible
- point cloud extent plausible
- `proprio` shape is `(3,8)` = pos3 + wxyz quaternion4 + gripper1
- `gripper[..., :7]` is pos + quaternion, as expected

Stored frame vs live render:

- Reset env to validation `.npz` `robot_obs` / `scene_obs`
- For episode 0/1/40/219635, live RGB/depth almost matches stored frame pixel/depth:
  - static RGB mean_abs roughly 0.8–1.4 for many opening frames
  - gripper RGB mean_abs ~0.001 for opening frames
  - depth mean_abs ~1e-4 or lower
- Some mid-trajectory frames differ more, possibly because `.npz` state is not a full dynamic simulator snapshot. But evidence does not support a global render/assets/camera bug.

### 6.6 Precision / TF32 / autocast — ruled out

Same 20 sequences:

```text
base fp16+TF32:   sum=19, avg=0.95
fp32+TF32-off:    sum=18, avg=0.90
```

Patch used:

```python
with torch.cuda.amp.autocast(enabled=False):
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
```

No recovery.

### 6.7 CPU vs GPU model inference — ruled out

This is a strong test.

Same real eval obs, same checkpoint, same CLIP embedding, same fixed diffusion initial noise and per-step variance noise. Compare CPU vs GPU(fp32/TF32-off):

```text
raw policy trajectory:
  max_abs  = 9.6e-8
  mean_abs = 2.1e-8

full eval postprocess absolute actions
(quat→Euler + relative_to_absolute, includes pytorch3d transform):
  max_abs = 1.9e-7
```

Therefore Blackwell/aarch64 forward kernels / attention / pytorch3d / DGL gather are almost certainly not silently wrong for model inference.

### 6.8 diffusers scheduler / clip_sample — tested, not the fix

Current diffusers is 0.37.1. Repo does not pin diffusers.

I tested `clip_sample=False` because DDPMScheduler default clips predicted x0 to [-1,1]. 20-seq looked better, but 100-seq disproved it:

```text
original 100-seq, clip_sample=True:  avg 0.57
noclip 100-seq, clip_sample=False:   avg 0.45
```

So `clip_sample=False` is not the solution.

Also checked diffusers 0.30.3 scheduler config vs 0.37.1:

- timesteps [24..0] identical
- betas / alphas_cumprod identical
- clip_sample=True identical
- timestep_spacing=leading identical

Trying diffusers 0.27.2 fails under modern `huggingface_hub` because `cached_download` was removed. I did not downgrade hub because that risks breaking transformers/CLIP.

### 6.9 pyhash shim — ruled out

Custom `pyhash.fnv1_32` shim matches canonical FNV-1 32-bit vectors:

```text
'a'      -> 0x050c5d7e
'foobar' -> 0x31f0b262
```

Even if scene shuffle differed, it would mainly swap valid block positions; task oracle uses actual sim state. Not likely a global score killer.

### 6.10 `calvin_env` commit mismatch — ruled out for first 20

README says checkout `calvin_env main`. My env was detached `1431a46`; main is `797142c` and contains relevant-looking fixes:

```text
27a27a2 fix bug with actions being modified after step
6e7ceaa fix bug with wrong object sizes after changing a scene
5a0eb8a add control in joint space
797142c fix bug in button during rollouts
```

I temporarily checked out `calvin_env main`, ran the exact same first 20 sequence eval, then restored `1431a46`.

Result: **identical per-sequence output** to base run:

```text
0 0
1 5
2 0
3 0
4 1
...
19 0
sum=19/20, avg=0.95
```

So this commit difference is not the cause.

### 6.11 Language token length / truncation — ruled out

Actual eval wrapper sets `tokenizer.model_max_length=16`. Verified:

```text
tokenizer.model_max_length = 16
embedding shape = (1, 16, 512)
```

All 34 `new_playtable_validation.yaml` annotations tokenize to length 16 under current CLIP tokenizer. No >16 long-instruction issue.

Note: validation `auto_lang_ann.npy` contains `language.emb` shape `(1087,1,384)` — CALVIN’s own embedding, not 3D-DA’s CLIP embedding. The repo’s `instructions/calvin_task_ABC_D/` directory is absent in my clone, so I cannot compare training-time precomputed CLIP embeddings.

---

## 7. External clue found

GitHub issue [nickgkan/3d_diffuser_actor#102](https://github.com/nickgkan/3d_diffuser_actor/issues/102) appears highly relevant.

Summary:

- User reports low/error CALVIN result.
- Maintainer says the pasted eval code looks correct and suggests trying no-history checkpoint.
- User says no-history also does not work.
- User later says: experiments on V100 server kept failing, but after reconfiguring environment on 4090 workstation, they successfully reproduced results.

This is very similar to my symptom: old and no-history checkpoints both low, code/flags appear correct.

Question: what exactly differed between the V100 server and 4090 workstation? GPU? pybullet? Python? torch? CALVIN env commit? diffusers/transformers? dataset packaging? The issue does not say.

---

## 8. Remaining hypotheses

Given all above, the remaining plausible causes seem to be:

### Hypothesis A — public checkpoint / public eval expected score is lower than paper

Maybe `diffuser_actor_calvin.pth` public checkpoint does not reproduce the paper number, and README’s “no-history better” still does not imply paper-level score in this setup.

But issue #102 suggests someone did reproduce after environment change, so this is uncertain.

Please find any reliable source for expected public checkpoint score.

### Hypothesis B — pybullet / physics / control dynamics version issue

`pybullet==3.2.7` on Python 3.12 / aarch64 is likely very different from author’s original environment. Model inference is correct, but closed-loop rollout could fail due to physics/control differences.

Need know: what pybullet version did successful reproductions use? Did CALVIN require a specific pybullet/gym/Python combo?

### Hypothesis C — CALVIN packaged dataset / env config mismatch

Training script references `./data/calvin/packaged_ABC_D`, while online eval script references `calvin/dataset/task_ABC_D`. I extracted only validation from official zip, not full packaged dataset.

Eval initial states are generated by code, but env config/assets/statistics still come from validation `.hydra` and stats. Maybe packaged validation contains subtly different config/statistics? Need verify expected `.hydra/merged_config.yaml` / `statistics.yaml` for task_ABC_D validation.

### Hypothesis D — language embeddings / CLIP version mismatch

Online eval uses current `transformers==4.44.2` CLIP to encode 34 validation annotations live. If the checkpoint was trained with embeddings from a different CLIP/tokenizer version, language grounding could degrade.

However openai/clip-vit-base-patch32 should be stable, and token length is correct. Need know author’s transformers version and whether they used precomputed CLIP embeddings in eval or live encoding.

### Hypothesis E — still some hidden high-level rollout bug

Could be task oracle, action scaling, env.step API, gripper convention, control mode, max velocity/force, physics timestep/action_repeat.

I verified source-level many things, but not yet a qualitative action-path comparison against a known-good trajectory.

---

## 9. Precise asks for you

Please research / reason about these concrete questions:

1. **What exact software environment is known to reproduce 3D-Diffuser-Actor CALVIN ABC→D public checkpoints?**  
   Please look for Python version, CUDA/torch, pybullet, gym, calvin_env commit, calvin_models commit, diffusers, transformers.

2. **What score should `diffuser_actor_calvin.pth` and `diffuser_actor_calvin_nohistory.pth` get under the public script?**  
   Need exact 1/5..5/5 and avg seq len if possible. If only paper number exists, clarify whether it is public checkpoint or retrained/internal checkpoint.

3. **Does CALVIN online eval depend sensitively on pybullet version / Python version / gym version?**  
   Are there known issues where pybullet 3.2.7 or Python 3.12 causes poor closed-loop performance without crashing?

4. **Are there known changes in `calvin_env` / `calvin_models` after 2024 that affect task oracle or rollout behavior?**  
   I tested `calvin_env main` vs `1431a46` for first 20 sequences and got identical result, but maybe there are other commits/branches/tags expected by 3D-DA.

5. **Is my partial validation extraction sufficient for online eval?**  
   Since initial states come from `get_sequences()`, I think yes, but please verify whether online eval reads anything else from the validation dataset besides `.hydra/merged_config.yaml`, `statistics.yaml`, and language annotations.

6. **Could live CLIP embedding with `transformers==4.44.2` differ enough from training-time CLIP to destroy language grounding?**  
   If yes, what version should be pinned?

7. **Given only this GB10 machine, what next experiment would maximally reduce uncertainty?**  
   My current candidate: dump failure/success 19-waypoint absolute action paths and judge whether model output is semantically reasonable; or create a known scripted/expert action sanity rollout to test env/oracle.

---

## 10. Useful concrete remote artifacts / scripts

On remote container `bts_m1`:

```text
/workspace/bts/3d_diffuser_actor                 repo
/workspace/bts/calvin/dataset/task_ABC_D         validation data
/workspace/bts/eval_abc100_logs/result.txt       original 100-seq result (sum 57)
/workspace/bts/eval_noclip100_logs/result.txt    clip_sample=False 100-seq result (sum 45)
/workspace/bts/eval_fpsexp_base3_logs/result.txt first 20 base factor=3 result (sum 19)
/workspace/bts/eval_fpsexp_fps1_logs/result.txt  first 20 fps_factor=1 result (sum 11)
/workspace/bts/eval_fp32_run1_logs/result.txt    first 20 fp32 result (sum 18)
/workspace/bts/eval_fpsexp_envmain20_logs/result.txt first 20 calvin_env main result (sum 19)
```

Diagnostic scripts:

```text
/tmp/check_load.py                     checkpoint load keys
/tmp/check_val.py                      validation lang-window completeness
/tmp/fps_probe.py                      FPS dtype determinism
/tmp/check_fps_canonical.py            DGL CPU FPS vs canonical FPS
/tmp/geom_probe.py                     live depth/pcd/proprio sanity
/tmp/cmp_depth.py                      stored depth stats
/tmp/compare_stored_live_frame.py      reset-to-npz live vs stored RGB/depth
/tmp/run_eval_fpsexp.sh                parametric eval launcher <N> <fps_factor> <tag>
/tmp/run_eval_fp32.sh                  fp32/TF32-off eval
/tmp/run_eval_noclip.sh                clip_sample=False eval
/tmp/run_eval_envmain.sh               temporarily checkout calvin_env main and eval
/tmp/sched_probe.py                    DDPMScheduler config/timesteps probe
/tmp/pyhash_check.py                   pyhash FNV-1 test vectors
/tmp/fnv_variants.py                   FNV-1 vs FNV-1a check
/tmp/cpu_gpu_step_compare.py           CPU vs GPU raw model.step compare
/tmp/cpu_gpu_full_step_compare.py      CPU vs GPU full postprocess compare
```

---

## 11. My current best conclusion

After all tests, I no longer think the main issue is:

- checkpoint loading
- eval flags
- FPS CPU fallback
- GB10 fp16/TF32 precision
- Blackwell GPU model forward kernel
- DGL/PyTorch3D device mismatch
- missing validation files
- camera/depth/point cloud
- calvin_env main vs detached commit
- text length/truncation

The failure is now most likely either:

1. **a high-level CALVIN/pybullet/control/task-oracle environment mismatch**, or
2. **a public-checkpoint reproducibility issue where expected public score is lower/unclear**, or
3. **a remaining hidden dependency/version mismatch not yet identified**.

Please focus your research on these remaining areas.
