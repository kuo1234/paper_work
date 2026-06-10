# BTS PoC — Model & Experiment Card (for external review)

> **Purpose**: self-contained technical brief so another model/researcher can understand the architecture, environment, training, evaluation, and the current blocker — and advise on **how to get a positive P3 result in this toy** before scaling to CALVIN. No codebase access needed.
> Last updated: 2026-06-05 (v7 + three diagnosis rounds + BFS bug fix).

---

## 1. Research goal (one paragraph)

We want to liberate meta-RL's *task-belief inference* from "must interact at test time" into "interaction-free multi-modal specification". A **spec acts as a prior**, **observations act as likelihood**, and the policy should behave **risk-aware** w.r.t. the belief (this is "BTS"). The toy PoC must demonstrate three phenomena to justify scaling to CALVIN:

- **P1**: ambiguous spec → higher initial belief entropy (prior works).
- **P2**: observations arrive → belief entropy decreases (likelihood works).
- **P3**: belief / risk-aware policy **beats** a single-point (T2DA-style argmax) baseline (belief has **decision value**).

**Status: P1, P2 robust across 5 seeds. P3 has never passed.** This card is about P3.

---

## 2. Environment — `AmbiguousSpecGridWorld`

A 7×7 grid. Agent starts at center (3,3). Goal: step onto a target object defined by a (color, shape) task.

- **Objects**: 4 objects placed at random empty cells. Each has `color ∈ {red,blue,green}`, `shape ∈ {square,circle,triangle}`. The 4 objects = 2 colors × 2 shapes (Cartesian), so ambiguity is meaningful (same-color-different-shape and vice versa always exist).
- **Tasks**: `K = 9` possible tasks (`{color}_{shape}`). One is the episode target.
- **Spec** `c`: a language-attribute vector. Can be *exact* (`"go to the red square"` → 1 task) or *ambiguous* (`"go to a square"` → multiple compatible tasks). Spec mode is "mixed" (≈65% ambiguous when ambiguity is available).
- **Hints (the belief-narrowing channel)**: 2 hint tiles on the grid. Stepping on hint1 reveals **one** target attribute (e.g. color); hint2 reveals the **complementary** attribute (shape). Revealed 0 → ~4 candidate tasks; 1 → ~2 (2-peak); 2 → 1 (converged). **Hints, not object identities, are what narrow the task belief.**
- **Actions**: 5 discrete = `{up, right, down, left, stay}` (grid-clamped at edges).
- **Horizon**: 20 steps.
- **Reward**: `step = −0.01`, `success(on target) = +1.0 (done)`, `wrong_object(step onto any non-target object) = −1.0 (done)`. The −1.0 makes "guessing wrong" costly — meant to give risk-aware belief an edge.
- **Observability flag** `observe_object_identity` (v7): `True` (default) = object color/shape visible from t=0 (fully-observable, = v6). `False` = **partial-obs**: an object's color/shape is masked (zeros) until the agent has been adjacent (Manhattan ≤ 1, sticky); object **positions are always visible**. Intended to make belief un-bypassable.

**Oracle posterior** `p*(z | c, o_{1:t})`: computed from spec + revealed hints + agent-on-target. Used as the belief supervision target. It does **not** depend on object-identity visibility (so P1/P2 targets are identical across obs modes).

**Expert** (for offline data): BFS shortest path. Ambiguous spec → `start → hint1 → hint2 → target`; exact → straight to target.

---

## 3. Data — offline behavior-cloning dataset

`data/generate.py` rolls expert episodes and stores per-step records (jsonl): `obs_vec`, `action` (expert label), `oracle_posterior`, plus spec/hints/objects metadata. Train/test split = **held-out unseen task** (`green_triangle` goes to test; this is the unseen-task generalization axis).

Vector layouts (exact):
- `spec_vec` dim = **8**: `color one-hot(3) + color_is_null(1) + shape one-hot(3) + shape_is_null(1)`.
- `obs_vec` dim = **56 (fully-obs)** / **60 (partial-obs)**:
  - agent pos (2, normalized) + hint1 block (11) + hint2 block (11) + 4 objects.
  - each hint block (11): pos(2) + revealed(1) + revealed-color one-hot(3) + revealed-shape one-hot(3) + kind(2).
  - each object: fully-obs = pos(2)+color(3)+shape(3) = 8; partial-obs = pos(2)+color(3)+shape(3)+seen(1) = 9 (identity zeroed until revealed).
- Coordinates are **normalized floats in [0,1]** (`r/6, c/6`). There is **no relative agent→target vector** in the obs — the policy must infer direction from absolute coords.

---

## 4. Model — `models/transformer.py`

A small in-context transformer shared by both the belief model and the baseline.

**Backbone `TinyBTS`**:
- Input sequence = `[spec_token] ++ [history tokens]`, causal mask. Each history token = `obs_vec ++ prev_action_onehot(5)`, linearly projected to `d_model`.
- Transformer: `d_model=128, heads=4, layers=4, dim_feedforward=256, norm_first=True, dropout=0.1`. Sinusoidal positional encoding.
- `ctx` = representation of the **last valid token**.
- **belief_head**: MLP `ctx → belief_logits[K=9]` → softmax = `belief_probs`.
- **policy_head (belief model)**: MLP on `[ctx ++ belief_probs]` → `action_logits[5]`.

**`SinglePointBaseline`** (T2DA-style single point; v7 ctx-bottleneck variant):
- Reuses the same backbone + belief_head.
- `point_idx = argmax(belief_probs)` → `point_onehot[K]`.
- **point_policy**: MLP on `[nav_feat ++ point_onehot]` → `action_logits[5]`, where `nav_feat = last raw obs token` (positions visible; identity masked to 0 in partial-obs). **It does NOT consume `ctx`** — so task identity can only reach the policy via the belief argmax, making the belief channel the sole task-identity bottleneck (clean "distribution vs single-point" contrast).

---

## 5. Training — `train.py`

- **Pure offline behavior cloning + belief supervision.** No RL, no reward in the loss.
- Per-timestep teacher forcing: for each `t`, feed ground-truth history `[:t+1]`, predict `a_t` and `b_t`.
- Loss = `action_CE + λ · belief_KL`, `λ = 0.5`. `action_CE` = cross-entropy to expert action; `belief_KL` = KL(belief_logits ‖ oracle_posterior).
- AdamW lr=3e-4, batch 256, 15 epochs, grad clip 1.0. obs_dim inferred from data (so dim changes flow through automatically).
- **The −1.0 wrong-object penalty is eval-only — it never enters the loss.**

---

## 6. Evaluation — `eval/phenomena.py`

- **P1**: belief entropy at t=0, ambiguous vs exact specs (teacher-forced).
- **P2**: belief entropy over context timesteps on ambiguous episodes (per-episode drop + linear slope).
- **P3**: **free rollout** (model picks argmax action, env steps, no teacher forcing) of belief model vs single baseline on **ambiguous** episodes; compares `success_rate` and `avg_return`. **`return_gap = belief_return − single_return` is the headline metric.**

---

## 7. Results

### P1/P2 (robust, 5 seeds)
P1 gap ≈ `+1.0` (ambiguous entropy ≫ exact). P2 slope < 0, per-episode drop ≈ `+0.5`. Both stable.

### P3 (never passes)
v6 (fully-obs): dual `return_gap = −0.014 ± 0.072` (≈0, sign-flipping noise).
v7 (partial-obs + ctx bottleneck): `dual_partial return_gap = −0.117 ± 0.079` (5 seeds **all negative**); `dual_full = −0.100 ± 0.082`. Pre-registered pass criterion (`mean − 1σ > 0`) **failed**.

### Root-cause diagnosis (four hypotheses, each refuted by a cheap test, last one confirmed)
1. "belief reaches target but inefficiently" → **refuted**: belief model's losses come from charging into wrong objects (55%), not detours.
2. "multi-modal collapse to mean action" → **refuted**: even EXACT specs (zero multi-modality) rollout-fail.
3. "BC covariate shift; fix with DAgger" → **refuted**: DAgger-lite (relabel deviated states, noise 0.3) did **not** help (17.5%→14.0%).
4. **"the policy simply can't learn navigation"** → **confirmed**: teacher-forced 1-step action accuracy is only **~50% on SEEN tasks** (and ~50% on unseen — so not a generalization issue).

### A real bug found & fixed (but not the main cause)
The expert BFS did **not** treat non-target objects as obstacles, so shortest paths often passed **through** wrong-object cells → 33.7% of "expert" trajectories self-terminated on a wrong object (−1.0). **Fixed** (BFS now routes around them): expert success 66%→97.7%. **But retraining on clean data barely moved TF accuracy** (EXACT 50.6% seen / 48.8% unseen; AMBIGUOUS ~69%). So the bug was real and worth fixing, yet **not** the cause of the 50% ceiling.

### Key anomaly for advisors to chew on
- **EXACT TF acc (50%) < AMBIGUOUS TF acc (69%)** — counterintuitive (exact should be easier). EXACT trajectories are very short (mean length 3.6 vs 13.3 ambiguous), first-step distribution is heavily skewed (fixed center start + BFS tie-break `[up,right,down,left]` order → first-step up 624 vs left 138).
- A **lenient** accuracy ("predicted action ∈ set of distance-reducing optimal actions", on-policy rollout) is still only **23.8%** on EXACT → the model genuinely walks the wrong way / takes detours; it's not just choosing an equivalent shortest path that tie-break didn't pick.

---

## 8. Our current interpretation

The toy is **not too simple**; it **couples two orthogonal difficulties** onto one tiny policy:
1. **belief inference** (what P1/P2/P3 want to test), and
2. **learning spatial navigation from absolute normalized coordinates** (direction = f(agent_pos, target_pos) is hard for a 4-layer tiny transformer).

Difficulty (2) **drowns out** the belief signal in P3. Evidence: P1/P2 (which need no rollout / no navigation) are robust; P3 (which needs a policy that can navigate) collapses. So P3 can't be observed until navigation is learnable.

---

## 9. The question we want advice on

**How to get a positive P3 (belief model's free-rollout return > single-point baseline) in this toy, cheaply, without just moving to CALVIN yet?**

Candidate directions we're considering (advisors: please critique / rank / propose better):
- **(A) Make navigation learnable by changing the obs representation** — e.g. add explicit relative vectors (agent→each object, agent→each hint) so direction becomes a lookup rather than spatial reasoning from absolute coords. Smallest change.
- **(B) Decouple navigation from belief entirely** — e.g. give the policy a hand-coded "go-to(target_xy)" controller and let belief only decide *which* target; then P3 tests belief's decision value without a navigation-learning confound.
- **(C) Stronger policy class** — bigger net, different action parameterization, or a non-greedy decoder. (Note: sampling helped the single baseline but not the belief model; multi-modal heads don't help EXACT.)
- **(D) Put the −1.0 risk into the training loss** (currently eval-only) so the policy is actually trained to be risk-aware under wide belief. (We deferred this because navigation崩壊 dominates; but maybe it interacts.)
- **(E) Reconsider whether the toy's "decision" should even require navigation** — perhaps the action space should directly be "commit to task z" so belief's value is tested without locomotion.

Specific things we'd value an outside opinion on:
1. Is the **coupling diagnosis** (navigation-learning swamps belief signal) correct, or is there a simpler explanation we missed?
2. Among (A)–(E), which is the **minimal change** most likely to yield a clean, *interpretable* positive P3 — and which risk making P3 a *trivial/unfair* win that wouldn't transfer to CALVIN?
3. Is BC-from-shortest-path-expert fundamentally the wrong supervision for testing "belief decision value", regardless of representation?
4. Any concern that fixing the toy to pass P3 would **overfit the conclusion** (i.e. we'd show belief helps only because we engineered it to)?

---

## 10. Files (if code access is later granted)
- `envs/gridworld.py` — env, spec, oracle posterior, BFS expert (now obstacle-aware), vectorizers.
- `data/generate.py` — offline dataset (now supports `--observe-object-identity` / `--dagger-noise`).
- `models/transformer.py` — `TinyBTS` + `SinglePointBaseline`.
- `train.py` — BC + belief-KL.
- `eval/phenomena.py` — P1/P2/P3 metrics + rollout.
- `experiments/diagnose_p3.py` — rollout-outcome decomposition, greedy-vs-sample, TF-accuracy probes.
- `experiments/REPORT.md` — full honest narrative (§6.5–6.11 cover v7 + diagnoses + this analysis).
