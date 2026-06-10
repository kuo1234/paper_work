#!/usr/bin/env python3
"""E0 / P3-oracle-decision: 不訓練、不用神經網路，純 oracle 測「環境是否有 decision value」。

核心問題：在這個 toy 裡，用 FULL belief distribution 做決策，是否真的比用 argmax 單點更好？
若 oracle 都贏不了 → 環境沒有 decision value，後面 neural P3 不可能成立（該改環境）。
若 oracle 明顯贏 → decision value 存在，neural P3 失敗可歸因於 learning/representation。

設計（對齊外部建議，避免「我設計它贏」）：
- 三個 agent 共用同一套「最大化 belief 下期望 return」的決策規則與同一個 BFS low-level controller。
- 唯一差異 = 決策時用的 belief：
  - BELIEF  : 完整後驗分布 b(z)
  - SINGLE  : argmax 塌成單點 (T2DA 式)
  - SINGLE+H: argmax 單點 + entropy 純量（stronger baseline，E6）
- 決策時機：在「資訊不足」時，agent 可選 macro-action = 去收集 hint（縮窄 belief）或直接賭某個物件。
  期望 return 由 oracle simulator 算（賭錯 −1.0、賭對 +1.0−步數成本、收集 hint 多走幾步但降低賭錯機率）。
- 全程用 env 真實 step 結算實際 return / success（不是用估計值報結果）。

決策規則（expected return maximization，對三個 agent相同）：
  對每個候選 macro-action m，算 EV_b(m) = Σ_z b(z) · return(m | true task = z)
  選 argmax_m EV_b(m)，執行其第一段路徑；到達 subgoal（hint 或物件）後若 episode 未結束則重新決策。
  - BELIEF 用完整 b；SINGLE 用塌成 argmax 的 b（pmf 全壓在單一 task）；SINGLE+H 同 SINGLE 但決策時可用 entropy 當「要不要先收集 hint」的旋鈕。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import math
import random

from envs.gridworld import (
    AmbiguousSpecGridWorld, ALL_TASKS, TASK_TO_IDX,
    STEP_REWARD, SUCCESS_REWARD, WRONG_OBJECT_REWARD,
)


def belief_from_oracle(env, spec):
    """t=0 的 oracle posterior（list over ALL_TASKS），只在當前盤面物件上有質量。"""
    return env.oracle_posterior(spec, env.agent_pos)


def collapse_argmax(b):
    """把 belief 塌成 argmax 單點（其餘為 0）。"""
    out = [0.0] * len(b)
    if max(b) <= 0:
        return out
    out[b.index(max(b))] = 1.0
    return out


def entropy(b):
    s = sum(b)
    if s <= 0:
        return 0.0
    return -sum((p / s) * math.log(p / s + 1e-12) for p in b if p > 0)


def path_len(env, start, goal, blocked):
    acts = env.shortest_path_actions(start, list(goal) if isinstance(goal, (set, list)) else [goal], blocked=blocked)
    return len(acts), acts


def object_pos_for_task(env, task):
    for o in env.objects:
        if o.task_id == task:
            return o.pos
    return None


def expected_return_go_object(env, b, obj_task, from_pos, extra_steps=0):
    """macro: 直接走到 obj_task 的物件。EV = Σ_z b(z)·return。
    走到該物件後：若 z==obj_task → success(+1)；否則 → wrong_object(−1)。沿途 step 成本。"""
    blocked = {o.pos for o in env.objects if o.task_id != obj_task}  # 繞過其他物件
    goal = object_pos_for_task(env, obj_task)
    if goal is None:
        return -1e9
    steps, _ = path_len(env, from_pos, goal, blocked)
    steps += extra_steps
    step_cost = STEP_REWARD * steps
    ev = 0.0
    bsum = sum(b) or 1.0
    for ti, p in enumerate(b):
        if p <= 0:
            continue
        pz = p / bsum
        z = ALL_TASKS[ti]
        # 走到 obj_task 物件格：對 true task z 而言，這格是 success iff z==obj_task
        outcome = SUCCESS_REWARD if z == obj_task else WRONG_OBJECT_REWARD
        ev += pz * (outcome + step_cost)
    return ev


def collect_hints_then_best(env, b, spec, from_pos):
    """macro: 先去收集尚未揭露的 hint（縮窄 belief），再賭最佳物件。
    回傳 (EV, 第一步要去的 subgoal_pos, kind)。EV 用『收集後 belief 會收斂』的事實估算：
    收集 hint 後，belief 會塌到與 true task 相容的子集；保守用『收集後能正確鎖定 → success − 路徑成本』。"""
    # 找尚未揭露的 hint
    hints = []
    if env.hint1_pos is not None and not env.hint1_revealed:
        hints.append(("hint1", env.hint1_pos))
    if env.hint2_pos is not None and not env.hint2_revealed:
        hints.append(("hint2", env.hint2_pos))
    if not hints:
        return -1e9, None, None
    blocked = {o.pos for o in env.objects}  # 去 hint 路上繞過所有物件
    # 估算：去最近的 hint，路徑成本 + 之後從 hint 走到 target 的期望成本
    best = None
    for kind, hpos in hints:
        steps_to_hint, _ = path_len(env, from_pos, hpos, blocked)
        # 收集後 belief 收斂：保守估計能鎖定 true task（因為 hint 揭露真屬性）。
        # 之後從 hint 走到 target 的期望步數：用 belief 加權各候選物件距離。
        bsum = sum(b) or 1.0
        exp_steps_after = 0.0
        for ti, p in enumerate(b):
            if p <= 0:
                continue
            pz = p / bsum
            opos = object_pos_for_task(env, ALL_TASKS[ti])
            if opos is None:
                continue
            blk = {o.pos for o in env.objects if o.task_id != ALL_TASKS[ti]}
            s2, _ = path_len(env, hpos, opos, blk)
            exp_steps_after += pz * s2
        total_steps = steps_to_hint + exp_steps_after
        # 收集兩個 hint 後鎖定 → success；成本為總步數
        ev = SUCCESS_REWARD + STEP_REWARD * total_steps
        if best is None or ev > best[0]:
            best = (ev, hpos, kind)
    return best if best is not None else (-1e9, None, None)


def decide_macro(env, b, spec, use_entropy=False):
    """回傳要前往的 subgoal_pos（hint 或物件格）。
    比較『直接賭最佳物件』vs『先收集 hint』的 EV，選高者的第一段。"""
    candidates = [t for t in ALL_TASKS if b[TASK_TO_IDX[t]] > 0 and object_pos_for_task(env, t) is not None]
    # 直接賭：對每個候選物件算 EV，取最佳
    best_bet_ev, best_bet_pos = -1e9, None
    for t in candidates:
        ev = expected_return_go_object(env, b, t, env.agent_pos)
        if ev > best_bet_ev:
            best_bet_ev, best_bet_pos = ev, object_pos_for_task(env, t)
    # 收集 hint
    collect_ev, collect_pos, _ = collect_hints_then_best(env, b, spec, env.agent_pos)
    if collect_pos is not None and collect_ev > best_bet_ev:
        return collect_pos
    return best_bet_pos


# ============================================================================
# Pre-commit 路徑（--pre-commit）：把反例 counterexample_p3.py 的「時限壓力 +
# 揭曉前預先 commit」機制移植進真實 env，用真實 env.step 結算。
# 不改 env 動態；機制純在決策迴圈複製（target 身分在 t<t_reveal 時不可知）。
# 公平版 single+H 只能吃 (argmax_pos, entropy 純量)，看不到 mass 形狀。
# ============================================================================
FAIL_REWARD = 0.0  # 揭曉後剩餘步數不足走到 target = timeout 失敗，得 0（不是賭錯）


def build_precommit_problem(env, spec):
    """從 t=0 oracle posterior 取候選任務、mass、目標物件位置、起點。
    回 (cands, masses, targets, start)，僅含盤面上有物件且 mass>0 的候選。"""
    b = belief_from_oracle(env, spec)
    cands, masses, targets = [], [], []
    for t in ALL_TASKS:
        p = b[TASK_TO_IDX[t]]
        if p <= 0:
            continue
        opos = object_pos_for_task(env, t)
        if opos is None:
            continue
        cands.append(t)
        masses.append(p)
        targets.append(opos)
    s = sum(masses) or 1.0
    masses = [m / s for m in masses]  # 重新正規化（只留有物件的候選）
    return cands, masses, targets, env.start_pos


def _bfs_len(env, start, goal, blocked):
    acts = env.shortest_path_actions(start, [goal], blocked=blocked)
    # shortest_path_actions：start==goal 回 []；無路時退化（見 env 實作）
    return len(acts) if (acts or start == goal) else 10 ** 6


def _make_env(objs, snap, n_hints, horizon):
    env = AmbiguousSpecGridWorld(seed=0, n_hints=n_hints, horizon=horizon,
                                 observe_object_identity=True)
    env.set_episode(objects=objs, **snap)
    return env


def settle_ev(objs, snap, cands, masses, p, t_reveal, budget, n_hints):
    """EV = 對每個候選 z 當真 task，用『與最終結算完全相同』的 run_episode_precommit
    跑真實 env.step，再 mass 加權其 return。**杜絕 EV/結算背離**——
    這把『預移路徑撞 wrong object 提前死』『揭曉後步數不足 timeout』全自然算進去。"""
    horizon = t_reveal + budget
    ev = 0.0
    for i, z in enumerate(cands):
        env = _make_env(objs, snap, n_hints, horizon)
        _, ret = run_episode_precommit(env, p, z, t_reveal, budget)
        ev += masses[i] * ret
    return ev


def _all_cells(env):
    return [(r, c) for r in range(env.size) for c in range(env.size)]


def _reachable_cells(objs, snap, start, t_reveal, n_hints, budget):
    """揭曉前可達的 pre-commit 候選格（d_start<=t_reveal，繞過所有物件）。"""
    env = _make_env(objs, snap, n_hints, t_reveal + budget)
    all_obj = {o.pos for o in env.objects}
    out = []
    for p in _all_cells(env):
        if _bfs_len(env, start, p, blocked=all_obj - {p}) <= t_reveal:
            out.append(p)
    return out


def best_precommit(objs, snap, cands, masses, start, t_reveal, budget, n_hints):
    """belief：看完整 mass → 全盤搜真實結算 EV 最大的 pre-commit 點。"""
    best_p, best_ev = start, -1e18
    for p in _reachable_cells(objs, snap, start, t_reveal, n_hints, budget):
        ev = settle_ev(objs, snap, cands, masses, p, t_reveal, budget, n_hints)
        if ev > best_ev:
            best_p, best_ev = p, ev
    return best_p


def act_single_precommit(objs, snap, cands, masses, targets, start, t_reveal, budget, n_hints):
    """single：mass 全壓 argmax 候選 → 在 reachable 內選『假設真 task=argmax』結算 EV 最高的點。
    （只用 argmax 候選結算，看不到其他 mass。）"""
    ai = max(range(len(masses)), key=lambda i: masses[i])
    best_p, best_ev = start, -1e18
    for p in _reachable_cells(objs, snap, start, t_reveal, n_hints, budget):
        ev = settle_ev(objs, snap, [cands[ai]], [1.0], p, t_reveal, budget, n_hints)
        if ev > best_ev:
            best_p, best_ev = p, ev
    return best_p


def build_single_h_precommit_policy(boards, t_reveal, budget, n_hints):
    """公平版 single+H：決策只能吃 (argmax_pos, round(H,6))。
    後見之明最大優勢：每特徵桶選桶內『平均真實結算 EV』最高的單格（reachable 交集）。
    boards = list of (objs, snap, cands, masses, targets, start)。
    同桶不同盤面必用同一格 → 純量瓶頸代價（看不到 mass 落哪群）。"""
    H_DEC = 6

    def feat(masses, targets):
        ai = max(range(len(masses)), key=lambda i: masses[i])
        return (targets[ai], round(entropy(masses), H_DEC))

    bucket = {}
    for bd in boards:
        objs, snap, cands, masses, targets, start = bd
        bucket.setdefault(feat(masses, targets), []).append(bd)

    chosen = {}
    for key, items in bucket.items():
        # 候選格 = 桶內各盤面 reachable 格的『交集』（single+H 須對全桶用同一格）
        reach_sets = [set(_reachable_cells(o, s, st, t_reveal, n_hints, budget))
                      for (o, s, c, m, t, st) in items]
        common = set.intersection(*reach_sets) if reach_sets else set()
        best_pt, best_avg = None, -1e18
        for pt in common:
            tot = sum(settle_ev(o, s, c, m, pt, t_reveal, budget, n_hints)
                      for (o, s, c, m, t, st) in items)
            avg = tot / len(items)
            if avg > best_avg:
                best_pt, best_avg = pt, avg
        chosen[key] = best_pt if best_pt is not None else items[0][5]  # fallback: start

    def policy(masses, targets):
        return chosen[feat(masses, targets)]

    return policy, feat


def run_episode_precommit(env, p_star, true_task, t_reveal, budget):
    """兩階段：揭曉前走向 p_star（最多 t_reveal 步，不知 target）；揭曉後衝真 target。
    env.horizon 已設為 t_reveal+budget，timeout=真失敗。用真實 env.step 結算。"""
    total_return = 0.0
    all_obj = {o.pos for o in env.objects}
    # 階段1：揭曉前預移
    if p_star is not None and p_star != env.agent_pos:
        acts = env.shortest_path_actions(env.agent_pos, [p_star], blocked=all_obj - {p_star})
        for a in acts[:t_reveal]:
            _, r, done, info = env.step(a)
            total_return += r
            if done:  # 揭曉前就踩到物件 → env 自然結算（可能 wrong/success）
                return info["success"], total_return
    # 階段2：揭曉後衝真 target
    tgt = object_pos_for_task(env, true_task)
    blk = {o.pos for o in env.objects if o.pos != tgt}
    while True:
        if env.agent_pos == tgt:
            a = 4
        else:
            acts = env.shortest_path_actions(env.agent_pos, [tgt], blocked=blk)
            a = acts[0] if acts else 4
        _, r, done, info = env.step(a)
        total_return += r
        if done:
            return info["success"], total_return


def run_precommit(args):
    """--pre-commit 主流程：兩 pass（先建 single+H 桶 policy，再跑三 agent）。
    支援 --budget-sweep 掃 budget∈{2..8} 印表格。"""
    budgets = list(range(2, 9)) if args.budget_sweep else [args.budget]
    t_reveal = args.t_reveal

    # 先凍結一批歧義盤面（三 agent 共用同盤面，公平對照）
    base = AmbiguousSpecGridWorld(seed=args.seed, n_hints=args.n_hints,
                                  horizon=t_reveal + max(budgets), observe_object_identity=True)
    frozen = []
    for ep in range(args.n_episodes):
        base.reset()
        task = base.target_task
        spec = base.sample_spec_for_task(task, mode=args.spec_mode)
        if len(base.candidate_tasks_from_spec(spec)) <= 1:
            continue
        frozen.append((list(base.objects), task, spec,
                       dict(target_task=task,
                            hint1_pos=base.hint1_pos, hint1_attr_kind=base.hint1_attr_kind,
                            hint1_attr_value=base.hint1_attr_value,
                            hint2_pos=base.hint2_pos, hint2_attr_kind=base.hint2_attr_kind,
                            hint2_attr_value=base.hint2_attr_value)))

    print("=" * 72)
    print(f"E0 pre-commit  (spec={args.spec_mode}, n_hints={args.n_hints}, seed={args.seed}, "
          f"t_reveal={t_reveal}, n_amb_ep={len(frozen)})")
    print("=" * 72)
    print(f"  {'budget':>6} | {'belief':>9} {'single':>9} {'single+H':>9} | {'b−s+H':>8} {'b−single':>9}")
    print("  " + "-" * 62)

    pos_budgets = []
    for budget in budgets:
        horizon = t_reveal + budget
        # 建每盤面的 problem（用 t=0 oracle posterior 取候選/mass/目標）
        problems, boards = [], []
        for (objs, task, spec, snap) in frozen:
            env = _make_env(objs, snap, args.n_hints, horizon)
            cands, masses, targets, start = build_precommit_problem(env, spec)
            if not cands:
                continue
            problems.append((objs, task, spec, snap, cands, masses, targets, start))
            boards.append((objs, snap, cands, masses, targets, start))
        sh_policy, _ = build_single_h_precommit_policy(boards, t_reveal, budget, args.n_hints)

        agg = {m: {"succ": 0, "ret": 0.0, "n": 0} for m in ("belief", "single", "single_h")}
        for (objs, task, spec, snap, cands, masses, targets, start) in problems:
            # 三 agent 各自選 p*（全部用真實結算 EV，杜絕 EV/結算背離）
            p_belief = best_precommit(objs, snap, cands, masses, start, t_reveal, budget, args.n_hints)
            p_single = act_single_precommit(objs, snap, cands, masses, targets, start,
                                            t_reveal, budget, args.n_hints)
            p_single_h = sh_policy(masses, targets)
            for mode, p_star in (("belief", p_belief), ("single", p_single), ("single_h", p_single_h)):
                env = _make_env(objs, snap, args.n_hints, horizon)
                succ, ret = run_episode_precommit(env, p_star, task, t_reveal, budget)
                agg[mode]["succ"] += int(succ)
                agg[mode]["ret"] += ret
                agg[mode]["n"] += 1

        n = max(agg["belief"]["n"], 1)
        rb = agg["belief"]["ret"] / n
        rs = agg["single"]["ret"] / n
        rh = agg["single_h"]["ret"] / n
        gap_h = rb - rh
        if gap_h > 1e-6:
            pos_budgets.append(budget)
        print(f"  {budget:>6} | {rb:>+9.4f} {rs:>+9.4f} {rh:>+9.4f} | {gap_h:>+8.4f} {rb-rs:>+9.4f}")

    print("  " + "-" * 62)
    print(f"  belief − single+H > 0 的 budget：{pos_budgets}")
    print("\n判讀：belief 應在緊預算嚴格 > 公平版 single+H（分布形狀有價值）。")
    print("      真實 env 中 gap 隨 budget 放寬而衰減但未必歸零——因 wrong-object 致命格")
    print("      讓『每盤面客製避雷』的價值持續存在（比抽象反例更持久）。")
    print("      gap>0 在連續預算帶成立 = 機制移植成功；只在孤立 budget = artifact。")


def run_episode(env, spec, belief_mode):
    """belief_mode ∈ {'belief','single','single_h'}。回傳 (success, total_return)。"""
    total_return = 0.0
    for _ in range(env.horizon):
        b_raw = belief_from_oracle(env, spec)
        if belief_mode == "belief":
            b = b_raw
        elif belief_mode == "single":
            b = collapse_argmax(b_raw)
        elif belief_mode == "single_h":
            # 用 entropy 當旋鈕：高 entropy 時允許先收集 hint，否則用 argmax 直賭
            b = b_raw if entropy(b_raw) > 0.5 else collapse_argmax(b_raw)
        else:
            b = b_raw
        subgoal = decide_macro(env, b, spec, use_entropy=(belief_mode == "single_h"))
        if subgoal is None:
            _, r, done, info = env.step(4)  # stay
            total_return += r
            if done:
                return info["success"], total_return
            continue
        # 走一步往 subgoal（每步重新決策 → 揭露 hint 後 belief 會更新）
        blocked = {o.pos for o in env.objects if o.pos != subgoal}
        acts = env.shortest_path_actions(env.agent_pos, [subgoal], blocked=blocked)
        a = acts[0] if acts else 4
        _, r, done, info = env.step(a)
        total_return += r
        if done:
            return info["success"], total_return
    return False, total_return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-hints", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--spec-mode", type=str, default="ambiguous", choices=["ambiguous", "mixed"])
    ap.add_argument("--pre-commit", action="store_true",
                    help="啟用時限壓力 + 揭曉前預先 commit 路徑（移植自 counterexample_p3.py）")
    ap.add_argument("--t-reveal", type=int, default=3, help="target 身分揭曉前可預移的步數")
    ap.add_argument("--budget", type=int, default=3, help="揭曉後剩餘步數預算")
    ap.add_argument("--budget-sweep", action="store_true", help="掃 budget∈{2..8} 印表格")
    args = ap.parse_args()

    if args.pre_commit:
        run_precommit(args)
        return

    modes = ["belief", "single", "single_h"]
    agg = {m: {"succ": 0, "ret": 0.0, "n": 0} for m in modes}

    # 每個 episode 用同一盤面跑三個 agent（公平對照）
    base = AmbiguousSpecGridWorld(seed=args.seed, n_hints=args.n_hints, horizon=args.horizon,
                                  observe_object_identity=True)
    rng = random.Random(args.seed)
    for ep in range(args.n_episodes):
        base.reset()
        task = base.target_task
        spec = base.sample_spec_for_task(task, mode=args.spec_mode)
        # 只看歧義 episode（與 neural P3 一致）
        if len(base.candidate_tasks_from_spec(spec)) <= 1:
            continue
        # 凍結盤面
        objs = list(base.objects)
        snap = dict(
            target_task=task,
            hint1_pos=base.hint1_pos, hint1_attr_kind=base.hint1_attr_kind, hint1_attr_value=base.hint1_attr_value,
            hint2_pos=base.hint2_pos, hint2_attr_kind=base.hint2_attr_kind, hint2_attr_value=base.hint2_attr_value,
        )
        for m in modes:
            env = AmbiguousSpecGridWorld(seed=0, n_hints=args.n_hints, horizon=args.horizon,
                                         observe_object_identity=True)
            env.set_episode(objects=objs, **snap)
            succ, ret = run_episode(env, spec, m)
            agg[m]["succ"] += int(succ)
            agg[m]["ret"] += ret
            agg[m]["n"] += 1

    print("=" * 64)
    print(f"E0 / P3-oracle-decision  (spec={args.spec_mode}, n_hints={args.n_hints}, seed={args.seed})")
    print("=" * 64)
    for m in modes:
        n = max(agg[m]["n"], 1)
        print(f"  {m:>9}: success={agg[m]['succ']/n:.3f}  avg_return={agg[m]['ret']/n:+.4f}  (n={agg[m]['n']})")
    nb = max(agg["belief"]["n"], 1)
    print("-" * 64)
    print(f"  P3-oracle return_gap (belief − single)   = {(agg['belief']['ret']-agg['single']['ret'])/nb:+.4f}")
    print(f"  P3-oracle return_gap (belief − single+H) = {(agg['belief']['ret']-agg['single_h']['ret'])/nb:+.4f}")
    print(f"  P3-oracle succ_gap   (belief − single)   = {(agg['belief']['succ']-agg['single']['succ'])/nb:+.4f}")
    print("\n判讀：若 belief 明顯 > single（且 ≥ single+H），環境有 decision value，neural P3 失敗是 learning/representation 問題。")
    print("      若 belief ≈ single，環境本身沒有 decision value，需改環境設計。")


if __name__ == "__main__":
    main()
