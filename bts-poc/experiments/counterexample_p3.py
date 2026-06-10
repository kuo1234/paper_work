#!/usr/bin/env python3
"""反例測試 v3（不訓練、不改 env、本地秒跑）：horizon-壓力模型 + robustness sweep。

== 兩輪負結果的教訓 ==
v1（賭單物件）：最優動作≈argmax → 0 碰撞。
v2（hedge 點、中央起點、寬鬆 horizon）：belief 最優 hedge 全是 START（待原地等揭曉成本最低）→ 0 碰撞。
→ 真相：toy 自然設定下完整分布無用武之地（這正是 E0 追平的結構根因）。

== v3 核心：揭曉前須 pre-commit 移動 ==
真實「完整分布 > entropy 純量」機制 = **時限壓力下，揭曉前就得朝機率質量集中處預移**。
模型：
  - 揭曉在第 T_reveal 步發生（之前 agent 只知 belief，不知真 target）。
  - agent 揭曉前移動到 pre-commit 點 p（須 reachable：dist(start,p) <= T_reveal）。
  - 揭曉後真 target = 候選 i（機率 b_i），從 p 走向 target_i。
  - 成功 iff 剩餘步數夠：dist(p, target_i) <= BUDGET（= horizon − T_reveal）。
  EV(p) = Σ_i b_i · [ reachable_i ? (SUCCESS − STEP·總步數) : FAIL ]
最優 p 取決於 mass 落在哪些候選（加權覆蓋/facility-location）→ 完整分布的函數。
entropy 相同但 mass 幾何不同 → 最優 p 不同 → single+H（只有 argmax,H）分不出 → 必輸其一。

== robustness sweep（防呆）==
掃一系列 BUDGET，報告每個預算下的碰撞數與 belief−single+H gap。
  - gap>0 只在刀刃般窄帶出現 → artifact，別改 env。
  - gap>0 在合理區間穩定 → 真結構，才動 env。

== agent（最嚴格公平版）==
  belief   : 看完整 mass → 全盤搜每盤面最優 pre-commit 點。
  single   : 只看 argmax → pre-commit = 朝 argmax 物件走。
  single+H : 只能吃 (argmax_pos, H_bucket)。同桶盤面必用同一 pre-commit 點；
             給它後見之明最大優勢：每桶選「全盤任一格」中桶內平均 EV 最高者
             （比給選單更強：它能挑最佳單格，但無法區分同桶的不同盤面）。
"""
from __future__ import annotations

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
import itertools
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from envs.gridworld import STEP_REWARD, SUCCESS_REWARD

Pos = Tuple[int, int]

STEP_COST = abs(STEP_REWARD)   # 沿用 env 0.01；成功與否由 reachability 主導，step 只當 tie-break
FAIL_REWARD = 0.0              # 揭曉後來不及走到 = 失敗，得 0（不賭錯，是 timeout）

SIZE = 7
START: Pos = (3, 3)
# 4 候選分兩群：LEFT 群 2 個、RIGHT 群 2 個。群內聚攏、兩群分得開（pre-commit 方向才有意義）。
POSITIONS: List[Pos] = [(0, 0), (2, 0), (4, 6), (6, 6)]
GROUP = ["LEFT", "LEFT", "RIGHT", "RIGHT"]
N = len(POSITIONS)
ALL_CELLS: List[Pos] = [(r, c) for r in range(SIZE) for c in range(SIZE)]


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def entropy(masses: List[float]) -> float:
    s = sum(masses)
    if s <= 0:
        return 0.0
    return -sum((p / s) * math.log(p / s + 1e-12) for p in masses if p > 0)


def argmax_idx(masses: List[float]) -> int:
    return max(range(len(masses)), key=lambda i: masses[i])


def ev_precommit(p: Pos, masses: List[float], t_reveal: int, budget: int) -> float:
    """pre-commit 到 p（須 reachable），揭曉後走向真 target。
    EV = Σ_i b_i·[ dist(p,target_i)<=budget ? SUCCESS−STEP·(dist(start,p)+dist(p,target_i)) : FAIL ]。"""
    d_start = manhattan(START, p)
    if d_start > t_reveal:
        return -1e18  # 揭曉前到不了 p，非法
    ev = 0.0
    for i, b in enumerate(masses):
        d_after = manhattan(p, POSITIONS[i])
        if d_after <= budget:
            ev += b * (SUCCESS_REWARD - STEP_COST * (d_start + d_after))
        else:
            ev += b * FAIL_REWARD
    return ev


def best_precommit(masses: List[float], t_reveal: int, budget: int,
                   cells: Optional[List[Pos]] = None) -> Tuple[Pos, float]:
    cells = cells if cells is not None else ALL_CELLS
    best_p, best_ev = None, -1e18
    for p in cells:
        ev = ev_precommit(p, masses, t_reveal, budget)
        if ev > best_ev:
            best_p, best_ev = p, ev
    return best_p, best_ev


# ----------------------------------------------------------------------------
# agents
# ----------------------------------------------------------------------------

def act_belief(masses, t_reveal, budget) -> Pos:
    return best_precommit(masses, t_reveal, budget)[0]


def act_single(masses, t_reveal, budget) -> Pos:
    """mass 全壓 argmax → 朝 argmax 物件走，且只能走 t_reveal 步（揭曉前）。
    取『往 argmax 物件方向、距離<=t_reveal 的最遠可達點』。"""
    ai = argmax_idx(masses)
    target = POSITIONS[ai]
    # 全盤找：reachable 且離 argmax 物件最近的點（mass 全壓單點時這就是最優）
    best_p, best_d = START, manhattan(START, target)
    for p in ALL_CELLS:
        if manhattan(START, p) <= t_reveal and manhattan(p, target) < best_d:
            best_p, best_d = p, manhattan(p, target)
    return best_p


def build_single_h_policy(boards, t_reveal, budget):
    """公平版 single+H：決策只能吃 (argmax_pos, H_bucket)。
    後見之明最大優勢：每桶在『全盤任一格』中選桶內平均 EV 最高的單格。
    （它能挑最佳單格，但同桶不同盤面必用同一格 → 純量資訊瓶頸的代價。）"""
    H_DEC = 6

    def feat(m):
        return (POSITIONS[argmax_idx(m)], round(entropy(m), H_DEC))

    bucket: Dict[Tuple[Pos, float], List[List[float]]] = defaultdict(list)
    for m in boards:
        bucket[feat(m)].append(m)

    chosen: Dict[Tuple[Pos, float], Pos] = {}
    for key, ms in bucket.items():
        best_pt, best_avg = None, -1e18
        for pt in ALL_CELLS:
            if manhattan(START, pt) > t_reveal:
                continue
            avg = sum(ev_precommit(pt, m, t_reveal, budget) for m in ms) / len(ms)
            if avg > best_avg:
                best_pt, best_avg = pt, avg
        chosen[key] = best_pt

    return lambda m: chosen[feat(m)]


# ----------------------------------------------------------------------------
# mass 枚舉 + 碰撞
# ----------------------------------------------------------------------------

def enumerate_mass_configs(grid: int = 12) -> List[List[float]]:
    configs = []
    for a in range(1, grid):
        for b in range(1, grid):
            for c in range(1, grid):
                d = grid - a - b - c
                if d <= 0:
                    continue
                configs.append([a / grid, b / grid, c / grid, d / grid])
    return configs


def precommit_side(p: Pos) -> str:
    return "LEFT" if p[1] < START[1] else ("RIGHT" if p[1] > START[1] else "MID")


def count_collisions(configs, t_reveal, budget) -> int:
    """(argmax_pos,H) 相同但 belief 最優 pre-commit 偏向不同側的盤面對數。"""
    H_DEC = 6
    buckets = defaultdict(list)
    for m in configs:
        buckets[(POSITIONS[argmax_idx(m)], round(entropy(m), H_DEC))].append(m)
    n = 0
    for key, ms in buckets.items():
        if len(ms) < 2:
            continue
        sides = [precommit_side(best_precommit(m, t_reveal, budget)[0]) for m in ms]
        for i, j in itertools.combinations(range(len(ms)), 2):
            if sides[i] in ("LEFT", "RIGHT") and sides[j] in ("LEFT", "RIGHT") and sides[i] != sides[j]:
                n += 1
    return n


def gaps_on_all(configs, t_reveal, budget):
    """全盤面上三 agent 平均 EV（single+H 後見之明最佳）。"""
    sh = build_single_h_policy(configs, t_reveal, budget)
    eb = sum(ev_precommit(act_belief(m, t_reveal, budget), m, t_reveal, budget) for m in configs) / len(configs)
    es = sum(ev_precommit(act_single(m, t_reveal, budget), m, t_reveal, budget) for m in configs) / len(configs)
    eh = sum(ev_precommit(sh(m), m, t_reveal, budget) for m in configs) / len(configs)
    return eb, es, eh


def fmt(m):
    return " | ".join(f"{GROUP[i][0]}{i}@{POSITIONS[i]} {m[i]:.3f}" for i in range(N))


def main():
    grid = 12
    configs = enumerate_mass_configs(grid=grid)
    t_reveal = 3  # 揭曉在第 3 步（agent 有 3 步可預移）
    print("=" * 80)
    print("反例測試 v3：完整 belief 分布 vs (argmax+entropy 純量) — horizon 壓力 + pre-commit")
    print("=" * 80)
    print(f"幾何 size={SIZE} start={START}  候選：" +
          ", ".join(f"{GROUP[i]}{POSITIONS[i]}" for i in range(N)))
    print(f"t_reveal={t_reveal}（揭曉前可預移步數）, reward SUCCESS={SUCCESS_REWARD} FAIL={FAIL_REWARD}")
    print(f"mass 配置數={len(configs)} (grid={grid})")

    # ---- robustness sweep over BUDGET（揭曉後剩餘步數）----
    print("\n[Robustness sweep] BUDGET = 揭曉後剩餘步數預算：")
    print(f"  {'budget':>6} | {'collisions':>10} | {'belief':>8} {'single':>8} {'single+H':>9} | {'b−s+H':>8}")
    print("  " + "-" * 70)
    sweep_rows = []
    for budget in range(2, 13):
        ncol = count_collisions(configs, t_reveal, budget)
        eb, es, eh = gaps_on_all(configs, t_reveal, budget)
        sweep_rows.append((budget, ncol, eb, es, eh, eb - eh))
        print(f"  {budget:>6} | {ncol:>10} | {eb:>+8.4f} {es:>+8.4f} {eh:>+9.4f} | {eb-eh:>+8.4f}")

    # 找 gap>0 的穩定區間
    pos_budgets = [r[0] for r in sweep_rows if r[5] > 1e-6]
    print("  " + "-" * 70)
    print(f"  belief−single+H > 0 的 budget：{pos_budgets}")

    # ---- 展示一個代表碰撞對（用 gap 最大的 budget）----
    if sweep_rows:
        best_row = max(sweep_rows, key=lambda r: r[5])
        budget = best_row[0]
        H_DEC = 6
        buckets = defaultdict(list)
        for m in configs:
            buckets[(POSITIONS[argmax_idx(m)], round(entropy(m), H_DEC))].append(m)
        shown = False
        for key, ms in buckets.items():
            if shown or len(ms) < 2:
                continue
            for m1, m2 in itertools.combinations(ms, 2):
                p1 = best_precommit(m1, t_reveal, budget)[0]
                p2 = best_precommit(m2, t_reveal, budget)[0]
                s1, s2 = precommit_side(p1), precommit_side(p2)
                if s1 in ("LEFT", "RIGHT") and s2 in ("LEFT", "RIGHT") and s1 != s2:
                    print(f"\n[代表碰撞對 @ budget={budget}] argmax 與 entropy 完全相同：")
                    print(f"  共同特徵 argmax_pos={key[0]} H={key[1]:.6f}")
                    print(f"  Board A: {fmt(m1)}")
                    print(f"           belief pre-commit={p1}（偏 {s1}）")
                    print(f"  Board B: {fmt(m2)}")
                    print(f"           belief pre-commit={p2}（偏 {s2}）")
                    print("  → 同 (argmax,H)，最優預移方向相反：純量無法區分。")
                    shown = True
                    break

    print("\n[結論]")
    stable = len(pos_budgets) >= 3 and max(pos_budgets) - min(pos_budgets) >= 2
    knife = 0 < len(pos_budgets) < 3
    if stable:
        print(f"  ✅ belief 在 budget∈{pos_budgets} 區間穩定優於 single+H（非刀刃窄帶）。")
        print("     → 真結構：『時限壓力 + pre-commit + 多峰質量』下完整分布有不可替代價值。")
        print("     → 照此改 env（加揭曉延遲 + 收緊 horizon + 兩群多峰）有依據，可進 E0。")
        print("     ⚠ 誠實命題收窄：novelty = belief 在『須預先 commit 的時限決策』下勝，非普遍勝。")
        sys.exit(0)
    elif knife:
        print(f"  ⚠ gap>0 只在窄帶 budget={pos_budgets} → 疑似 artifact，先別改 env，需再驗。")
        sys.exit(3)
    else:
        print("  ❌ 沒有 budget 讓 belief 優於 single+H → 此幾何救不了 novelty，別改 env。")
        sys.exit(2)


if __name__ == "__main__":
    main()
