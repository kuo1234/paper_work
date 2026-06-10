#!/usr/bin/env python3
"""因素1 判別測試：絕對座標 vs 相對 Δ 向量，對「單步方向推理」可學性的影響。

背景：路徑2 測試 D 發現連 SEEN task + teacher-forcing，下一步 action acc 只 55%。
假設（因素1）：obs 用 normalized **絕對座標** (r/(s-1), c/(s-1))，模型須自學「target−agent
取號→映射動作」，對小容量模型出奇地難。若改 **相對 Δ 向量** (Δr, Δc) 直接給，方向資訊
唾手可得，acc 應大跳。

設計（乾淨隔離，CPU 秒級，不碰主 pipeline / GPU）：
- 用真實 AmbiguousSpecGridWorld 幾何採樣盤面 + BFS 專家動作當 label（與真實任務同分布）。
- 同一批 (state, action) 樣本，編兩種特徵：
    ABS : agent(2) + 每物件 pos(2) + target one-hot           ← 模擬現狀
    REL : 每物件 Δ=(obj−agent) 正規化(2) + target one-hot      ← 直接給相對向量
- 同容量小 MLP、同超參、同訓練步數，只差輸入編碼。比 held-out 單步 action acc。
- **不需歷史/transformer**：因素1 是單步方向推理問題，memoryless MLP 已足以隔離。
  同容量 apples-to-apples → 差異純來自編碼。

判讀：REL acc ≫ ABS acc → 因素1（絕對座標表徵）是 policy 學不起來的主因之一。
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

import random
import torch
import torch.nn as nn

from envs.gridworld import AmbiguousSpecGridWorld, COLORS, SHAPES

# 動作: 0 up,1 right,2 down,3 left（不含 stay；最短路不需 stay）
N_ACTIONS = 4
MAX_OBJ = 4


def sample_states(n_boards, seed):
    """採樣盤面，沿 BFS 專家最短路逐步記 (agent_pos, objects, target_pos, action)。
    回傳 list of (agent_pos, [obj_pos...], target_obj_index)。"""
    env = AmbiguousSpecGridWorld(seed=seed, n_hints=2, horizon=30)
    samples = []
    for _ in range(n_boards):
        env.reset()
        objs = list(env.objects)
        target = env.target_task
        tgt_idx = next(i for i, o in enumerate(objs) if o.task_id == target)
        tgt_pos = objs[tgt_idx].pos
        # 沿最短路走，每步記當前 state + 該步動作（繞過非 target 物件，與真實一致）
        blocked = {o.pos for o in objs if o.task_id != target}
        pos = env.start_pos
        guard = 0
        while pos != tgt_pos and guard < 40:
            acts = env.shortest_path_actions(pos, [tgt_pos], blocked=blocked)
            if not acts:
                break
            a = acts[0]
            samples.append((pos, [o.pos for o in objs], tgt_idx, a))
            dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][a]
            pos = (pos[0] + dr, pos[1] + dc)
            guard += 1
    return samples, env.size


def encode(samples, size, mode):
    X, Y = [], []
    for (apos, opos_list, tgt_idx, a) in samples:
        ar, ac = apos
        feat = []
        if mode == "abs":
            feat += [ar / (size - 1), ac / (size - 1)]
            for i in range(MAX_OBJ):
                if i < len(opos_list):
                    r, c = opos_list[i]
                    feat += [r / (size - 1), c / (size - 1)]
                else:
                    feat += [0.0, 0.0]
        elif mode == "rel":
            # 不給 agent 絕對位置；只給每物件相對 Δ（方向資訊直接可讀）
            for i in range(MAX_OBJ):
                if i < len(opos_list):
                    r, c = opos_list[i]
                    feat += [(r - ar) / (size - 1), (c - ac) / (size - 1)]
                else:
                    feat += [0.0, 0.0]
        # target one-hot（兩種編碼都給，公平）
        oh = [0.0] * MAX_OBJ
        oh[tgt_idx] = 1.0
        feat += oh
        X.append(feat)
        Y.append(a)
    return torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.long)


class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


def train_eval(Xtr, Ytr, Xte, Yte, epochs=300, seed=0):
    torch.manual_seed(seed)
    model = TinyMLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xtr), Ytr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        tr_acc = (model(Xtr).argmax(-1) == Ytr).float().mean().item()
        te_acc = (model(Xte).argmax(-1) == Yte).float().mean().item()
    return tr_acc, te_acc


def main():
    random.seed(0)
    samples_tr, size = sample_states(n_boards=600, seed=0)
    samples_te, _ = sample_states(n_boards=200, seed=999)
    print("=" * 64)
    print("因素1 判別測試：絕對座標 vs 相對 Δ 向量（單步 action 可學性）")
    print("=" * 64)
    print(f"grid size={size}, N_ACTIONS={N_ACTIONS}（up/right/down/left）")
    print(f"訓練樣本={len(samples_tr)}, 測試樣本={len(samples_te)}")
    print(f"動作分布(train)：" +
          " ".join(f"{a}:{sum(1 for s in samples_tr if s[3]==a)}" for a in range(N_ACTIONS)))
    print("-" * 64)
    rows = []
    for mode in ("abs", "rel"):
        Xtr, Ytr = encode(samples_tr, size, mode)
        Xte, Yte = encode(samples_te, size, mode)
        # 多 seed 平均，避免單 seed 假象
        trs, tes = [], []
        for sd in range(3):
            tr, te = train_eval(Xtr, Ytr, Xte, Yte, seed=sd)
            trs.append(tr); tes.append(te)
        tr_m = sum(trs) / len(trs); te_m = sum(tes) / len(tes)
        rows.append((mode, Xtr.shape[1], tr_m, te_m))
        print(f"  {mode.upper():>4} (in_dim={Xtr.shape[1]:>2}): "
              f"train_acc={tr_m:.3f}  test_acc={te_m:.3f}  (3 seeds)")
    print("-" * 64)
    abs_te = next(r[3] for r in rows if r[0] == "abs")
    rel_te = next(r[3] for r in rows if r[0] == "rel")
    print(f"  REL − ABS test_acc gap = {rel_te - abs_te:+.3f}")
    print("\n[判讀]")
    if rel_te - abs_te > 0.15:
        print(f"  ✅ 相對 Δ 向量大幅勝絕對座標（+{rel_te-abs_te:.2f}）→ 因素1 坐實：")
        print("     現狀『normalized 絕對座標』表徵是 policy 學不起來的主因之一。")
        print("     救 toy 的最便宜改法 = obs 改 entity-centric relative vector。")
    else:
        print(f"  ⚠️ 兩編碼差距不大（{rel_te-abs_te:+.2f}）→ 因素1 非主因，瓶頸在他處")
        print("     （belief KL 競爭 / ctx 單點瓶頸 / BC 軌跡覆蓋）。")


if __name__ == "__main__":
    main()
