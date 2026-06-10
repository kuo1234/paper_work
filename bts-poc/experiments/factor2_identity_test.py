#!/usr/bin/env python3
"""因素2 判別測試：task-identity inference → action 傳遞，是否為 toy policy 的瓶頸。

背景：
- 測試 D：真實 transformer 連 SEEN task + teacher-forcing，單步 action acc 僅 55%。
- 因素1 測試：同容量 MLP + 絕對座標 + **給 target one-hot**，單步 acc ~87%
  → 座標表徵不是瓶頸。但那測試把最難的「哪個物件是 target」當輸入送掉了。

本測試隔離 identity inference：MLP 拿到與真實模型單步**相同**的資訊
（真實 vectorize_obs + vectorize_spec，含 spec、hint 揭露狀態、物件身分），
唯一變因 = 給不給 target one-hot：
  ORACLE_ID : obs_vec + spec_vec + target one-hot   ← identity 免費（上界）
  INFER     : obs_vec + spec_vec                     ← 須自推 target（真實設定）

沿專家**完整軌跡**（含去 hint 階段）採樣，涵蓋「hint 揭露前往 hint、揭露後往 target」
的真實決策序列。memoryless MLP → 隔離掉 history/transformer/belief-KL，只測單步可學性。

判讀：
  ORACLE_ID ≫ INFER → identity inference（spec+hint→哪個物件）是瓶頸主因。
  ORACLE_ID ≈ INFER（且都高）→ 連 inference 都學得起來，真實 transformer 的 55%
     來自他處：ctx 單點瓶頸 / belief-KL 競爭 / history attention，非單步資訊不足。
  兩者都低 → 任務單步本質難（不太可能，因素1 已見 ~87%）。
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

import torch
import torch.nn as nn

from envs.gridworld import AmbiguousSpecGridWorld, vectorize_obs, vectorize_spec

N_ACTIONS = 5  # up/right/down/left/stay（expert 收集 hint 階段可能含 stay）
MAX_OBJ = 4


def sample_expert_steps(n_boards, seed, spec_mode="mixed"):
    """沿專家完整軌跡（去 hint→target）逐步記 (obs_dict, spec, target_idx, action)。"""
    env = AmbiguousSpecGridWorld(seed=seed, n_hints=2, horizon=30)
    samples = []
    for _ in range(n_boards):
        env.reset()
        task = env.target_task
        spec = env.sample_spec_for_task(task, mode=spec_mode)
        tgt_idx = next(i for i, o in enumerate(env.objects) if o.task_id == task)
        for _ in range(env.horizon):
            a = env.expert_next_action(task, spec)
            obs = env.get_obs()
            samples.append((obs, spec, tgt_idx, a))
            _, _, done, _ = env.step(a)
            if done:
                break
    return samples, env.size


def encode(samples, size, give_target_id):
    X, Y = [], []
    for (obs, spec, tgt_idx, a) in samples:
        feat = list(vectorize_obs(obs, size=size, max_objects=MAX_OBJ))
        feat += list(vectorize_spec(spec))
        if give_target_id:
            oh = [0.0] * MAX_OBJ
            oh[tgt_idx] = 1.0
            feat += oh
        X.append(feat)
        Y.append(a)
    return torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.long)


class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


def train_eval(Xtr, Ytr, Xte, Yte, epochs=400, seed=0):
    torch.manual_seed(seed)
    model = TinyMLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        lossf(model(Xtr), Ytr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        tr = (model(Xtr).argmax(-1) == Ytr).float().mean().item()
        te = (model(Xte).argmax(-1) == Yte).float().mean().item()
    return tr, te


def main():
    samples_tr, size = sample_expert_steps(n_boards=600, seed=0)
    samples_te, _ = sample_expert_steps(n_boards=200, seed=999)
    print("=" * 68)
    print("因素2 判別測試：identity inference → action 傳遞是否為瓶頸")
    print("=" * 68)
    print(f"grid size={size}, 真實 vectorize_obs+vectorize_spec 編碼")
    print(f"訓練樣本={len(samples_tr)}, 測試樣本={len(samples_te)}")
    dist = " ".join(f"{a}:{sum(1 for s in samples_tr if s[3]==a)}" for a in range(N_ACTIONS))
    print(f"動作分布(train, 0u/1r/2d/3l/4stay)：{dist}")
    print("-" * 68)
    rows = []
    for name, give in (("ORACLE_ID", True), ("INFER", False)):
        Xtr, Ytr = encode(samples_tr, size, give)
        Xte, Yte = encode(samples_te, size, give)
        trs, tes = [], []
        for sd in range(3):
            tr, te = train_eval(Xtr, Ytr, Xte, Yte, seed=sd)
            trs.append(tr); tes.append(te)
        tr_m, te_m = sum(trs) / 3, sum(tes) / 3
        rows.append((name, te_m))
        print(f"  {name:>9} (in_dim={Xtr.shape[1]:>2}): train={tr_m:.3f}  test={te_m:.3f}  (3 seeds)")
    print("-" * 68)
    oid = next(r[1] for r in rows if r[0] == "ORACLE_ID")
    inf = next(r[1] for r in rows if r[0] == "INFER")
    print(f"  ORACLE_ID − INFER test_acc gap = {oid - inf:+.3f}")
    print("\n[判讀]")
    if oid - inf > 0.15:
        print(f"  ✅ 給 target id 大幅勝自推（+{oid-inf:.2f}）→ identity inference 是瓶頸主因：")
        print("     toy 卡在『spec+hint → 哪個物件是 target』這步推論，非座標、非 rollout。")
    elif inf > 0.75:
        print(f"  ⚠️ 自推也學得起來（INFER={inf:.2f}）→ 單步資訊充足，inference 非瓶頸。")
        print("     真實 transformer 的 55% 來自他處：ctx 單點瓶頸 / belief-KL 競爭 /")
        print("     history attention 整合失敗。瓶頸在『架構如何匯總資訊』，非單步可學性。")
    else:
        print(f"  ⚠️ 兩者皆中低（ORACLE_ID={oid:.2f}, INFER={inf:.2f}）→ 需再細分。")


if __name__ == "__main__":
    main()
