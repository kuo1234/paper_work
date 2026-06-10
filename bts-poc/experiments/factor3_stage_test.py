#!/usr/bin/env python3
"""因素3 判別測試：multi-stage subgoal switching 是否為 toy policy 的瓶頸。

線索（因素2）：MLP+真實 vectorize_obs+完整專家軌跡 → test acc ~62%（接近真實 55%），
且 train 0.87/test 0.62 過擬合。專家軌跡是多階段 start→hint1→hint2→target，
每階段該往不同 subgoal；切換取決於 hint_revealed 狀態。

本測試把單步樣本依「已揭露 hint 數」拆階段分別算 acc：
  phase 0: 0 個 hint 揭露 → 該往 hint1
  phase 1: 1 個 hint 揭露 → 該往 hint2
  phase 2: 2 個 hint 揭露 → 該往 target
若某階段 acc 特別低、或交界混淆，坐實瓶頸 = 依狀態切換 subgoal（hierarchical 才解）。
對照：把每階段**單獨**訓一個 MLP（無須切換），看 acc 是否大跳——若是，切換確為瓶頸。
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

N_ACTIONS = 5
MAX_OBJ = 4


def sample_with_phase(n_boards, seed, spec_mode="mixed"):
    """逐步記 (obs, spec, action, phase)。phase = 已揭露 hint 數 (0/1/2)。"""
    env = AmbiguousSpecGridWorld(seed=seed, n_hints=2, horizon=30)
    samples = []
    for _ in range(n_boards):
        env.reset()
        task = env.target_task
        spec = env.sample_spec_for_task(task, mode=spec_mode)
        for _ in range(env.horizon):
            a = env.expert_next_action(task, spec)
            obs = env.get_obs()
            phase = int(obs["hint1_revealed"]) + int(obs["hint2_revealed"])
            samples.append((obs, spec, a, phase))
            _, _, done, _ = env.step(a)
            if done:
                break
    return samples, env.size


def encode(samples, size):
    X, Y, P = [], [], []
    for (obs, spec, a, phase) in samples:
        feat = list(vectorize_obs(obs, size=size, max_objects=MAX_OBJ)) + list(vectorize_spec(spec))
        X.append(feat); Y.append(a); P.append(phase)
    return (torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.long),
            torch.tensor(P, dtype=torch.long))


class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, N_ACTIONS))

    def forward(self, x):
        return self.net(x)


def train_model(Xtr, Ytr, epochs=400, seed=0):
    torch.manual_seed(seed)
    m = TinyMLP(Xtr.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train(); opt.zero_grad(); lf(m(Xtr), Ytr).backward(); opt.step()
    m.eval()
    return m


def acc(m, X, Y):
    with torch.no_grad():
        return (m(X).argmax(-1) == Y).float().mean().item()


def main():
    tr, size = sample_with_phase(600, seed=0)
    te, _ = sample_with_phase(200, seed=999)
    Xtr, Ytr, Ptr = encode(tr, size)
    Xte, Yte, Pte = encode(te, size)
    print("=" * 70)
    print("因素3 判別測試：multi-stage subgoal switching 是否為瓶頸")
    print("=" * 70)
    for ph in (0, 1, 2):
        print(f"  phase{ph}: train={int((Ptr==ph).sum())} test={int((Pte==ph).sum())}", end="  ")
    print()
    print("-" * 70)

    # (A) 單一 MLP 學全部階段（須自行依狀態切換）→ 分階段看 acc
    m_all = train_model(Xtr, Ytr)
    print("[A] 單一 MLP（須自切換 subgoal）— 分階段 test acc：")
    overall = acc(m_all, Xte, Yte)
    for ph in (0, 1, 2):
        mask = Pte == ph
        if mask.sum() == 0:
            continue
        print(f"    phase{ph} (揭露{ph}hint→往{'hint1' if ph==0 else 'hint2' if ph==1 else 'target'}): "
              f"acc={acc(m_all, Xte[mask], Yte[mask]):.3f}")
    print(f"    overall={overall:.3f}")

    # (B) 每階段各訓一個專屬 MLP（無須切換）→ 看 acc 是否大跳
    print("\n[B] 每階段各訓專屬 MLP（無須切換，oracle phase）— test acc：")
    per_phase = []
    for ph in (0, 1, 2):
        mtr = Ptr == ph; mte = Pte == ph
        if mtr.sum() < 20 or mte.sum() == 0:
            continue
        m_ph = train_model(Xtr[mtr], Ytr[mtr])
        a = acc(m_ph, Xte[mte], Yte[mte])
        per_phase.append((ph, a, int(mte.sum())))
        print(f"    phase{ph}: acc={a:.3f}")
    # 加權平均（B 的上界 = 若能完美切換）
    if per_phase:
        wsum = sum(a * n for _, a, n in per_phase)
        ntot = sum(n for _, _, n in per_phase)
        b_upper = wsum / ntot
        print(f"    加權平均（完美切換上界）={b_upper:.3f}")

    print("\n[判讀]")
    if per_phase:
        gain = b_upper - overall
        print(f"  分階段專屬 MLP − 單一 MLP = {gain:+.3f}")
        if gain > 0.12:
            print(f"  ✅ 拆掉切換需求後 acc 大跳（+{gain:.2f}）→ multi-stage subgoal switching")
            print("     是瓶頸主因：單一 flat policy 難學『依 hint 狀態切換往哪走』。")
            print("     → 對應 memory 建議：hierarchical policy（subgoal head + go_to controller）。")
        else:
            print(f"  ⚠️ 拆階段幫助有限（+{gain:.2f}）→ 切換非主因；瓶頸更可能在真實模型的")
            print("     history-attention 整合 / belief-KL 競爭（flat MLP 已逼近其單步上界）。")


if __name__ == "__main__":
    main()
