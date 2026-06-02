# 過程問題與注意事項

## 1. 不要太早做 zero-shot

`plan.md` 的重點是先理解 few-shot meta-RL 的 adaptation mechanism。K=0 可以記錄，但第一版不應該把 K=0 擴大成 zero-shot 主題。

## 2. RL² 評估時不能每條 trajectory 都 reset hidden state

RL² 的 adaptation 依賴 RNN hidden state。如果每個 episode 都 reset hidden state，就等於把它的核心能力移除。正確方式是在同一個 test task 內連續跑多條 trajectories，只在 task 切換時 reset。

## 3. MAML 的 support/query 必須分清楚

MAML 的 K-shot support trajectories 用來做 inner update，query trajectories 用來評估更新後 policy。若 support 和 query 混在一起，會高估 adaptation 表現。

## 4. PEARL 的 context batch 與 RL batch 概念要分開

PEARL 不是單純把所有 replay buffer 資料餵給 actor-critic。encoder 使用 context 推論 z，actor/critic 使用 RL batch 更新控制能力，兩者抽樣目的不同。

## 5. Dense reward 先於 sparse reward

第一版先用 dense reward，原因是它比較容易確認三種方法是否真的在 K 增加時改善。等 dense reward protocol 跑通後，再切換 sparse reward 觀察探索能力。

## 6. 結果要跨 seeds

Meta-RL variance 通常很大。正式結果至少需要 3 個 seeds，較理想是 5 個 seeds，並回報 mean 與 std。
