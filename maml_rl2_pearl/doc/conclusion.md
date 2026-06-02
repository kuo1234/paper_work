# 第一版結論

## 研究問題

第一版最適合的研究問題是：

> 在簡單 2D Goal Navigation few-shot meta-RL 任務中，比較 MAML、RL²、PEARL 三種 adaptation mechanism：gradient update、recurrent memory、probabilistic context inference，在不同 adaptation trajectories 數量下的 test-time return 與 meta-training sample efficiency。

這個問題比一開始直接討論 zero-shot 更合適，因為三種方法都需要先在 few-shot adaptation protocol 下被理解。等 K=0、K=1、K=2、K=5 的曲線跑通後，才適合把 K=0 獨立拿出來討論 zero-shot。

## 三種方法的核心差異

| 方法 | 學到的東西 | 適應新任務靠什麼 | 測試時是否更新參數 |
| --- | --- | --- | --- |
| MAML | 好微調的 policy initialization | support trajectories 上的 gradient update | 是 |
| RL² | RNN hidden state 內化的學習程序 | 同一任務內連續 trajectories 的 reward feedback | 否 |
| PEARL | 從 context 推論 task latent variable 的 encoder | posterior inference over z | 否 |

## 第一版判讀方式

- 如果 MAML 在 K=1 後明顯提升，代表初始化參數確實容易被少量資料微調。
- 如果 RL² 的第二條 trajectory 比第一條好，代表 RNN hidden state 有利用上一條 trajectory 的回饋。
- 如果 PEARL 在 context 增加後 return 上升，代表 encoder 能把 transitions 壓縮成有用的 task latent variable。
- 如果 PEARL 用更少 meta-training environment steps 達到同等 K=2 return，才算支撐 sample efficiency 的主張。

## 第一版不主張的事

- 不主張 PEARL 一定在所有設定都最好。
- 不把 K=0 當成主要 zero-shot 結論。
- 不用單一 final return 判斷演算法好壞。
- 不用不同 train/test task split 比較三個方法，因為那會讓比較不公平。
