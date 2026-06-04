from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Tiny in-context Transformer for BTS PoC
# ------------------------------------------------------------
# 輸入:
#   spec token: [B, 1, spec_dim]
#   history tokens: 每步一個 token,由 [obs_vec, prev_action_onehot] 組成
# 輸出:
#   - belief logits over K tasks
#   - action logits over A actions
# ------------------------------------------------------------


def causal_mask(seq_len: int, device=None):
    # PyTorch Transformer 需要 True 表示禁止 attention
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1)]


class TinyBTS(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        spec_dim: int,
        n_tasks: int,
        n_actions: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.spec_dim = spec_dim
        self.n_tasks = n_tasks
        self.n_actions = n_actions
        self.d_model = d_model

        # history token = obs_vec + prev_action_onehot
        hist_in_dim = obs_dim + n_actions
        self.spec_proj = nn.Linear(spec_dim, d_model)
        self.hist_proj = nn.Linear(hist_in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 顯式 belief head：從最後一個 token 的表徵輸出 task posterior logits
        self.belief_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_tasks),
        )

        # policy head 吃 [context_repr, belief_probs]
        self.policy_head = nn.Sequential(
            nn.Linear(d_model + n_tasks, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_actions),
        )

    def forward(
        self,
        spec_vec: torch.Tensor,
        hist_obs: torch.Tensor,
        hist_prev_actions_onehot: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        spec_vec: [B, spec_dim]
        hist_obs: [B, T, obs_dim]
        hist_prev_actions_onehot: [B, T, n_actions]
        attention_mask: optional [B, T] where 1 means valid token, 0 padding
        """
        B, T, _ = hist_obs.shape
        hist_in = torch.cat([hist_obs, hist_prev_actions_onehot], dim=-1)  # [B, T, obs+n_actions]

        spec_tok = self.spec_proj(spec_vec).unsqueeze(1)  # [B,1,D]
        hist_tok = self.hist_proj(hist_in)                # [B,T,D]
        x = torch.cat([spec_tok, hist_tok], dim=1)       # [B,T+1,D]
        x = self.pos_enc(x)

        src_mask = causal_mask(x.size(1), device=x.device)

        src_key_padding_mask = None
        if attention_mask is not None:
            # prepend spec token as always valid
            spec_valid = torch.ones(B, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            valid = torch.cat([spec_valid, attention_mask], dim=1)  # [B,T+1]
            src_key_padding_mask = (valid == 0)  # True means ignore

        h = self.transformer(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)  # [B,T+1,D]

        # 取最後一個有效 history token 的表徵；若無 mask 就用最後一個 token
        if attention_mask is None:
            ctx = h[:, -1]
        else:
            idx = attention_mask.sum(dim=1).long()  # number of valid hist tokens
            # 因為前面有 1 個 spec token，最後有效 token = idx（而非 idx-1）
            ctx = h[torch.arange(B, device=h.device), idx]

        belief_logits = self.belief_head(ctx)
        belief_probs = F.softmax(belief_logits, dim=-1)

        policy_in = torch.cat([ctx, belief_probs], dim=-1)
        action_logits = self.policy_head(policy_in)

        return {
            "belief_logits": belief_logits,
            "belief_probs": belief_probs,
            "action_logits": action_logits,
            "context_repr": ctx,
        }


class SinglePointBaseline(nn.Module):
    """
    模擬「T2DA 式單點」：先從 spec/history argmax 一個 task,再只餵 one-hot task 做 policy。
    同樣用 transformer backbone，差異只在 policy 不吃整個 belief，而是吃 argmax one-hot。
    """
    def __init__(self, base_model: TinyBTS):
        super().__init__()
        self.base = base_model
        d_model = base_model.d_model
        n_tasks = base_model.n_tasks
        n_actions = base_model.n_actions
        self.point_policy = nn.Sequential(
            nn.Linear(d_model + n_tasks, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_actions),
        )

    def forward(self, spec_vec, hist_obs, hist_prev_actions_onehot, attention_mask=None):
        out = self.base(spec_vec, hist_obs, hist_prev_actions_onehot, attention_mask=attention_mask)
        belief_logits = out["belief_logits"]
        belief_probs = out["belief_probs"]
        ctx = out["context_repr"]

        point_idx = torch.argmax(belief_probs, dim=-1)
        point_onehot = F.one_hot(point_idx, num_classes=belief_probs.size(-1)).float()
        action_logits = self.point_policy(torch.cat([ctx, point_onehot], dim=-1))
        out["action_logits"] = action_logits
        out["point_task_idx"] = point_idx
        return out


def make_prev_action_onehot(actions: torch.Tensor, n_actions: int) -> torch.Tensor:
    """
    actions: [B, T] current actions labels.
    return prev_action_onehot: [B, T, n_actions], where t=0 uses zero vector.
    """
    B, T = actions.shape
    prev = torch.zeros(B, T, n_actions, device=actions.device)
    if T > 1:
        prev[:, 1:] = F.one_hot(actions[:, :-1], num_classes=n_actions).float()
    return prev


def sequence_attention_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    lengths: [B] valid sequence lengths
    return mask [B, max_len] with 1 valid / 0 pad
    """
    device = lengths.device
    ar = torch.arange(max_len, device=device).unsqueeze(0)
    return (ar < lengths.unsqueeze(1)).float()
