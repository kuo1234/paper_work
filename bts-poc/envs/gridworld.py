from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple, Optional
import random

# ------------------------------------------------------------
# Ambiguous-Spec GridWorld
# ------------------------------------------------------------
# 7x7 格子，agent 從固定起點出發，目標是走到某個顏色×形狀屬性的物件上。
# 規格可以是精確或歧義。這個檔案同時提供：
# 1) 環境 step / reset
# 2) 任務與規格的對應規則 Z(c)
# 3) 給定規格與當前 state 的 oracle compatible posterior
# 4) BFS 最短路徑專家規劃
# ------------------------------------------------------------

# 動作: up, right, down, left, stay
ACTION_NAMES = ["up", "right", "down", "left", "stay"]
ACTION_TO_DELTA = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
    4: (0, 0),
}

COLORS = ["red", "blue", "green"]
SHAPES = ["square", "circle", "triangle"]

# v4: reward 結構（不對稱懲罰）。penalty 為 eval-only 的環境動態，
# 不進訓練 loss；只有 rollout 時學到的 policy 猜錯走到非 target 物件才觸發。
STEP_REWARD = -0.01
SUCCESS_REWARD = 1.0
WRONG_OBJECT_REWARD = -1.0
ALL_TASKS = [f"{c}_{s}" for c in COLORS for s in SHAPES]
TASK_TO_IDX = {t: i for i, t in enumerate(ALL_TASKS)}
IDX_TO_TASK = {i: t for t, i in TASK_TO_IDX.items()}


@dataclass(frozen=True)
class ObjSpec:
    color: str
    shape: str
    pos: Tuple[int, int]

    @property
    def task_id(self) -> str:
        return f"{self.color}_{self.shape}"


@dataclass(frozen=True)
class Spec:
    color: Optional[str]
    shape: Optional[str]
    text: str

    def key(self) -> Tuple[Optional[str], Optional[str]]:
        return (self.color, self.shape)


class AmbiguousSpecGridWorld:
    def __init__(
        self,
        size: int = 7,
        horizon: int = 15,
        n_objects: int = 4,
        train_excluded_tasks: Optional[List[str]] = None,
        seed: int = 0,
    ):
        self.size = size
        self.horizon = horizon
        self.n_objects = n_objects
        self.rng = random.Random(seed)
        self.train_excluded_tasks = set(train_excluded_tasks or [])

        self.start_pos = (size // 2, size // 2)
        self.agent_pos: Tuple[int, int] = self.start_pos
        self.objects: List[ObjSpec] = []
        self.target_task: Optional[str] = None
        self.hint_pos: Optional[Tuple[int, int]] = None
        self.hint_task: Optional[str] = None
        self.hint_revealed: bool = False
        # v3: hint 改成 partial reveal —— 只揭露 target 的單一屬性（color 或 shape）
        self.hint_attr_kind: Optional[str] = None   # "color" | "shape"
        self.hint_attr_value: Optional[str] = None
        self.t = 0

    # ------------------------------
    # 規格與任務對應
    # ------------------------------
    @staticmethod
    def task_from_attrs(color: str, shape: str) -> str:
        return f"{color}_{shape}"

    @staticmethod
    def parse_task(task: str) -> Tuple[str, str]:
        c, s = task.split("_")
        return c, s

    def candidate_tasks_from_spec(self, spec: Spec) -> List[str]:
        tasks = []
        for task in ALL_TASKS:
            c, s = self.parse_task(task)
            if spec.color is not None and c != spec.color:
                continue
            if spec.shape is not None and s != spec.shape:
                continue
            tasks.append(task)
        return tasks

    def sample_spec_for_task(self, task: str, mode: str = "mixed") -> Spec:
        """
        mode:
          - exact: 精確規格
          - ambiguous: 刻意省略一個屬性（但要保證在當前盤面上真的 >1 相容）
          - mixed: exact / ambiguous 混抽
        """
        color, shape = self.parse_task(task)
        present_tasks = [o.task_id for o in self.objects]

        def support_count(c: Optional[str], s: Optional[str]) -> int:
            cnt = 0
            for t in present_tasks:
                tc, ts = self.parse_task(t)
                if c is not None and tc != c:
                    continue
                if s is not None and ts != s:
                    continue
                cnt += 1
            return cnt

        exact = Spec(color=color, shape=shape, text=f"go to the {color} {shape}")
        color_only = Spec(color=color, shape=None, text=f"go to the {color} object")
        shape_only = Spec(color=None, shape=shape, text=f"go to a {shape}")
        fully_amb = Spec(color=None, shape=None, text="go to an object")

        ambiguous_candidates = []
        if support_count(color_only.color, color_only.shape) > 1:
            ambiguous_candidates.append(color_only)
        if support_count(shape_only.color, shape_only.shape) > 1:
            ambiguous_candidates.append(shape_only)
        if support_count(fully_amb.color, fully_amb.shape) > 1:
            ambiguous_candidates.append(fully_amb)

        if mode == "exact":
            return exact
        if mode == "ambiguous":
            return self.rng.choice(ambiguous_candidates) if ambiguous_candidates else exact
        if ambiguous_candidates and self.rng.random() < 0.65:
            return self.rng.choice(ambiguous_candidates)
        return exact

    # ------------------------------
    # 環境動態
    # ------------------------------
    def _random_empty_position(self, used: set[Tuple[int, int]]) -> Tuple[int, int]:
        while True:
            p = (self.rng.randrange(self.size), self.rng.randrange(self.size))
            if p not in used:
                return p

    def _sample_objects(self) -> List[ObjSpec]:
        # n_objects=4 時，取 2 個顏色 × 2 個形狀的 Cartesian product
        # 保證存在同色不同形、同形不同色，歧義規格才有意義。
        if self.n_objects == 4:
            chosen_colors = self.rng.sample(COLORS, 2)
            chosen_shapes = self.rng.sample(SHAPES, 2)
            chosen = [(c, s) for c in chosen_colors for s in chosen_shapes]
            self.rng.shuffle(chosen)
        else:
            attrs = [(c, s) for c in COLORS for s in SHAPES]
            self.rng.shuffle(attrs)
            chosen = attrs[: self.n_objects]

        used = {self.start_pos}
        objects = []
        for c, s in chosen:
            pos = self._random_empty_position(used)
            used.add(pos)
            objects.append(ObjSpec(c, s, pos))
        return objects

    def _sample_hint_position(self) -> Tuple[int, int]:
        used = {self.start_pos, *[o.pos for o in self.objects]}
        return self._random_empty_position(set(used))

    def _hint_attr_from_task(self, task: str, kind: str) -> str:
        color, shape = self.parse_task(task)
        return color if kind == "color" else shape

    def reset(self, target_task: Optional[str] = None) -> Dict:
        self.objects = self._sample_objects()
        possible_tasks = [o.task_id for o in self.objects if o.task_id not in self.train_excluded_tasks]
        if not possible_tasks:
            possible_tasks = [o.task_id for o in self.objects]
        self.target_task = target_task or self.rng.choice(possible_tasks)
        self.hint_pos = self._sample_hint_position()
        self.hint_task = self.target_task
        # v3: 隨機揭露 color 或 shape 其一，讓 hint 後仍是 2-peak 分布
        self.hint_attr_kind = self.rng.choice(["color", "shape"])
        self.hint_attr_value = self._hint_attr_from_task(self.target_task, self.hint_attr_kind)
        self.hint_revealed = False
        self.agent_pos = self.start_pos
        self.t = 0
        return self.get_obs()

    def set_episode(
        self,
        objects: List[ObjSpec],
        target_task: str,
        agent_pos: Optional[Tuple[int, int]] = None,
        hint_pos: Optional[Tuple[int, int]] = None,
        hint_task: Optional[str] = None,
        hint_revealed: bool = False,
        hint_attr_kind: Optional[str] = None,
        hint_attr_value: Optional[str] = None,
    ) -> Dict:
        self.objects = list(objects)
        self.target_task = target_task
        self.hint_pos = self._sample_hint_position() if hint_pos is None else hint_pos
        self.hint_task = target_task if hint_task is None else hint_task
        # v3: 重建 partial-reveal 屬性。eval rollout 應一律從資料帶入這兩個欄位；
        # 缺值時 deterministic fallback 成 color（不靠 rng，確保可重現）。
        self.hint_attr_kind = hint_attr_kind if hint_attr_kind is not None else "color"
        self.hint_attr_value = (
            hint_attr_value
            if hint_attr_value is not None
            else self._hint_attr_from_task(target_task, self.hint_attr_kind)
        )
        self.hint_revealed = hint_revealed
        self.agent_pos = self.start_pos if agent_pos is None else agent_pos
        self.t = 0
        return self.get_obs()

    def get_obs(self) -> Dict:
        return {
            "agent_pos": self.agent_pos,
            "objects": self.objects,
            "target_task": self.target_task,
            "hint_pos": self.hint_pos,
            "hint_revealed": self.hint_revealed,
            # v3: 揭露後只透露單一屬性，不再給完整 task id
            "hint_attr_kind": self.hint_attr_kind if self.hint_revealed else None,
            "hint_attr_value": self.hint_attr_value if self.hint_revealed else None,
            "t": self.t,
        }

    def _on_target(self, pos: Tuple[int, int], task: str) -> bool:
        for obj in self.objects:
            if obj.task_id == task and obj.pos == pos:
                return True
        return False

    def _on_any_wrong_object(self, pos: Tuple[int, int]) -> bool:
        # v4: 站到某個非 target 物件上（task_id != target_task）
        for obj in self.objects:
            if obj.pos == pos and obj.task_id != self.target_task:
                return True
        return False

    def step(self, action: int):
        dr, dc = ACTION_TO_DELTA[action]
        r, c = self.agent_pos
        nr = min(max(r + dr, 0), self.size - 1)
        nc = min(max(c + dc, 0), self.size - 1)
        self.agent_pos = (nr, nc)
        self.t += 1

        if self.hint_pos is not None and self.agent_pos == self.hint_pos:
            self.hint_revealed = True

        done = False
        reward = STEP_REWARD
        success = False
        wrong_object = False
        if self.target_task is not None and self._on_target(self.agent_pos, self.target_task):
            reward = SUCCESS_REWARD
            done = True
            success = True
        elif self._on_any_wrong_object(self.agent_pos):
            # v4: 不對稱懲罰 —— 猜錯走到錯物件，重罰並結束 episode
            reward = WRONG_OBJECT_REWARD
            done = True
            wrong_object = True
        elif self.t >= self.horizon:
            done = True

        return self.get_obs(), reward, done, {"success": success, "wrong_object": wrong_object}

    # ------------------------------
    # Posterior / compatibility
    # ------------------------------
    def compatible_tasks_given_state(self, spec: Spec, agent_pos: Tuple[int, int]) -> List[str]:
        """
        第三版 oracle compatible set（partial reveal）：
        1. 由 spec 給候選 Z(c)
        2. 若 hint 已揭露，用揭露的單一屬性（color 或 shape）過濾候選
           —— 通常從 ~4 縮到 ~2（仍是 2-peak，不直接收斂成單點）
        3. 若 agent 已站在某個候選 task 物件上，收斂到該 task（最終單點）
        4. 否則維持當前盤面上與 spec 相容的任務集合
        """
        candidates = self.candidate_tasks_from_spec(spec)

        if self.hint_revealed and self.hint_attr_kind is not None and self.hint_attr_value is not None:
            filtered = []
            for t in candidates:
                tc, ts = self.parse_task(t)
                attr = tc if self.hint_attr_kind == "color" else ts
                if attr == self.hint_attr_value:
                    filtered.append(t)
            if filtered:
                candidates = filtered

        matching = [t for t in candidates if self._on_target(agent_pos, t)]
        if matching:
            return matching

        present = {o.task_id for o in self.objects}
        return [t for t in candidates if t in present]

    def oracle_posterior(self, spec: Spec, agent_pos: Tuple[int, int]) -> List[float]:
        comp = self.compatible_tasks_given_state(spec, agent_pos)
        probs = [0.0] * len(ALL_TASKS)
        if len(comp) == 0:
            return probs
        p = 1.0 / len(comp)
        for t in comp:
            probs[TASK_TO_IDX[t]] = p
        return probs

    # ------------------------------
    # Expert planning (BFS)
    # ------------------------------
    def target_positions_for_task(self, task: str) -> List[Tuple[int, int]]:
        return [o.pos for o in self.objects if o.task_id == task]

    def shortest_path_actions(self, start: Tuple[int, int], goals: List[Tuple[int, int]]) -> List[int]:
        goals = set(goals)
        if start in goals:
            return []
        q = deque([start])
        parent = {start: None}
        parent_action = {}
        found_goal = None

        while q:
            cur = q.popleft()
            for a in [0, 1, 2, 3]:
                dr, dc = ACTION_TO_DELTA[a]
                nr = min(max(cur[0] + dr, 0), self.size - 1)
                nc = min(max(cur[1] + dc, 0), self.size - 1)
                nxt = (nr, nc)
                if nxt in parent:
                    continue
                parent[nxt] = cur
                parent_action[nxt] = a
                if nxt in goals:
                    found_goal = nxt
                    q.clear()
                    break
                q.append(nxt)

        if found_goal is None:
            return []

        actions = []
        cur = found_goal
        while parent[cur] is not None:
            actions.append(parent_action[cur])
            cur = parent[cur]
        actions.reverse()
        return actions

    def expert_trajectory(self, task: str, spec: Optional[Spec] = None) -> List[int]:
        # 若規格是歧義的，專家先去 hint tile 再去真正目標；
        # 若規格精確，直接去目標。
        if spec is not None and len(self.candidate_tasks_from_spec(spec)) > 1 and self.hint_pos is not None:
            to_hint = self.shortest_path_actions(self.agent_pos, [self.hint_pos])
            to_goal = self.shortest_path_actions(self.hint_pos, self.target_positions_for_task(task))
            return to_hint + to_goal
        goals = self.target_positions_for_task(task)
        return self.shortest_path_actions(self.agent_pos, goals)


# ------------------------------
# Vectorization utilities
# ------------------------------
def one_hot(items: List[str], value: Optional[str]) -> List[int]:
    return [1 if value == x else 0 for x in items]


def vectorize_spec(spec: Spec) -> List[float]:
    v = []
    v += one_hot(COLORS, spec.color)
    v += [1.0 if spec.color is None else 0.0]
    v += one_hot(SHAPES, spec.shape)
    v += [1.0 if spec.shape is None else 0.0]
    return v


def vectorize_obs(obs: Dict, size: int = 7, max_objects: int = 4) -> List[float]:
    r, c = obs["agent_pos"]
    hint_r, hint_c = obs["hint_pos"] if obs.get("hint_pos") is not None else (-1, -1)
    vec = [
        r / (size - 1),
        c / (size - 1),
        (hint_r / (size - 1)) if hint_r >= 0 else -1.0,
        (hint_c / (size - 1)) if hint_c >= 0 else -1.0,
        1.0 if obs.get("hint_revealed", False) else 0.0,
    ]
    # v3: 不再帶完整 task one-hot；改成編碼揭露的單一屬性
    hint_kind = obs.get("hint_attr_kind", None)
    hint_value = obs.get("hint_attr_value", None)
    revealed_color = hint_value if hint_kind == "color" else None
    revealed_shape = hint_value if hint_kind == "shape" else None
    vec += one_hot(COLORS, revealed_color)  # 3 維
    vec += one_hot(SHAPES, revealed_shape)  # 3 維
    vec += [
        1.0 if hint_kind == "color" else 0.0,
        1.0 if hint_kind == "shape" else 0.0,
    ]  # kind 指示 2 維（未揭露時全 0）
    objs: List[ObjSpec] = list(obs["objects"])[:max_objects]
    for obj in objs:
        vec += [obj.pos[0] / (size - 1), obj.pos[1] / (size - 1)]
        vec += one_hot(COLORS, obj.color)
        vec += one_hot(SHAPES, obj.shape)
    per_obj_dim = 2 + len(COLORS) + len(SHAPES)
    while len(objs) < max_objects:
        vec += [0.0] * per_obj_dim
        objs.append(ObjSpec("red", "square", (0, 0)))
    return vec


def decode_task_idx(idx: int) -> str:
    return IDX_TO_TASK[idx]


def task_idx(task: str) -> int:
    return TASK_TO_IDX[task]


if __name__ == "__main__":
    env = AmbiguousSpecGridWorld(seed=0)
    _ = env.reset()
    task = env.target_task
    assert task is not None
    spec = env.sample_spec_for_task(task, mode="mixed")
    print("Target task:", task)
    print("Objects:", [o.task_id for o in env.objects])
    print("Spec:", spec.text)
    print("Candidate tasks:", env.candidate_tasks_from_spec(spec))
    print("Compatible now:", env.compatible_tasks_given_state(spec, env.agent_pos))
    acts = env.expert_trajectory(task)
    print("Expert actions:", [ACTION_NAMES[a] for a in acts])
