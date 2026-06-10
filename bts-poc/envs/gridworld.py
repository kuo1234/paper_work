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
        n_hints: int = 2,
        observe_object_identity: bool = True,
    ):
        self.size = size
        self.horizon = horizon
        self.n_objects = n_objects
        self.n_hints = n_hints  # v6: 1=single-hint(v3 機制) / 2=dual-hint(v5)
        # v7: partial observability。True=現狀(物件身分 t=0 全可見)；
        # False=鄰格揭露(物件 color/shape 僅在 agent 走到 Manhattan<=1 後才揭露，
        # 位置永遠可見)，逼 belief 跨時間累積、結構上不可繞過。
        self.observe_object_identity = observe_object_identity
        self.rng = random.Random(seed)
        self.train_excluded_tasks = set(train_excluded_tasks or [])

        self.start_pos = (size // 2, size // 2)
        self.agent_pos: Tuple[int, int] = self.start_pos
        self.objects: List[ObjSpec] = []
        # v7: 每物件的「身分是否已揭露」狀態，與 self.objects 同索引對齊。
        # fully-obs 模式下一律 True；partial-obs 由鄰格揭露翻為 True (sticky)。
        self.object_revealed: List[bool] = []
        self.target_task: Optional[str] = None
        # v5: 雙 hint，兩階段揭露互補屬性（一個 color、一個 shape）。
        # 集滿兩個 → 完全確定 target；只集一個 → 2-peak（可再被第二 hint 縮小）。
        self.hint1_pos: Optional[Tuple[int, int]] = None
        self.hint1_attr_kind: Optional[str] = None   # "color" | "shape"
        self.hint1_attr_value: Optional[str] = None
        self.hint1_revealed: bool = False
        self.hint2_pos: Optional[Tuple[int, int]] = None
        self.hint2_attr_kind: Optional[str] = None
        self.hint2_attr_value: Optional[str] = None
        self.hint2_revealed: bool = False
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

    def _sample_hint_position(self, extra_used: Optional[set] = None) -> Tuple[int, int]:
        used = {self.start_pos, *[o.pos for o in self.objects]}
        if extra_used:
            used |= set(extra_used)
        return self._random_empty_position(set(used))

    def _hint_attr_from_task(self, task: str, kind: str) -> str:
        color, shape = self.parse_task(task)
        return color if kind == "color" else shape

    def _init_object_reveals(self) -> None:
        # v7: 依模式初始化 object_revealed。fully-obs 全 True；
        # partial-obs 全 False 後套用起點鄰格揭露(deterministic，可重現 data-gen)。
        if self.observe_object_identity:
            self.object_revealed = [True] * len(self.objects)
        else:
            self.object_revealed = [False] * len(self.objects)
            self._update_object_reveals()

    def _update_object_reveals(self) -> None:
        # v7: 鄰格揭露 (Manhattan <= 1，含同格)，sticky。fully-obs 不作用。
        if self.observe_object_identity:
            return
        ar, ac = self.agent_pos
        for i, o in enumerate(self.objects):
            if not self.object_revealed[i] and abs(o.pos[0] - ar) + abs(o.pos[1] - ac) <= 1:
                self.object_revealed[i] = True

    def reset(self, target_task: Optional[str] = None) -> Dict:
        self.objects = self._sample_objects()
        possible_tasks = [o.task_id for o in self.objects if o.task_id not in self.train_excluded_tasks]
        if not possible_tasks:
            possible_tasks = [o.task_id for o in self.objects]
        self.target_task = target_task or self.rng.choice(possible_tasks)
        # v5/v6: hint1 一定有；hint2 只在 n_hints==2 時存在
        self.hint1_pos = self._sample_hint_position()
        self.hint1_attr_kind = self.rng.choice(["color", "shape"])
        self.hint1_attr_value = self._hint_attr_from_task(self.target_task, self.hint1_attr_kind)
        self.hint1_revealed = False
        if self.n_hints >= 2:
            self.hint2_pos = self._sample_hint_position(extra_used={self.hint1_pos})
            self.hint2_attr_kind = "shape" if self.hint1_attr_kind == "color" else "color"
            self.hint2_attr_value = self._hint_attr_from_task(self.target_task, self.hint2_attr_kind)
        else:
            self.hint2_pos = None
            self.hint2_attr_kind = None
            self.hint2_attr_value = None
        self.hint2_revealed = False
        self.agent_pos = self.start_pos
        self.t = 0
        self._init_object_reveals()
        return self.get_obs()

    def set_episode(
        self,
        objects: List[ObjSpec],
        target_task: str,
        agent_pos: Optional[Tuple[int, int]] = None,
        hint1_pos: Optional[Tuple[int, int]] = None,
        hint1_attr_kind: Optional[str] = None,
        hint1_attr_value: Optional[str] = None,
        hint2_pos: Optional[Tuple[int, int]] = None,
        hint2_attr_kind: Optional[str] = None,
        hint2_attr_value: Optional[str] = None,
        hint1_revealed: bool = False,
        hint2_revealed: bool = False,
        observe_object_identity: Optional[bool] = None,
    ) -> Dict:
        # v7: eval rollout 可強制 partial-obs；None 表示沿用 constructor 設定。
        if observe_object_identity is not None:
            self.observe_object_identity = observe_object_identity
        self.objects = list(objects)
        self.target_task = target_task
        # v5/v6: eval rollout 一律從資料帶入 hint 欄位。
        # hint1 缺值時 deterministic fallback；hint2 只在 n_hints>=2 時建立，
        # single-hint 資料的 hint2_* 為 None，這裡保持 None（不自動補）。
        self.hint1_pos = self._sample_hint_position() if hint1_pos is None else hint1_pos
        self.hint1_attr_kind = hint1_attr_kind if hint1_attr_kind is not None else "color"
        self.hint1_attr_value = (
            hint1_attr_value if hint1_attr_value is not None
            else self._hint_attr_from_task(target_task, self.hint1_attr_kind)
        )
        if self.n_hints >= 2 and hint2_pos is not None:
            self.hint2_pos = hint2_pos
            self.hint2_attr_kind = (
                hint2_attr_kind if hint2_attr_kind is not None
                else ("shape" if self.hint1_attr_kind == "color" else "color")
            )
            self.hint2_attr_value = (
                hint2_attr_value if hint2_attr_value is not None
                else self._hint_attr_from_task(target_task, self.hint2_attr_kind)
            )
        else:
            self.hint2_pos = None
            self.hint2_attr_kind = None
            self.hint2_attr_value = None
        self.hint1_revealed = hint1_revealed
        self.hint2_revealed = hint2_revealed if self.hint2_pos is not None else False
        self.agent_pos = self.start_pos if agent_pos is None else agent_pos
        self.t = 0
        self._init_object_reveals()
        return self.get_obs()

    def get_obs(self) -> Dict:
        return {
            "agent_pos": self.agent_pos,
            "objects": self.objects,
            "target_task": self.target_task,
            # v5: 兩個 hint，各自揭露後只透露單一互補屬性
            "hint1_pos": self.hint1_pos,
            "hint1_revealed": self.hint1_revealed,
            "hint1_attr_kind": self.hint1_attr_kind if self.hint1_revealed else None,
            "hint1_attr_value": self.hint1_attr_value if self.hint1_revealed else None,
            "hint2_pos": self.hint2_pos,
            "hint2_revealed": self.hint2_revealed,
            "hint2_attr_kind": self.hint2_attr_kind if self.hint2_revealed else None,
            "hint2_attr_value": self.hint2_attr_value if self.hint2_revealed else None,
            # v7: partial-obs 揭露狀態，供 vectorize_obs 遮罩物件身分
            "observe_object_identity": self.observe_object_identity,
            "object_revealed": list(self.object_revealed),
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

        if self.hint1_pos is not None and self.agent_pos == self.hint1_pos:
            self.hint1_revealed = True
        if self.hint2_pos is not None and self.agent_pos == self.hint2_pos:
            self.hint2_revealed = True
        # v7: 鄰格揭露物件身分（partial-obs 才作用），mirror hint reveal
        self._update_object_reveals()

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
        第五版 oracle compatible set（雙 hint 兩階段揭露）：
        1. 由 spec 給候選 Z(c)
        2. 對每個「已揭露」的 hint，用其單一屬性過濾候選
           - 揭露 0 個 → ~4；揭露 1 個 → ~2（2-peak）；揭露 2 個 → 1（收斂）
        3. 若 agent 已站在某個候選 task 物件上，收斂到該 task
        4. 否則維持當前盤面上與 spec 相容的任務集合
        """
        candidates = self.candidate_tasks_from_spec(spec)

        revealed = []
        if self.hint1_revealed and self.hint1_attr_kind is not None and self.hint1_attr_value is not None:
            revealed.append((self.hint1_attr_kind, self.hint1_attr_value))
        if self.hint2_revealed and self.hint2_attr_kind is not None and self.hint2_attr_value is not None:
            revealed.append((self.hint2_attr_kind, self.hint2_attr_value))

        for kind, value in revealed:
            filtered = []
            for t in candidates:
                tc, ts = self.parse_task(t)
                attr = tc if kind == "color" else ts
                if attr == value:
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

    def shortest_path_actions(self, start: Tuple[int, int], goals: List[Tuple[int, int]],
                              blocked: Optional[set] = None) -> List[int]:
        # v7 fix: blocked = 不可踏入的格子（非 target 物件），BFS 繞過它們，
        # 避免專家最短路穿過錯物件而觸發 wrong_object 提前失敗（污染 BC 資料）。
        goals = set(goals)
        blocked = set(blocked or set())
        blocked -= goals  # 目標格永遠可入（即使它剛好也在 blocked 集合裡）
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
                if nxt in blocked:
                    continue  # 繞過錯物件
                parent[nxt] = cur
                parent_action[nxt] = a
                if nxt in goals:
                    found_goal = nxt
                    q.clear()
                    break
                q.append(nxt)

        if found_goal is None:
            # 退路：若被完全擋住（理論上罕見），允許穿過 blocked 再找一次
            if blocked:
                return self.shortest_path_actions(start, list(goals), blocked=None)
            return []

        actions = []
        cur = found_goal
        while parent[cur] is not None:
            actions.append(parent_action[cur])
            cur = parent[cur]
        actions.reverse()
        return actions

    def _non_target_object_positions(self, task: str) -> set:
        # v7 fix: 所有「非 target task」物件的格子，專家應繞過。
        return {o.pos for o in self.objects if o.task_id != task}

    def expert_trajectory(self, task: str, spec: Optional[Spec] = None) -> List[int]:
        # v5/v6: 歧義規格 → 依序去所有 hint 收集屬性，再去 target；精確規格 → 直接去目標。
        # v7 fix: 全程繞過非 target 物件（避免最短路穿過錯物件提前失敗）。
        blocked = self._non_target_object_positions(task)
        if spec is not None and len(self.candidate_tasks_from_spec(spec)) > 1 and self.hint1_pos is not None:
            acts = self.shortest_path_actions(self.agent_pos, [self.hint1_pos], blocked=blocked)
            cur = self.hint1_pos
            if self.n_hints >= 2 and self.hint2_pos is not None:
                acts += self.shortest_path_actions(cur, [self.hint2_pos], blocked=blocked)
                cur = self.hint2_pos
            acts += self.shortest_path_actions(cur, self.target_positions_for_task(task), blocked=blocked)
            return acts
        goals = self.target_positions_for_task(task)
        return self.shortest_path_actions(self.agent_pos, goals, blocked=blocked)

    def expert_next_action(self, task: str, spec: Optional[Spec] = None) -> int:
        """v7 DAgger: 給定**當前** agent_pos 與已揭露狀態，回下一步正確動作。
        歧義規格：尚未集滿 hint → 先去未揭露的 hint；hint 集滿 → 去 target。
        精確規格：直接去 target。無路可走時回 stay(4)。"""
        ambiguous = spec is not None and len(self.candidate_tasks_from_spec(spec)) > 1
        blocked = self._non_target_object_positions(task)  # v7 fix: 繞過錯物件
        if ambiguous and self.hint1_pos is not None:
            # 依序補齊未揭露的 hint，再去 target
            if not self.hint1_revealed:
                acts = self.shortest_path_actions(self.agent_pos, [self.hint1_pos], blocked=blocked)
                return acts[0] if acts else 4
            if self.n_hints >= 2 and self.hint2_pos is not None and not self.hint2_revealed:
                acts = self.shortest_path_actions(self.agent_pos, [self.hint2_pos], blocked=blocked)
                return acts[0] if acts else 4
        acts = self.shortest_path_actions(self.agent_pos, self.target_positions_for_task(task), blocked=blocked)
        return acts[0] if acts else 4


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


def _encode_hint(obs: Dict, idx: int, size: int) -> List[float]:
    # v5: 單個 hint 的編碼：pos 2 維 + revealed 1 維 + 揭露屬性(color3/shape3) + kind 2 維
    hp = obs.get(f"hint{idx}_pos")
    hint_r, hint_c = hp if hp is not None else (-1, -1)
    revealed = obs.get(f"hint{idx}_revealed", False)
    kind = obs.get(f"hint{idx}_attr_kind", None)
    value = obs.get(f"hint{idx}_attr_value", None)
    revealed_color = value if kind == "color" else None
    revealed_shape = value if kind == "shape" else None
    out = [
        (hint_r / (size - 1)) if hint_r >= 0 else -1.0,
        (hint_c / (size - 1)) if hint_c >= 0 else -1.0,
        1.0 if revealed else 0.0,
    ]
    out += one_hot(COLORS, revealed_color)  # 3
    out += one_hot(SHAPES, revealed_shape)  # 3
    out += [1.0 if kind == "color" else 0.0, 1.0 if kind == "shape" else 0.0]  # 2
    return out


def vectorize_obs(obs: Dict, size: int = 7, max_objects: int = 4) -> List[float]:
    r, c = obs["agent_pos"]
    vec = [r / (size - 1), c / (size - 1)]
    # v5: 兩個 hint 各自編碼
    vec += _encode_hint(obs, 1, size)
    vec += _encode_hint(obs, 2, size)
    objs: List[ObjSpec] = list(obs["objects"])[:max_objects]
    # v7: partial-obs 遮物件身分。fully-obs(預設) 保持 v6 layout、不加 seen bit
    # → obs_dim byte-identical，舊資料/checkpoint 相容。partial-obs 每物件多 1 維 seen。
    observe = obs.get("observe_object_identity", True)
    revealed = obs.get("object_revealed", None)
    for i, obj in enumerate(objs):
        vec += [obj.pos[0] / (size - 1), obj.pos[1] / (size - 1)]
        if observe:
            # fully-obs：原 layout，身分永遠可見、無 seen bit
            vec += one_hot(COLORS, obj.color)
            vec += one_hot(SHAPES, obj.shape)
        else:
            # partial-obs：身分僅在揭露後給，否則全 0；附 seen bit
            is_revealed = revealed is not None and i < len(revealed) and revealed[i]
            if is_revealed:
                vec += one_hot(COLORS, obj.color)
                vec += one_hot(SHAPES, obj.shape)
            else:
                vec += [0.0] * len(COLORS)
                vec += [0.0] * len(SHAPES)
            vec += [1.0 if is_revealed else 0.0]
    per_obj_dim = 2 + len(COLORS) + len(SHAPES) + (0 if observe else 1)
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
