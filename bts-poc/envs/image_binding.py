from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import random
import math

import numpy as np

COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
    "green": (40, 170, 70),
    "yellow": (230, 200, 40),
}
SHAPES = ["square", "circle", "triangle"]
ALL_COMBOS = [(c, s) for c in COLORS for s in SHAPES]

# Hold out compositional pairs for OOD binding. Train still sees every color and every shape,
# but not these exact color-shape bindings as targets.
HELDOUT_COMBOS = {("red", "triangle"), ("blue", "square"), ("green", "circle")}
TRAIN_COMBOS = [cs for cs in ALL_COMBOS if cs not in HELDOUT_COMBOS]
OOD_COMBOS = list(HELDOUT_COMBOS)

# Deliberate train-time shortcut: target color tends to appear in a preferred quadrant.
# OOD breaks this correlation, so a location-prior policy fails.
COLOR_PRIOR_QUADRANT = {
    "red": "tl",
    "blue": "tr",
    "green": "bl",
    "yellow": "br",
}


@dataclass(frozen=True)
class ImageObj:
    color: str
    shape: str
    xy: Tuple[int, int]
    radius: int

    @property
    def combo(self) -> Tuple[str, str]:
        return (self.color, self.shape)

    @property
    def text(self) -> str:
        return f"{self.color} {self.shape}"


@dataclass(frozen=True)
class ImageBindingExample:
    image: np.ndarray  # [H,W,3], uint8
    instruction: str
    objects: List[ImageObj]
    target_index: int
    split: str
    target_combo: Tuple[str, str]


def _quadrant_bounds(q: str, size: int, margin: int) -> Tuple[int, int, int, int]:
    mid = size // 2
    if q == "tl":
        return margin, mid - margin, margin, mid - margin
    if q == "tr":
        return mid + margin, size - margin, margin, mid - margin
    if q == "bl":
        return margin, mid - margin, mid + margin, size - margin
    if q == "br":
        return mid + margin, size - margin, mid + margin, size - margin
    raise ValueError(q)


def _sample_xy(rng: random.Random, size: int, margin: int, quadrant: str | None = None) -> Tuple[int, int]:
    if quadrant:
        x0, x1, y0, y1 = _quadrant_bounds(quadrant, size, margin)
    else:
        x0, x1, y0, y1 = margin, size - margin, margin, size - margin
    return rng.randint(x0, x1), rng.randint(y0, y1)


def _far_enough(xy: Tuple[int, int], others: Sequence[ImageObj], min_dist: int) -> bool:
    x, y = xy
    for o in others:
        if math.hypot(x - o.xy[0], y - o.xy[1]) < min_dist:
            return False
    return True


def _draw_square(img: np.ndarray, x: int, y: int, r: int, rgb: Tuple[int, int, int]) -> None:
    img[max(0, y - r): y + r + 1, max(0, x - r): x + r + 1] = rgb


def _draw_circle(img: np.ndarray, x: int, y: int, r: int, rgb: Tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
    img[mask] = rgb


def _draw_triangle(img: np.ndarray, x: int, y: int, r: int, rgb: Tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    y0, y1 = max(0, y - r), min(h - 1, y + r)
    for yy in range(y0, y1 + 1):
        # Upright triangle: width grows toward bottom.
        frac = (yy - (y - r)) / max(1, 2 * r)
        half = int(frac * r)
        x0, x1 = max(0, x - half), min(w - 1, x + half)
        img[yy, x0:x1 + 1] = rgb


def render_scene(objects: Sequence[ImageObj], size: int = 96) -> np.ndarray:
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    for o in objects:
        rgb = COLORS[o.color]
        x, y = o.xy
        if o.shape == "square":
            _draw_square(img, x, y, o.radius, rgb)
        elif o.shape == "circle":
            _draw_circle(img, x, y, o.radius, rgb)
        elif o.shape == "triangle":
            _draw_triangle(img, x, y, o.radius, rgb)
        else:
            raise ValueError(o.shape)
    return img


class ControlledImageBindingBenchmark:
    """Synthetic image-language object binding benchmark.

    The action is a discrete object selection. This isolates binding before adding robot
    dynamics. The train split contains a color-location shortcut and excludes selected
    color-shape target pairs. The OOD split reintroduces held-out pairs and breaks the
    location shortcut.
    """

    def __init__(self, size: int = 96, n_objects: int = 4, radius: int = 8, seed: int = 0):
        self.size = size
        self.n_objects = n_objects
        self.radius = radius
        self.rng = random.Random(seed)

    def sample(self, split: str = "train") -> ImageBindingExample:
        if split not in {"train", "id", "ood"}:
            raise ValueError(f"unknown split: {split}")
        target_combo = self.rng.choice(TRAIN_COMBOS if split in {"train", "id"} else OOD_COMBOS)
        objects: List[ImageObj] = []

        # Target placement: train/id preserve shortcut, ood breaks it.
        q = COLOR_PRIOR_QUADRANT[target_combo[0]] if split in {"train", "id"} else None
        xy = self._sample_nonoverlap(objects, q)
        objects.append(ImageObj(target_combo[0], target_combo[1], xy, self.radius))

        # Distractors include same-color/different-shape and different-color/same-shape when possible.
        distractor_combos = []
        same_color = [(target_combo[0], s) for s in SHAPES if s != target_combo[1]]
        same_shape = [(c, target_combo[1]) for c in COLORS if c != target_combo[0]]
        self.rng.shuffle(same_color)
        self.rng.shuffle(same_shape)
        distractor_combos.extend(same_color[:1])
        distractor_combos.extend(same_shape[:1])
        remaining = [cs for cs in ALL_COMBOS if cs != target_combo and cs not in distractor_combos]
        self.rng.shuffle(remaining)
        distractor_combos.extend(remaining[: max(0, self.n_objects - 1 - len(distractor_combos))])

        for combo in distractor_combos[: self.n_objects - 1]:
            xy = self._sample_nonoverlap(objects, None)
            objects.append(ImageObj(combo[0], combo[1], xy, self.radius))

        self.rng.shuffle(objects)
        target_index = next(i for i, o in enumerate(objects) if o.combo == target_combo)
        img = render_scene(objects, self.size)
        return ImageBindingExample(
            image=img,
            instruction=f"pick the {target_combo[0]} {target_combo[1]}",
            objects=objects,
            target_index=target_index,
            split=split,
            target_combo=target_combo,
        )

    def _sample_nonoverlap(self, objects: Sequence[ImageObj], quadrant: str | None) -> Tuple[int, int]:
        margin = self.radius + 4
        for _ in range(200):
            xy = _sample_xy(self.rng, self.size, margin, quadrant)
            if _far_enough(xy, objects, min_dist=self.radius * 3):
                return xy
        return _sample_xy(self.rng, self.size, margin, quadrant)


def diagnose_prediction(example: ImageBindingExample, pred_index: int) -> Dict[str, object]:
    target = example.objects[example.target_index]
    pred = example.objects[pred_index]
    success = pred_index == example.target_index
    return {
        "success": success,
        "wrong_object": not success,
        "wrong_color": pred.color != target.color,
        "wrong_shape": pred.shape != target.shape,
        "target": target.text,
        "pred": pred.text,
        "instruction": example.instruction,
        "split": example.split,
    }


def location_prior_policy(example: ImageBindingExample) -> int:
    """Shortcut baseline: choose object nearest the train-time color quadrant center."""
    color = example.target_combo[0]
    q = COLOR_PRIOR_QUADRANT[color]
    bounds = _quadrant_bounds(q, example.image.shape[0], 0)
    x0, x1, y0, y1 = bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dists = [math.hypot(o.xy[0] - cx, o.xy[1] - cy) for o in example.objects]
    return int(np.argmin(dists))


def oracle_language_policy(example: ImageBindingExample) -> int:
    """Upper-bound parser policy: choose object matching the instruction attributes."""
    return example.target_index
