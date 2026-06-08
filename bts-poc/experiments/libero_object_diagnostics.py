from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List


def normalize_obj(s: str) -> str:
    return s.lower().replace(" ", "_").replace("-", "_")


def parse_language(language: str) -> Dict[str, str | None]:
    s = language.lower()
    target = None
    receptacle = None
    relation = None
    m = re.search(r"pick up the (.*?) and place it in the (.*)$", s)
    if m:
        target = normalize_obj(m.group(1))
        receptacle = normalize_obj(m.group(2))
    else:
        # Spatial form: "pick up the black bowl between/next/from/on/in ... and place it on the plate"
        m = re.search(r"pick up the (.*?) (between|next to|from|on|in) (.*?) and place it (?:on|in) the (.*)$", s)
        if m:
            target = normalize_obj(m.group(1))
            relation = f"{m.group(2)} {m.group(3)}"
            receptacle = normalize_obj(m.group(4))
        else:
            m = re.search(r"put the (.*?) (?:in|on) the (.*)$", s)
            if m:
                target = normalize_obj(m.group(1))
                receptacle = normalize_obj(m.group(2))
    return {"target_object": target, "receptacle": receptacle, "relation": relation}


def parse_bddl_goal_and_init(bddl_text: str):
    goal = None
    init_on = []
    goal_match = re.search(r"\(:goal\s*\n\s*\(And\s*\(On\s+([^\s\)]+)\s+([^\s\)]+)\)", bddl_text, re.IGNORECASE)
    if goal_match:
        goal = {"goal_object_instance": goal_match.group(1), "goal_receptacle_instance": goal_match.group(2)}
    for m in re.finditer(r"\(On\s+([^\s\)]+)\s+([^\s\)]+)\)", bddl_text):
        init_on.append({"object": m.group(1), "region_or_receptacle": m.group(2)})
    return goal or {}, init_on


def inspect_suite(suite_name: str, render_one: bool = False):
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark(suite_name)()
    rows = []
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        parsed = parse_language(task.language)
        bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        bddl_text = Path(bddl_path).read_text(errors="replace") if Path(bddl_path).exists() else ""
        goal_info, init_on = parse_bddl_goal_and_init(bddl_text)
        rows.append({
            "task_id": i,
            "name": task.name,
            "language": task.language,
            "problem_folder": task.problem_folder,
            "bddl_file": task.bddl_file,
            "target_object": parsed["target_object"],
            "receptacle": parsed["receptacle"],
            "relation": parsed.get("relation"),
            "goal_object_instance": goal_info.get("goal_object_instance"),
            "goal_receptacle_instance": goal_info.get("goal_receptacle_instance"),
            "init_on": init_on,
            "bddl_path": bddl_path,
            "bddl_contains_target": bool(parsed["target_object"] and parsed["target_object"] in bddl_text.lower()),
            "bddl_contains_receptacle": bool(parsed["receptacle"] and parsed["receptacle"] in bddl_text.lower()),
            "init_states_shape": list(getattr(suite.get_task_init_states(i), "shape", [])),
        })
    obs_keys = []
    if render_one and suite.n_tasks:
        from libero.libero.envs import OffScreenRenderEnv
        task = suite.get_task(0)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        obs = env.reset()
        obs_keys = sorted(obs.keys())
        env.close()
    return {"suite": suite_name, "n_tasks": suite.n_tasks, "tasks": rows, "obs_keys_task0": obs_keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--render-one", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/libero_object_diagnostics.json"))
    args = ap.parse_args()
    payload = inspect_suite(args.suite, args.render_one)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("suite", payload["suite"], "n_tasks", payload["n_tasks"])
    for r in payload["tasks"]:
        print(
            r["task_id"],
            r["target_object"], "->", r["receptacle"],
            "| relation", r.get("relation"),
            "| goal", r.get("goal_object_instance"), "->", r.get("goal_receptacle_instance"),
            "|", r["language"],
            "| bddl", r["bddl_contains_target"], r["bddl_contains_receptacle"],
        )
    if payload["obs_keys_task0"]:
        object_keys = [k for k in payload["obs_keys_task0"] if k.endswith("_pos") or k == "object-state"]
        print("obs_object_keys", object_keys[:30])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
