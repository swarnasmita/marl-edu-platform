"""Sanity check for the scene: every robot start and every victim spawn point must be connected to the
triage area on the robots' route map, and a robot must be able to get within pickup range of every spawn point.
Run after changing envs/scene_builder.py:

    python envs/scene_builder.py
    python envs/check_scene.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.heuristic_policy import GridPlanner                     # noqa: E402
from envs.disaster_env import PICKUP_RADIUS, SAFE_CENTER, DisasterResponseEnv   # noqa: E402


def main():
    env = DisasterResponseEnv()
    env.reset(seed=0)
    plan = GridPlanner(env)
    to_triage = plan.field(SAFE_CENTER)
    reachable = np.isfinite(to_triage) & plan.free
    pts = plan.centres[np.argwhere(reachable)]
    problems = []
    for i, p in enumerate(env.robot_positions()):
        if not np.isfinite(to_triage[plan.nearest_free(plan.cell(p))]):
            problems.append(f"robot_{i} start {p.round(1)} cannot reach the triage area")
    for name, sp in zip(env.spawn_names, env.spawn_points):
        gap = np.min(np.linalg.norm(pts - sp[:2], axis=1))
        if gap > PICKUP_RADIUS - 0.05:
            problems.append(f"{name} {sp[:2].round(1)}: closest reachable point is {gap:.2f} m away")
    if problems:
        print("SCENE PROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"scene OK: {len(env.robot_positions())} robot starts and {len(env.spawn_points)} spawn points all reachable")


if __name__ == "__main__":
    main()
