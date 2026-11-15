"""Full-information planner: an upper reference for the blue robots.

It knows everything the trained agents cannot: every victim that will be reported later (where, when, how
severe), every deadline, and which victim the human is heading for. At each decision of a blue robot it tries
every option (each waiting victim, or waiting 1 s) and simulates the rest of the episode for both blue robots
with travel times from the route planner - including victims reported later - then takes the option whose
simulated plan rescues the most (+10 x severity per rescue, +1 per pickup, -5 x severity per victim lost;
rescues that happen sooner count slightly more, 0.995 per second, so it never waits without reason).
It moves the robots with the same planner and rules as all other policies, so it is a strong reference
("how much better could the robots do with perfect information?"), not a guaranteed optimum.

    python agents/oracle_planner.py --workers 2
Evaluates on the same 100 test scenarios x 3 simulated humans as agents/evaluate.py (same seeds, same human
behaviour); writes results/oracle/eval_episodes.csv and eval_table.md.
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from envs.disaster_env import (CARRY_SPEED, DT, MAX_STEPS, N_VICTIMS, SAFE_CENTER, SPEED, TIME_LIMIT,   # noqa: E402
                               UNREPORTED, WAITING)

TEST_SEEDS = list(range(10000, 10100))
TEAMMATES = ("greedy", "noisy", "idle")
END = MAX_STEPS * DT
SOONER = 0.995       # per second: among plans that rescue the same victims, prefer the one that rescues them sooner


class OraclePlanner:
    def __init__(self, env, ctrl):
        self.env, self.ctrl = env, ctrl
        self._cache = {}

    def d(self, a, b):
        from agents.hrl import path_dist
        key = (round(float(a[0]), 1), round(float(a[1]), 1), round(float(b[0]), 1), round(float(b[1]), 1))
        if key not in self._cache:
            self._cache[key] = path_dist(self.ctrl.planner, np.asarray(a), np.asarray(b))
        return self._cache[key]

    def _victims(self, i):
        """Victims still to be planned for, with (available from, deadline) in seconds."""
        env, ctrl = self.env, self.ctrl
        now = env.t * DT
        out = {}
        taken = {ctrl.target[k] for k in range(3) if k != i and ctrl.target[k] is not None}
        for j in range(N_VICTIMS):
            if j in taken:
                continue
            sev = int(env.severity[j])
            if env.status[j] == WAITING:
                dl = now + env.time_left[j] * DT if sev > 1 else np.inf
                out[j] = (now, dl, sev)
            elif env.status[j] == UNREPORTED:
                rt = env.report_time[j] * DT
                out[j] = (rt, rt + TIME_LIMIT[sev] * DT if sev > 1 else np.inf, sev)
        return out

    def _robots(self, i):
        """(free time, position) of the other blue robot once it has finished its current job, and the value
        of that job."""
        env, ctrl = self.env, self.ctrl
        now = env.t * DT
        pos = env.robot_positions()
        k = 2 if i == 1 else 1
        tgt = ctrl.target[k]
        if env.carrying[k] >= 0:
            return [(now + self.d(pos[k], SAFE_CENTER) / CARRY_SPEED, SAFE_CENTER)], 0.0
        if tgt is not None and env.status[tgt] == WAITING:
            arrive = now + self.d(pos[k], env.victim_pos[tgt]) / SPEED
            return [(arrive + self.d(env.victim_pos[tgt], SAFE_CENTER) / CARRY_SPEED, SAFE_CENTER)], 0.0
        return [(now, pos[k])], 0.0

    def _rollout(self, robots, victims):
        """Greedy continuation in travel-time space; returns the plan's value."""
        env = self.env
        robots = [list(r) for r in robots]
        left = dict(victims)
        value = 0.0
        while left and robots:
            r = min(range(len(robots)), key=lambda q: robots[q][0])
            t0, p = robots[r]
            best, best_s = None, -1.0
            for j, (avail, dl, sev) in left.items():
                arrive = max(t0, avail) + self.d(p, env.victim_pos[j]) / SPEED
                if arrive > dl or arrive + self.d(env.victim_pos[j], SAFE_CENTER) / CARRY_SPEED > END:
                    continue
                s = sev * 0.95 ** (arrive - t0)
                if s > best_s:
                    best, best_s, best_t = j, s, arrive
            if best is None:
                robots.pop(r)
                continue
            sev = left.pop(best)[2]
            done_t = best_t + self.d(env.victim_pos[best], SAFE_CENTER) / CARRY_SPEED
            value += (10 * sev + 1) * SOONER ** (done_t - env.t * DT)
            robots[r] = [done_t, SAFE_CENTER]
        value -= sum(5 * sev for (_, dl, sev) in left.values() if dl < END)
        return value

    def __call__(self, i):
        from agents.hrl import WAIT
        env = self.env
        now = env.t * DT
        pos = env.robot_positions()[i]
        victims = self._victims(i)
        other, _ = self._robots(i)
        best_a, best_v = WAIT, self._rollout([(now + 1.0, pos)] + other, victims)
        for j, (avail, dl, sev) in victims.items():
            if env.status[j] != WAITING:
                continue
            arrive = now + self.d(pos, env.victim_pos[j]) / SPEED
            rest = {q: v for q, v in victims.items() if q != j}
            if arrive <= dl and arrive + self.d(env.victim_pos[j], SAFE_CENTER) / CARRY_SPEED <= END:
                done_t = arrive + self.d(env.victim_pos[j], SAFE_CENTER) / CARRY_SPEED
                gain = (10 * sev + 1) * SOONER ** (done_t - now)
                me = (done_t, SAFE_CENTER)
            else:
                gain, me = -5 * sev if dl < END else 0.0, (arrive, env.victim_pos[j])
                rest[j] = victims[j]          # not actually saved by this trip
            v = gain + self._rollout([me] + other, rest)
            if v > best_v - 1e-9:                # ties: act rather than wait
                best_a, best_v = j, v
        return best_a


_W = {}


def _init():
    import agents.hrl as H
    from envs.disaster_env import DisasterResponseEnv
    env = DisasterResponseEnv(observe=False)
    env.reset(seed=0)
    ctrl = H.HierarchicalController(env)
    _W.update(env=env, ctrl=ctrl, oracle=OraclePlanner(env, ctrl))


def _episode(job):
    seed, teammate = job
    import agents.hrl as H
    from agents.train_mappo import episode_metrics
    env, ctrl, oracle = _W["env"], _W["ctrl"], _W["oracle"]
    rng = np.random.default_rng(seed)            # same as agents/evaluate.py, so the human behaves identically
    env.reset(seed=seed)
    ctrl.reset()
    oracle._cache = {}
    mate = H.Teammate(teammate, rng)
    while env.agents:
        _, _, _, _, info = env.step(ctrl.actions(oracle, mate))
        ctrl.notify(info["robot_1"]["events"])
    m = episode_metrics(env, ctrl)
    m.update(seed=seed, teammate=teammate, policy="oracle")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 2)))
    p.add_argument("--scenarios", type=int, default=len(TEST_SEEDS))
    args = p.parse_args()
    jobs = [(s, tm) for tm in TEAMMATES for s in TEST_SEEDS[:args.scenarios]]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        rows = pool.map(_episode, jobs, chunksize=4)
    out = os.path.join(ROOT, "results", "oracle")
    os.makedirs(out, exist_ok=True)
    keys = list(rows[0].keys())
    with open(os.path.join(out, "eval_episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    L = ["# Full-information planner (upper reference)", "", "| Simulated human | Team score | Critical rescued | "
         "Severity saved | Critical lost |", "|---|---|---|---|---|"]
    for tm in TEAMMATES:
        r = [x for x in rows if x["teammate"] == tm]
        L.append(f"| {tm} | {np.mean([x['score'] for x in r]):.1f} | {np.mean([x['crit_survival'] for x in r]):.1%} | "
                 f"{np.mean([x['sev'] for x in r]):.1%} | {np.mean([x['crit_lost'] for x in r]):.2f} |")
    open(os.path.join(out, "eval_table.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
