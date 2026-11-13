"""Behaviour tests for the blue robots (MAPPO vs the greedy rule).

Part A - controlled situations. A scenario is set up by hand (which victims are
waiting, how severe, how much time they have left, where the robots are, who is
already heading where) and we check what the deciding robot chooses. Each
situation is repeated on many random placements to give a pass rate.

  1 Victim selection    severity, distance, near-vs-far trade-off, impossible deadline, urgency
  2 R1-R2 coordination  don't duplicate a closer teammate; split two victims between them
  3 Human coordination  leave a victim the human robot is clearly about to reach
  4 Re-planning         switch to a newly reported critical victim; don't drop a critical
                        victim for a newly reported minor one

Part B - full episodes (50 unseen scenarios x 2 simulated humans) with step-by-step
instrumentation: duplicate pursuit, clashes with the human, wasted trips by cause,
reaction time to new reports, survival of late-reported victims, stuck time, robot
jams, unjam manoeuvres, idle time while victims wait, and path efficiency.

    python agents/behaviour_tests.py
Writes results/behaviour_report.md and results/behaviour_tests.png
"""
import os
import sys
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mujoco
import numpy as np

import agents.hrl as H
from envs.disaster_env import (DT, LOST, MAX_STEPS, N_ROBOTS, N_VICTIMS, RESCUED, SPEED, TIME_LIMIT,
                               UNREPORTED, WAITING, DisasterResponseEnv)

CKPT = os.path.join(ROOT, "agents", "checkpoints", "mappo_disaster.pt")
N_CASES = 40                    # random placements per controlled situation
EPISODE_SEEDS = list(range(10000, 10050))
TEAMMATES = ("greedy", "noisy")


# =========================================================================== scene control helpers
class Bench:
    """Puts the environment into hand-made situations."""

    def __init__(self):
        self.env = DisasterResponseEnv(observe=False)
        self.env.reset(seed=0)
        self.ctrl = H.HierarchicalController(self.env)
        self.plan = self.ctrl.planner
        for p in self.env.spawn_points:
            self.plan.field(p[:2])
        m = self.env.model
        self.body0 = [m.body_pos[b][:2].copy() for b in self.env._robot_body]
        free = np.argwhere(self.plan.free)
        pts = np.array([self.plan.point(tuple(c)) for c in free])
        inside = np.all(np.abs(pts) < 14.5, axis=1)
        self.free_pts = pts[inside]
        self.policies = {"MAPPO": H.MappoChooser(self.env, self.ctrl, CKPT),
                         "Greedy rule": lambda i: H.greedy_choice(self.env, self.ctrl, i),
                         "Deadline-aware rule": lambda i: H.greedy_choice(self.env, self.ctrl, i, feasible_only=True)}

    def fresh(self, seed):
        env = self.env
        env.reset(seed=seed)
        self.ctrl.reset()
        env.status[:] = RESCUED            # hide everyone; tests switch on only the victims they use
        env.rescued_count = 0

    def place_robot(self, i, xy):
        env = self.env
        qx, qy = env._robot_qpos[i]
        env.data.qpos[qx], env.data.qpos[qy] = xy[0] - self.body0[i][0], xy[1] - self.body0[i][1]
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)

    def victim(self, j, spot, sev, time_left_s=None):
        env = self.env
        env.status[j] = WAITING
        env.victim_pos[j] = env.spawn_points[spot, :2]
        env.severity[j] = sev
        env.time_left[j] = (time_left_s / DT) if time_left_s is not None else TIME_LIMIT[sev]

    def dist(self, xy, spot):
        return H.path_dist(self.plan, xy, self.env.spawn_points[spot, :2])

    def near_point(self, spot, lo, hi, rng):
        """A free point whose route distance to a spawn spot is between lo and hi metres."""
        f = self.plan.field(self.env.spawn_points[spot, :2])
        cells = np.argwhere((f >= lo) & (f <= hi))
        if not len(cells):
            return None
        return self.plan.point(tuple(cells[rng.integers(len(cells))]))

    def random_free(self, rng):
        return self.free_pts[rng.integers(len(self.free_pts))]


# =========================================================================== part A: controlled situations
def scenario_cases(bench, kind, rng):
    """Yields (setup, expected) pairs. setup() builds the situation; expected(choice) -> pass/fail."""
    spots = np.arange(len(bench.env.spawn_points))
    tries = 0
    made = 0
    while made < N_CASES and tries < 5000:
        tries += 1
        seed = int(rng.integers(1, 10 ** 6))
        me = bench.random_free(rng)
        d = np.array([bench.dist(me, s) for s in spots])
        if not np.all(np.isfinite(d)):
            continue
        a, b = rng.choice(spots, 2, replace=False)
        da, db = d[a], d[b]
        case = None
        if kind == "severity" and abs(da - db) < 3 and min(da, db) > 3:
            case = ([(a, 3, None), (b, 1, None)], 0)                    # critical vs minor, same effort
        elif kind == "distance" and db - da >= 10 and da > 3 and db < 40:
            case = ([(a, 2, None), (b, 2, None)], 0)                    # same severity, near vs far
        elif kind == "trade_off" and db - da >= 10 and da > 3 and db / SPEED + 4 < TIME_LIMIT[3] * DT:
            case = ([(a, 1, None), (b, 3, None)], 1)                    # minor near vs critical far (reachable)
        elif kind == "deadline" and abs(da - db) < 6 and da / SPEED > 6:
            case = ([(a, 3, da / SPEED - 4), (b, 2, None)], 1)          # critical that cannot be reached in time
        elif kind == "urgency" and abs(da - db) < 3 and da > 4 and da / SPEED + 5 < TIME_LIMIT[3] * DT:
            case = ([(a, 3, da / SPEED + 5), (b, 3, None)], 0)          # two critical: one about to run out
        if case is None:
            continue
        made += 1
        victims, want = case

        def setup(seed=seed, me=me, victims=victims):
            bench.fresh(seed)
            bench.place_robot(1, me)
            for j, (spot, sev, tl) in enumerate(victims):
                bench.victim(j, spot, sev, tl)
        yield setup, (lambda choice, want=want: choice == want)


def coordination_cases(bench, kind, rng):
    spots = np.arange(len(bench.env.spawn_points))
    made = tries = 0
    while made < N_CASES and tries < 5000:
        tries += 1
        seed = int(rng.integers(1, 10 ** 6))
        me = bench.random_free(rng)
        a, b = rng.choice(spots, 2, replace=False)
        da, db = bench.dist(me, a), bench.dist(me, b)
        if not (np.isfinite(da) and np.isfinite(db)) or abs(da - db) > 6 or min(da, db) < 6:
            continue
        mate = bench.near_point(a, 2.0, 4.0, rng)          # the other robot is 2-4 m from victim 0
        if mate is None:
            continue
        made += 1
        if kind == "no_duplicate":           # R2 is about to reach victim 0 and is heading there
            def setup(seed=seed, me=me, mate=mate, a=a, b=b):
                bench.fresh(seed)
                bench.place_robot(1, me)
                bench.place_robot(2, mate)
                bench.victim(0, a, 2)
                bench.victim(1, b, 2)
                bench.ctrl.target[2] = 0
            yield setup, (lambda c: c == 1), 1
        elif kind in ("human_near", "human_heading"):
            # human_near: the human robot is 2-4 m from victim 0, standing still (no movement to read intent from)
            # human_heading: same placements, but the human has been moving towards victim 0, so the
            #                inferred-intent signal (version 2) points at it - as it would in a real episode
            def setup(seed=seed, me=me, mate=mate, a=a, b=b, heading=(kind == "human_heading")):
                bench.fresh(seed)
                bench.place_robot(1, me)
                bench.place_robot(0, mate)
                bench.place_robot(2, bench.free_pts[0])
                bench.victim(0, a, 2)
                bench.victim(1, b, 2)
                bench.ctrl.target[0] = 0
                if heading:
                    bench.ctrl.human_intent = 0
            yield setup, (lambda c: c == 1), 1
        elif kind == "split":                # R1 and R2 at the same spot choose one after the other
            def setup(seed=seed, me=me, a=a, b=b):
                bench.fresh(seed)
                bench.place_robot(1, me)
                bench.place_robot(2, me + np.array([0.0, 1.2]) if bench.plan.free[bench.plan.cell(me + [0, 1.2])] else me)
                bench.victim(0, a, 3)
                bench.victim(1, b, 3)
            yield setup, None, "split"


def replan_cases(bench, kind, rng):
    spots = np.arange(len(bench.env.spawn_points))
    made = tries = 0
    while made < N_CASES and tries < 5000:
        tries += 1
        seed = int(rng.integers(1, 10 ** 6))
        me = bench.random_free(rng)
        a, b = rng.choice(spots, 2, replace=False)
        da, db = bench.dist(me, a), bench.dist(me, b)
        if not (np.isfinite(da) and np.isfinite(db)) or da < 6:
            continue
        if kind == "switch_to_critical" and not (db <= da + 2 and db / SPEED + 4 < TIME_LIMIT[3] * DT):
            continue
        if kind == "keep_critical" and not (abs(da - db) < 6 and da / SPEED + 4 < TIME_LIMIT[3] * DT):
            continue
        made += 1
        first_sev, new_sev, want = (1, 3, 1) if kind == "switch_to_critical" else (3, 1, 0)

        def setup(seed=seed, me=me, a=a, b=b, first_sev=first_sev, new_sev=new_sev):
            bench.fresh(seed)
            bench.place_robot(1, me)
            bench.victim(0, a, first_sev)
            bench.ctrl.target[1] = 0                      # already heading for victim 0
            bench.ctrl.replan[1] = False
            bench.victim(1, b, new_sev)                    # ... when victim 1 is reported
            bench.env.status[1] = WAITING
            bench.ctrl.notify([("report", None, 1)])
        yield setup, (lambda c, want=want: c == want), 1


def run_controlled(bench):
    tests = [
        ("1 Victim selection", "Severity: critical vs minor, same effort", "severity", scenario_cases, "pick critical"),
        ("1 Victim selection", "Distance: same severity, one 10+ m closer", "distance", scenario_cases, "pick nearer"),
        ("1 Victim selection", "Trade-off: minor nearby vs critical 10+ m further (reachable)", "trade_off",
         scenario_cases, "pick critical"),
        ("1 Victim selection", "Impossible deadline: critical that can't be reached in time vs moderate", "deadline",
         scenario_cases, "pick moderate"),
        ("1 Victim selection", "Urgency: two critical, one about to run out", "urgency", scenario_cases, "pick urgent one"),
        ("2 R1-R2 coordination", "R2 heading for a victim and 2-4 m from it", "no_duplicate", coordination_cases,
         "pick the other victim"),
        ("2 R1-R2 coordination", "R1 and R2 free together, two critical victims", "split", coordination_cases,
         "different victims"),
        ("3 Human coordination", "Human robot 2-4 m from a victim", "human_near", coordination_cases,
         "pick the other victim"),
        ("3 Human coordination", "Human robot 2-4 m from a victim and moving towards it", "human_heading",
         coordination_cases, "pick the other victim"),
        ("4 Re-planning", "Heading for a minor victim when a critical one is reported (reachable)",
         "switch_to_critical", replan_cases, "switch to critical"),
        ("4 Re-planning", "Heading for a critical victim when a minor one is reported", "keep_critical", replan_cases,
         "keep critical"),
    ]
    results = []
    for group, desc, kind, gen, expect in tests:
        row = dict(group=group, test=desc, expect=expect, kind=kind)
        for pname, policy in bench.policies.items():
            rng = np.random.default_rng(zlib.crc32(kind.replace("human_heading", "human_near").encode()))      # same cases for every policy and every run
            passed = total = 0
            for item in gen(bench, kind, rng):
                setup, check = item[0], item[1]
                setup()
                if check is None:                                  # split test: both robots decide in turn
                    c1 = policy(1)
                    bench.ctrl.set_choice(1, c1)
                    c2 = policy(2)
                    ok = c1 != c2 and c1 != H.WAIT and c2 != H.WAIT
                else:
                    ok = check(policy(1))
                passed += ok
                total += 1
            row[pname] = (passed, total)
        results.append(row)
        print(f"{group:22s} | {desc[:60]:60s} | " +
              " | ".join(f"{p}: {row[p][0]}/{row[p][1]}" for p in bench.policies), flush=True)
    return results


# =========================================================================== part B: instrumented episodes
def run_episode(bench, policy, teammate, seed):
    env, ctrl = bench.env, bench.ctrl
    env.reset(seed=seed)
    ctrl.reset()
    mate = H.Teammate(teammate, np.random.default_rng(seed))
    m = dict(blue_decisions=0, dup_choices=0, human_clash=0, wasted_by_blue=0, wasted_by_human=0, wasted_lost=0,
             stuck_events=0, stuck_time=0.0, jam_steps=0, unjams=0, idle_while_waiting=0.0)
    reaction, late_crit, init_crit, efficiency = [], [], [], []
    reported, first_targeted = {}, {}
    prev_target = [None] * N_ROBOTS
    slow = [0] * N_ROBOTS
    trip = {}                       # robot -> (victim, route length at decision, distance travelled)
    last_pos = env.robot_positions().copy()
    escapes_before = 0

    def choose(i):
        a = policy(i)
        m["blue_decisions"] += 1
        if a != H.WAIT:
            if any(ctrl.target[k] == a for k in (1, 2) if k != i):
                m["dup_choices"] += 1
            if ctrl.target[0] == a:
                m["human_clash"] += 1
        return a

    while env.agents:
        acts = ctrl.actions(choose, mate)
        _, _, _, _, info = env.step(acts)
        ev = info["robot_1"]["events"]
        ctrl.notify(ev)
        pos = env.robot_positions()
        moved = np.linalg.norm(pos - last_pos, axis=1)
        for kind, robot, j in ev:
            if kind == "report":
                reported[j] = env.t
            if kind == "pickup" and robot in (1, 2) and robot in trip and trip[robot][0] == j and trip[robot][1] > 2:
                efficiency.append(trip[robot][2] / trip[robot][1])
        for i in range(N_ROBOTS):
            t = ctrl.target[i]
            # wasted trips by cause (blue robots only)
            if i in (1, 2):
                p = prev_target[i]
                if p is not None and p != t and env.status[p] != WAITING and env.picked_by[p] != i:
                    if env.status[p] == LOST:
                        m["wasted_lost"] += 1
                    elif env.picked_by[p] == 0:
                        m["wasted_by_human"] += 1
                    else:
                        m["wasted_by_blue"] += 1
                if t is not None and t != p:
                    trip[i] = [t, H.path_dist(bench.plan, pos[i], env.victim_pos[t]), 0.0]
                elif i in trip:
                    trip[i][2] += moved[i]
                # stuck: has somewhere to go but barely moving for 2 s or more
                has_goal = env.carrying[i] >= 0 or (t is not None and env.status[t] == WAITING)
                if has_goal and moved[i] < 0.15 * SPEED * DT:
                    slow[i] += 1
                    if slow[i] == 40:
                        m["stuck_events"] += 1
                    if slow[i] >= 40:
                        m["stuck_time"] += DT
                else:
                    slow[i] = 0
                if t is None and env.carrying[i] < 0 and np.any(env.status == WAITING):
                    m["idle_while_waiting"] += DT
            if t is not None:
                first_targeted.setdefault(t, env.t)
            prev_target[i] = t
        for a_ in range(N_ROBOTS):
            for b_ in range(a_ + 1, N_ROBOTS):
                if np.linalg.norm(pos[a_] - pos[b_]) < 1.2:
                    m["jam_steps"] += 1
        last_pos = pos.copy()
    for j in range(N_VICTIMS):
        if env.severity[j] == 3:
            (late_crit if env.report_time[j] > 0 else init_crit).append(env.status[j] == RESCUED)
        if env.report_time[j] > 0 and j in reported:
            t0 = reported[j]
            hit = [t for v, t in first_targeted.items() if v == j and t >= t0]
            pk = env.pickup_time[j] if env.pickup_time[j] >= 0 else None
            cands = [x for x in (hit[0] if hit else None, pk) if x is not None]
            reaction.append((min(cands) - t0) * DT if cands else np.nan)
    m.update(reaction=reaction, late_crit=late_crit, init_crit=init_crit, efficiency=efficiency,
             score=env.score, escapes=ctrl.nav.escape_count)
    return m


def run_episodes(bench):
    agg = {}
    for pname, policy in bench.policies.items():
        for tm in TEAMMATES:
            rows = [run_episode(bench, policy, tm, s) for s in EPISODE_SEEDS]
            n = len(rows)
            dec = sum(r["blue_decisions"] for r in rows)
            agg[(pname, tm)] = dict(
                score=np.mean([r["score"] for r in rows]),
                dup=100 * sum(r["dup_choices"] for r in rows) / dec,
                clash=100 * sum(r["human_clash"] for r in rows) / dec,
                w_blue=sum(r["wasted_by_blue"] for r in rows) / n,
                w_human=sum(r["wasted_by_human"] for r in rows) / n,
                w_lost=sum(r["wasted_lost"] for r in rows) / n,
                reaction=np.nanmean([x for r in rows for x in r["reaction"]]),
                no_reaction=100 * np.mean([np.isnan(x) for r in rows for x in r["reaction"]]),
                late_crit=100 * np.mean([x for r in rows for x in r["late_crit"]]),
                init_crit=100 * np.mean([x for r in rows for x in r["init_crit"]]),
                stuck_events=sum(r["stuck_events"] for r in rows) / n,
                stuck_time=sum(r["stuck_time"] for r in rows) / n,
                jam=sum(r["jam_steps"] for r in rows) * DT / n,
                unjams=sum(r["escapes"] for r in rows) / n,
                idle=sum(r["idle_while_waiting"] for r in rows) / n / 2,
                eff=np.median([x for r in rows for x in r["efficiency"]]),
                eff90=np.percentile([x for r in rows for x in r["efficiency"]], 90),
            )
            a = agg[(pname, tm)]
            print(f"{pname:11s} human={tm:6s} score {a['score']:.1f} dup {a['dup']:.1f}% clash {a['clash']:.1f}% "
                  f"wasted blue/human/lost {a['w_blue']:.2f}/{a['w_human']:.2f}/{a['w_lost']:.2f} "
                  f"reaction {a['reaction']:.1f}s stuck {a['stuck_time']:.1f}s unjams {a['unjams']:.1f} "
                  f"eff {a['eff']:.2f}", flush=True)
    return agg


# =========================================================================== report
def write_report(controlled, agg, out):
    pols = ["MAPPO", "Greedy rule", "Deadline-aware rule"]
    L = ["# Behaviour tests: blue robots (MAPPO vs two hand-written rules)", "",
         "## Part A: controlled situations", "",
         f"Each situation was built by hand on {N_CASES} random placements of the deciding robot and the victims "
         "(same placements for both policies). Pass = the robot made the expected choice.", "",
         "| Area | Situation | Expected | " + " | ".join(pols) + " |", "|---|---|---|" + "---|" * len(pols)]
    for r in controlled:
        cells = [f"{100 * r[p][0] / max(r[p][1], 1):.0f}% ({r[p][0]}/{r[p][1]})" for p in pols]
        L.append(f"| {r['group']} | {r['test']} | {r['expect']} | " + " | ".join(cells) + " |")
    L += ["", f"## Part B: full episodes ({len(EPISODE_SEEDS)} unseen scenarios per simulated human)", "",
          "| Measure | " + " | ".join(f"{p}, {tm} human" for p in pols for tm in TEAMMATES) + " |",
          "|---|" + "---|" * (len(pols) * len(TEAMMATES))]
    specs = [
        ("2 R1-R2", "Duplicate choices (% of blue decisions)", "dup", "{:.1f}%"),
        ("2 R1-R2", "Wasted trips: target taken by the other blue robot (per episode)", "w_blue", "{:.2f}"),
        ("3 Human", "Clashes: blue chose the human's current target (% of decisions)", "clash", "{:.1f}%"),
        ("3 Human", "Wasted trips: target taken by the human (per episode)", "w_human", "{:.2f}"),
        ("4 Re-planning", "Reaction time to a new report: first robot heading there (s)", "reaction", "{:.1f}"),
        ("4 Re-planning", "Late-reported victims nobody ever went for (%)", "no_reaction", "{:.0f}%"),
        ("4 Re-planning", "Survival: critical victims reported late", "late_crit", "{:.0f}%"),
        ("4 Re-planning", "Survival: critical victims known from the start", "init_crit", "{:.0f}%"),
        ("5 Waste", "Wasted trips: target lost before arrival (per episode)", "w_lost", "{:.2f}"),
        ("5 Stuck", "Stuck events: has a goal but no progress for 2+ s (per episode)", "stuck_events", "{:.2f}"),
        ("5 Stuck", "Time stuck (s per episode)", "stuck_time", "{:.1f}"),
        ("5 Stuck", "Time two robots are touching (s per episode)", "jam", "{:.1f}"),
        ("5 Waste", "Idle while victims are waiting (s per blue robot per episode)", "idle", "{:.1f}"),
        ("5 Waste", "Path efficiency: distance driven / planned route (median)", "eff", "{:.2f}"),
        ("5 Waste", "Path efficiency: 90th percentile", "eff90", "{:.2f}"),
        ("-", "Team score", "score", "{:.1f}"),
    ]
    for area, name, key, fmtr in specs:
        L.append(f"| {area}: {name} | " + " | ".join(fmtr.format(agg[(p, tm)][key]) for p in pols for tm in TEAMMATES) + " |")
    open(os.path.join(out, "behaviour_report.md"), "w").write("\n".join(L) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(controlled))
    h = 0.27
    for k, (p, col) in enumerate((("MAPPO", "#1f5aa6"), ("Greedy rule", "#9aa5b1"), ("Deadline-aware rule", "#d4a24c"))):
        vals = [100 * r[p][0] / max(r[p][1], 1) for r in controlled]
        ax.barh(y + (k - 1) * h, vals, h, color=col, label=p)
        for yi, v in zip(y, vals):
            ax.text(v + 1, yi + (k - 1) * h, f"{v:.0f}%", va="center", fontsize=6.5)
    ax.set_yticks(y)
    short = {"severity": "Severity: critical vs minor", "distance": "Distance: near vs far",
             "trade_off": "Trade-off: far critical vs near minor", "deadline": "Impossible deadline",
             "urgency": "Urgency: two critical", "no_duplicate": "R1-R2: don't duplicate",
             "split": "R1-R2: split two victims", "human_near": "Human: leave their victim (still)",
             "human_heading": "Human: leave their victim (moving)",
             "switch_to_critical": "Re-plan: switch to new critical", "keep_critical": "Re-plan: keep critical"}
    ax.set_yticklabels([short.get(r.get("kind"), r["test"][:35]) for r in controlled], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("pass rate on hand-made situations (%)")
    ax.set_title("Behaviour tests: does each robot make the expected choice?")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "behaviour_tests.png"), dpi=130)


def main():
    out = os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    bench = Bench()
    print("Part A: controlled situations", flush=True)
    controlled = run_controlled(bench)
    print(f"Part B: full episodes ({time.time() - t0:.0f}s so far)", flush=True)
    agg = run_episodes(bench)
    write_report(controlled, agg, out)
    print(f"wrote results/behaviour_report.md and behaviour_tests.png ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
