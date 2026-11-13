"""Compare policies for the blue robots on 100 unseen test scenarios.

Policies: random target, greedy rule, deadline-aware rule, and trained MAPPO
(one or more checkpoints, e.g. one per training seed). Every policy uses the
same navigation and is tested with the same scenarios and teammates, so only
the decision-making differs.

    python agents/evaluate.py --checkpoints runs/mappo_seed0/best.pt runs/mappo_seed1/best.pt
Writes results/eval_results.csv, results/eval_table.md and results/eval_episodes.csv.
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from agents.train_mappo import _episode, _init_worker

TEST_SEEDS = list(range(10000, 10100))
TEAMMATES = ("greedy", "noisy", "idle")
METRICS = ("score", "sev", "success", "rescued", "lost", "crit_lost", "crit_survival",
           "wait_critical", "wait_moderate", "wait_minor", "decisions", "duplicate_rate", "wasted_trips",
           "dup_time", "idle_time", "share_learner", "share_blue", "balance")
LABELS = {"random": "Random target", "greedy": "Greedy rule (severity / distance)",
          "feasible": "Deadline-aware rule", "mappo": "MAPPO (hierarchical)"}


def run_policy(pool, mode, ckpt=None):
    a_state = c_state = None
    if ckpt is not None:
        import torch
        state = torch.load(ckpt, map_location="cpu")
        a_state, c_state = state["actor"], state["critic"]
    jobs = [(a_state, c_state, s, tm, mode) for tm in TEAMMATES for s in TEST_SEEDS]
    res = pool.map(_episode, jobs, chunksize=4)
    return [(j[3], j[2], r[0]) for j, r in zip(jobs, res)]


def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))      # normal approximation is fine for n = 100


def paired_t(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() / se if se > 0 else float("inf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="*", default=["runs/mappo_seed0/best.pt"])
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 2)))
    p.add_argument("--out", default="results")
    p.add_argument("--table-only", action="store_true", help="rebuild the tables from eval_episodes.csv")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ckpts = [c for c in args.checkpoints if os.path.exists(c)]

    episodes = []          # (policy, run, teammate, seed, stats)
    if args.table_only:
        for r in csv.DictReader(open(os.path.join(args.out, "eval_episodes.csv"))):
            episodes.append((r["policy"], int(r["run"]), r["teammate"], int(r["seed"]),
                             {m: float(r[m]) for m in METRICS}))
        ckpts = sorted({e[1] for e in episodes if e[0] == "mappo"})
        write_tables(episodes, ckpts, args.out)
        return
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for mode in ("random", "greedy", "feasible"):
            print("evaluating", LABELS[mode], flush=True)
            episodes += [(mode, 0, tm, s, st) for tm, s, st in run_policy(pool, mode)]
        for k, c in enumerate(ckpts):
            print("evaluating MAPPO", c, flush=True)
            episodes += [("mappo", k, tm, s, st) for tm, s, st in run_policy(pool, "eval", c)]

    with open(os.path.join(args.out, "eval_episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "run", "teammate", "seed", *METRICS])
        for pol, run, tm, s, st in episodes:
            w.writerow([pol, run, tm, s, *[st[m] for m in METRICS]])
    write_tables(episodes, ckpts, args.out)


def write_tables(episodes, ckpts, out):
    def values(pol, tm, metric, run=None):
        return [st[metric] for p_, r, t, s, st in episodes if p_ == pol and t == tm and (run is None or r == run)]

    rows = []
    policies = ["random", "greedy", "feasible"] + (["mappo"] if ckpts else [])
    for tm in TEAMMATES:
        for pol in policies:
            row = {"teammate": tm, "policy": LABELS[pol]}
            for m in METRICS:
                if pol == "mappo" and len(ckpts) > 1:   # mean per run, then mean +/- sd across training seeds
                    per_run = [np.nanmean(values(pol, tm, m, run=k)) for k in range(len(ckpts))]
                    row[m] = float(np.mean(per_run))
                    row[m + "_sd"] = float(np.std(per_run))
                else:
                    v = np.asarray(values(pol, tm, m), dtype=float)
                    v = v[np.isfinite(v)]
                    row[m] = float(np.mean(v)) if len(v) else float("nan")
                    row[m + "_sd"] = float(np.std(v) / np.sqrt(len(v))) if len(v) else float("nan")
            rows.append(row)
    with open(os.path.join(out, "eval_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    note = ("± = standard error over scenarios." if len(ckpts) <= 1 else
            f"± = standard error over scenarios for the rules; for MAPPO, sd across {len(ckpts)} training seeds.")
    lines = [f"# Disaster Response: policy comparison ({len(TEST_SEEDS)} unseen scenarios per teammate)", "",
             note, ""]
    lines += ["Success = at least 80% of total severity saved. Wait = seconds from a victim being reported to being "
              "picked up. Duplicate choices = share of blue-robot decisions that picked the victim the other blue "
              "robot was already heading for. Wasted trips = targets taken by someone else or lost before arrival. "
              "Balance = rescues of the less busy blue robot / the busier one (1 = equal).", ""]
    for tm in TEAMMATES:
        sub = [r for r in rows if r["teammate"] == tm]
        lines += [f"## Teammate in the learner's slot: {tm}", "", "**Task performance**", "",
                  "| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | "
                  "Critical lost | Critical survival |", "|---|---|---|---|---|---|---|---|"]
        for r in sub:
            lines.append(f"| {r['policy']} | {r['score']:.1f} ± {r['score_sd']:.1f} | {100 * r['sev']:.1f}% | "
                         f"{100 * r['success']:.0f}% | {r['rescued']:.2f} | {r['lost']:.2f} | {r['crit_lost']:.2f} | "
                         f"{100 * r['crit_survival']:.0f}% |")
        lines += ["", "**Response time and coordination**", "",
                  "| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | "
                  "Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            lines.append(f"| {r['policy']} | {r['wait_critical']:.1f} | {r['wait_moderate']:.1f} | {r['wait_minor']:.1f} | "
                         f"{r['decisions']:.1f} | {100 * r['duplicate_rate']:.1f}% | {r['wasted_trips']:.2f} | "
                         f"{r['idle_time']:.1f} | {100 * r['share_learner']:.0f}% | {r['balance']:.2f} |")
        if ckpts:
            best_rule = max(("greedy", "feasible"), key=lambda pol: np.mean(values(pol, tm, "score")))
            lines.append("")
            for m, name in (("score", "team score"), ("crit_lost", "critical victims lost")):
                mappo = [np.mean([st[m] for p_, r, t, s2, st in episodes
                                  if p_ == "mappo" and t == tm and s2 == s]) for s in TEST_SEEDS]
                rule = [values(best_rule, tm, m)[TEST_SEEDS.index(s)] for s in TEST_SEEDS]
                diff, t = paired_t(mappo, rule)
                pval = 2 * (1 - _norm_cdf(abs(t)))
                lines.append(f"Paired comparison ({name}), MAPPO minus {LABELS[best_rule].lower()}: "
                             f"{diff:+.2f} (t = {t:.2f}, p ≈ {pval:.3f}, n = {len(TEST_SEEDS)}).  ")
        lines.append("")
    open(os.path.join(out, "eval_table.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
