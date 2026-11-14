"""Compare the learning algorithms for the blue robots: MAPPO (ours) vs IPPO, MAA2C and QMIX.

All algorithms use the same hierarchical set-up, features, network size, reward, number of training episodes
(2,400), simulated teammates, validation scenarios and 100 unseen test scenarios x 3 simulated humans, with
3 training seeds each. Only the learning algorithm differs.

    python agents/compare_algorithms.py
expects, for each algorithm, an evaluation folder (eval_episodes.csv from agents/evaluate.py with its 3
checkpoints) and its training runs (log.csv):
    MAPPO  results/eval_episodes.csv                 runs/mappo_v3_seed{0,1,2}/
    IPPO   results/algorithms/ippo/eval_episodes.csv  runs/ippo_seed{0,1,2}/
    MAA2C  results/algorithms/maa2c/...               runs/maa2c_seed{0,1,2}/
    QMIX   results/algorithms/qmix/...                runs/qmix_seed{0,1,2}/
Missing algorithms are skipped. Writes results/algorithm_comparison.md and results/algorithm_comparison.png
"""
import csv
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.training_report import convergence, load, moving   # noqa: E402

ALGOS = [  # name, evaluation folder, run folder prefix, description
    ("MAPPO", "results", "runs/mappo_v3_seed", "centralised critic + PPO clipped update (ours)"),
    ("IPPO", "results/algorithms/ippo", "runs/ippo_seed", "decentralised critic + PPO clipped update"),
    ("MAA2C", "results/algorithms/maa2c", "runs/maa2c_seed", "centralised critic + plain actor-critic update"),
    ("QMIX", "results/algorithms/qmix", "runs/qmix_seed", "value-based: per-robot Q-values + monotonic mixing"),
]
TEAMMATES = ("greedy", "noisy", "idle")
METRICS = [("score", "Team score", "{:.1f}"), ("sev", "Severity saved", "{:.1%}"),
           ("crit_lost", "Critical lost", "{:.2f}"), ("lost", "All lost", "{:.2f}"),
           ("duplicate_rate", "R1-R2 duplicates", "{:.1%}"), ("wasted_trips", "Wasted trips", "{:.2f}")]
COLOURS = {"MAPPO": "#1f5aa6", "IPPO": "#2e8b57", "MAA2C": "#d98c1f", "QMIX": "#8e44ad"}


def episodes_table(folder):
    path = os.path.join(ROOT, folder, "eval_episodes.csv")
    if not os.path.exists(path):
        return None
    return list(csv.DictReader(open(path)))


def per_seed(rows, tm, metric):
    """{run: {scenario seed: value}} for the learned policy."""
    out = {}
    for r in rows:
        if r["policy"] == "mappo" and r["teammate"] == tm:
            out.setdefault(int(r["run"]), {})[int(r["seed"])] = float(r[metric])
    return out


def scenario_means(rows, tm, metric):
    """Value per test scenario, averaged over the training seeds."""
    runs = per_seed(rows, tm, metric)
    seeds = sorted(next(iter(runs.values())))
    return seeds, np.array([np.nanmean([runs[k][s] for k in runs]) for s in seeds])


def paired_p(a, b):
    from math import erf, sqrt
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    se = d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:
        return d.mean(), 1.0
    z = abs(d.mean() / se)
    return d.mean(), 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))


def rule_rows(rows, tm, metric, policy):
    return np.array([float(r[metric]) for r in rows if r["policy"] == policy and r["teammate"] == tm])


def main():
    data = {}
    for name, folder, prefix, desc in ALGOS:
        rows = episodes_table(folder)
        runs = sorted(glob.glob(os.path.join(ROOT, prefix + "*")))
        if rows is None or not any(r["policy"] == "mappo" for r in rows):
            print(f"skipping {name}: no evaluation in {folder}")
            continue
        data[name] = dict(rows=rows, runs=[load(r) for r in runs if os.path.exists(os.path.join(r, "log.csv"))],
                          desc=desc)
    if "MAPPO" not in data:
        sys.exit("MAPPO evaluation (results/eval_episodes.csv) is missing")
    names = list(data)
    ref = data["MAPPO"]["rows"]

    L = ["# Algorithm comparison: which multi-agent RL algorithm should train the blue robots?", "",
         "Same hierarchical set-up for all: the algorithm chooses which victim each blue robot goes for; the same "
         "route planner drives. Same features, network size (deep set, 64 units), reward, 2,400 training episodes, "
         "simulated teammates, validation scenarios, and the same 100 unseen test scenarios per simulated human. "
         "3 training seeds per algorithm; the best checkpoint of each seed is chosen on validation scenarios.", "",
         "| Algorithm | What it is |", "|---|---|"]
    L += [f"| {n} | {data[n]['desc']} |" for n in names]

    # ---------------------------------------------------------------- training
    L += ["", "## Training (mean ± sd over 3 seeds)", "",
          "Validation score = mean of the greedy and idle teammate on the 20 validation scenarios. Convergence = the "
          "moving average of the training return reaches 95% of its own plateau (QMIX's training return includes "
          "epsilon-greedy exploration, so its convergence episode is not directly comparable).", "",
          "| Algorithm | Best validation score | Final validation score | Convergence episode | Training time (min) |",
          "|---|---|---|---|---|"]
    fmt = lambda xs, f="{:.1f}": (f.format(np.nanmean(xs)) + " ± " + f.format(np.nanstd(xs))) if xs else "-"
    for n in names:
        best, final, conv, mins = [], [], [], []
        for d in data[n]["runs"]:
            v = ((d["val_g"] + d["val_i"]) / 2)
            v = v[np.isfinite(v)]
            best.append(v.max())
            final.append(v[-1])
            conv.append(convergence(d)[0])
            mins.append(d["minutes"][-1])
        L.append(f"| {n} | {fmt(best)} | {fmt(final)} | {fmt(conv, '{:.0f}')} | {fmt(mins, '{:.0f}')} |")

    # ---------------------------------------------------------------- test results
    for tm in TEAMMATES:
        L += ["", f"## Test: 100 unseen scenarios, {tm} simulated human (mean ± sd over 3 seeds)", "",
              "| Policy | " + " | ".join(m[1] for m in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        for n in names:
            cells = []
            for k, _, f in METRICS:
                runs = per_seed(data[n]["rows"], tm, k)
                vals = [np.nanmean(list(r.values())) for r in runs.values()]
                cells.append(f.format(np.mean(vals)) + (" ± " + f.format(np.std(vals)) if k == "score" else ""))
            L.append(f"| **{n}** | " + " | ".join(cells) + " |" if n == "MAPPO" else f"| {n} | " + " | ".join(cells) + " |")
        for pol, label in (("greedy", "Greedy rule"), ("feasible", "Deadline-aware rule"), ("random", "Random target")):
            cells = [f.format(np.nanmean(rule_rows(ref, tm, k, pol))) for k, _, f in METRICS]
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L += ["", "Paired comparison with MAPPO over the 100 scenarios (each algorithm averaged over its 3 seeds; "
              "difference = other minus MAPPO; p from a paired t-test):", ""]
        L += ["| Algorithm | Team score difference | p | Critical lost difference | p |", "|---|---|---|---|---|"]
        for n in names:
            if n == "MAPPO":
                continue
            sa, a = scenario_means(data[n]["rows"], tm, "score")
            sb, b = scenario_means(ref, tm, "score")
            assert sa == sb, "different test scenarios"
            d1, p1 = paired_p(a, b)
            _, a2 = scenario_means(data[n]["rows"], tm, "crit_lost")
            _, b2 = scenario_means(ref, tm, "crit_lost")
            d2, p2 = paired_p(a2, b2)
            L.append(f"| {n} | {d1:+.1f} | {p1:.3f} | {d2:+.2f} | {p2:.3f} |")
    open(os.path.join(ROOT, "results", "algorithm_comparison.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    figure(data, names, ref)


def figure(data, names, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    for n in names:                                    # validation learning curves, mean ± sd over seeds
        curves = [(d["episodes"][np.isfinite(d["val_g"])], ((d["val_g"] + d["val_i"]) / 2)[np.isfinite(d["val_g"])])
                  for d in data[n]["runs"]]
        if not curves:
            continue
        m = min(len(c[0]) for c in curves)
        x = curves[0][0][:m]
        Y = np.array([c[1][:m] for c in curves])
        ax.plot(x, Y.mean(0), "o-", ms=3, color=COLOURS[n], label=n)
        ax.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color=COLOURS[n], alpha=0.15)
    ax.set_title("Validation score during training\n(mean of greedy + idle teammate; band = sd over 3 seeds)")
    ax.set_xlabel("training episodes")
    ax.set_ylabel("team score")
    ax.grid(alpha=0.3)
    ax.legend()
    for ax, (metric, title) in zip(axes[1:], (("score", "Test team score"), ("crit_lost", "Critical victims lost"))):
        width = 0.8 / (len(names) + 1)
        for k, n in enumerate(names + ["Greedy rule"]):
            means, sds = [], []
            for tm in TEAMMATES:
                if n == "Greedy rule":
                    means.append(np.nanmean(rule_rows(ref, tm, metric, "greedy")))
                    sds.append(0)
                else:
                    runs = per_seed(data[n]["rows"], tm, metric)
                    vals = [np.nanmean(list(r.values())) for r in runs.values()]
                    means.append(np.mean(vals))
                    sds.append(np.std(vals))
            ax.bar(np.arange(3) + (k - len(names) / 2) * width, means, width, yerr=sds, capsize=3,
                   color=COLOURS.get(n, "#999999"), label=n)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels([f"{tm} human" for tm in TEAMMATES])
        ax.set_title(title + " (100 unseen scenarios;\nerror bar = sd over 3 seeds)")
        ax.grid(alpha=0.3, axis="y")
        if metric == "score":
            lo = min(np.nanmean(rule_rows(ref, tm, "score", "greedy")) for tm in TEAMMATES)
            ax.set_ylim(lo - 40, None)
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "algorithm_comparison.png"), dpi=130)


if __name__ == "__main__":
    main()
