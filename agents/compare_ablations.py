"""Ablation study: does each design choice of the final MAPPO agent matter?

Each variant is the final hierarchical MAPPO (v3) with exactly one thing removed, trained the same way
(2,400 episodes, 3 seeds) and tested on the same 100 unseen scenarios x 3 simulated humans:
    no hierarchy        flat MAPPO: chooses a move every step from raw observations (agents/train_flat.py)
    no deadline mask    victims that can no longer be reached in time are still offered
    no duplicate penalty no -1 for choosing the other blue robot's victim
    no intent input     no "the human seems to be heading here" feature

    python agents/compare_ablations.py
Writes results/ablation_study.md and results/ablation_study.png
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.rl_statistics import B, boot_ci, holm, iqm, matrix, paired   # noqa: E402
from agents.training_report import convergence, load                    # noqa: E402

VARIANTS = [  # name, evaluation folder, run prefix, what is removed
    ("Full MAPPO (final)", "results", "runs/mappo_v3_seed", "-"),
    ("No hierarchy (flat)", "results/ablations/flat", "runs/flat_seed",
     "learns low-level moves from raw observations instead of choosing victims"),
    ("No deadline masking", "results/ablations/no_mask", "runs/mappo_no_mask_seed",
     "victims that can no longer be reached in time are still offered"),
    ("No duplicate penalty", "results/ablations/no_dup_penalty", "runs/mappo_no_dup_penalty_seed",
     "no -1 for choosing the other blue robot's victim"),
    ("No human-intent input", "results/ablations/no_intent", "runs/mappo_no_intent_seed",
     "no 'the human seems to be heading here' feature"),
]
TEAMMATES = ("greedy", "noisy", "idle")
METRICS = [("score", "Team score", "{:.1f}"), ("crit_survival", "Critical rescued", "{:.1%}"),
           ("sev", "Severity saved", "{:.1%}"), ("duplicate_rate", "R1-R2 duplicates", "{:.1%}"),
           ("wasted_trips", "Wasted trips", "{:.2f}")]


def main():
    data = {}
    for name, folder, prefix, what in VARIANTS:
        path = os.path.join(ROOT, folder, "eval_episodes.csv")
        if not os.path.exists(path):
            print("skipping", name, "- no", path)
            continue
        d = pd.read_csv(path)
        runs = [load(os.path.join(ROOT, f"{prefix}{k}")) for k in range(3)
                if os.path.exists(os.path.join(ROOT, f"{prefix}{k}", "log.csv"))]
        data[name] = dict(df=d[d.policy == "mappo"], runs=runs, what=what)
    names = list(data)
    ref_name = "Full MAPPO (final)"
    rule = pd.read_csv(os.path.join(ROOT, "results", "eval_episodes.csv"))
    rule = rule[rule.policy == "greedy"].assign(run=0)

    L = ["# Ablation study: does each design choice matter?", "",
         "Each variant removes exactly one design choice from the final MAPPO agent and is trained the same way "
         "(2,400 episodes, 3 seeds), then tested on the same 100 unseen scenarios per simulated human. "
         "Differences: variant minus full MAPPO, per scenario averaged over seeds, 95% stratified-bootstrap CI, "
         "Holm-corrected paired t-test.", "",
         "| Variant | What is removed |", "|---|---|"]
    L += [f"| {n} | {data[n]['what']} |" for n in names]
    L += ["", "## Training", "", "| Variant | Best validation score | Convergence episode | Training time (min) |",
          "|---|---|---|---|"]
    fmt = lambda xs, f="{:.1f}": (f.format(np.nanmean(xs)) + " ± " + f.format(np.nanstd(xs))) if len(xs) else "-"
    for n in names:
        best, conv, mins = [], [], []
        for d in data[n]["runs"]:
            v = (d["val_g"] + d["val_i"]) / 2
            best.append(np.nanmax(v))
            conv.append(convergence(d)[0])
            mins.append(d["minutes"][-1])
        L.append(f"| {n} | {fmt(best)} | {fmt(conv, '{:.0f}')} | {fmt(mins, '{:.0f}')} |")

    for tm in TEAMMATES:
        L += ["", f"## Test: {tm} simulated human (mean over 3 seeds)", "",
              "| Variant | " + " | ".join(m[1] for m in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        for n in names:
            d = data[n]["df"][data[n]["df"].teammate == tm]
            L.append(f"| {n} | " + " | ".join(f.format(d[k].mean()) for k, _, f in METRICS) + " |")
        d = rule[rule.teammate == tm]
        L.append("| (Greedy rule) | " + " | ".join(f.format(d[k].mean()) for k, _, f in METRICS) + " |")

    L += ["", "## Effect of removing each design choice (variant minus full MAPPO)", "",
          "| Variant | Human | Team score [95% CI] | p (Holm) | Critical rescued [95% CI] | p (Holm) |",
          "|---|---|---|---|---|---|"]
    rows = []
    for n in names:
        if n == ref_name:
            continue
        for tm in TEAMMATES:
            ref = data[ref_name]["df"][data[ref_name]["df"].teammate == tm]
            d = data[n]["df"][data[n]["df"].teammate == tm]
            s = paired(matrix(d, "score"), matrix(ref, "score"))
            c = paired(matrix(d, "crit_survival"), matrix(ref, "crit_survival"))
            rows.append((n, tm, s, c))
    ps = holm([r[2][4] for r in rows])
    pc = holm([r[3][4] for r in rows])
    for (n, tm, s, c), p1, p2 in zip(rows, ps, pc):
        L.append(f"| {n} | {tm} | {s[0]:+.1f} [{s[1]:+.1f}, {s[2]:+.1f}] | {p1:.3g} | "
                 f"{c[0] * 100:+.1f} pts [{c[1] * 100:+.1f}, {c[2] * 100:+.1f}] | {p2:.3g} |")
    open(os.path.join(ROOT, "results", "ablation_study.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    figure(data, names, rule)


def figure(data, names, rule):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    ax = axes[0]
    for n in names:
        curves = [(d["episodes"][np.isfinite(d["val_g"])], ((d["val_g"] + d["val_i"]) / 2)[np.isfinite(d["val_g"])])
                  for d in data[n]["runs"]]
        if not curves:
            continue
        m = min(len(c[0]) for c in curves)
        Y = np.array([c[1][:m] for c in curves])
        ax.plot(curves[0][0][:m], Y.mean(0), "o-", ms=3, label=n)
        ax.fill_between(curves[0][0][:m], Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), alpha=0.15)
    ax.set_title("Validation score during training (band = sd over 3 seeds)")
    ax.set_xlabel("training episodes")
    ax.set_ylabel("team score")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    w = 0.8 / (len(names) + 1)
    for k, n in enumerate(names + ["(Greedy rule)"]):
        if n == "(Greedy rule)":
            vals, err = [rule[rule.teammate == tm].score.mean() for tm in TEAMMATES], [0] * 3
        else:
            d = data[n]["df"]
            per = [[d[(d.teammate == tm) & (d.run == r)].score.mean() for r in sorted(d.run.unique())] for tm in TEAMMATES]
            vals, err = [np.mean(p) for p in per], [np.std(p) for p in per]
        ax.bar(np.arange(3) + (k - len(names) / 2) * w, vals, w, yerr=err, capsize=3, label=n,
               color="#999999" if n == "(Greedy rule)" else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{tm} human" for tm in TEAMMATES])
    ax.set_title("Test team score (100 unseen scenarios; error bar = sd over seeds)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "ablation_study.png"), dpi=130)


if __name__ == "__main__":
    main()
