"""Robust statistics for the RL results, following Agarwal et al. (2021), "Deep Reinforcement Learning at the
Edge of the Statistical Precipice" (NeurIPS): interval estimates instead of point estimates.

For every algorithm and simulated human (standard test: 100 unseen scenarios x 3 training seeds; unseen-learner
test: 4 learner types x 50 scenarios x 3 seeds):
  - mean and interquartile mean (IQM: mean of the middle 50% of episodes, robust to outliers)
  - 95% confidence intervals from a stratified bootstrap (resampling test scenarios and training seeds)
  - paired difference from MAPPO with a bootstrap 95% CI, effect size (Cohen's d_z) and a paired t-test p-value,
    Holm-Bonferroni corrected over all comparisons in the same table
  - probability of improvement P(MAPPO > other): chance that MAPPO does better on a random test scenario
  - if results/oracle/eval_episodes.csv exists (agents/oracle_planner.py), the normalised score
    (score - random) / (full-information planner - random)

    python agents/rl_statistics.py
Writes results/rl_statistics.md and results/rl_statistics.png
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
B = 2000
RNG = np.random.default_rng(0)
ALGOS = {"MAPPO": "results/eval_episodes.csv", "IPPO": "results/algorithms/ippo/eval_episodes.csv",
         "MAA2C": "results/algorithms/maa2c/eval_episodes.csv", "QMIX": "results/algorithms/qmix/eval_episodes.csv"}
RULES = {"greedy": "Greedy rule", "feasible": "Deadline-aware rule", "random": "Random target"}
TEAMMATES = ("greedy", "noisy", "idle")
METRICS = {"score": ("Team score", "{:.1f}"), "crit_survival": ("Critical victims rescued", "{:.1%}"),
           "sev": ("Severity saved", "{:.1%}")}


def iqm(x):
    x = np.sort(np.asarray(x, float).ravel())
    n = len(x)
    return x[int(np.floor(0.25 * n)):int(np.ceil(0.75 * n))].mean()


def matrix(df, metric, scen_col="seed"):
    """runs x scenarios array (one row for a rule)."""
    piv = df.pivot_table(index="run" if "run" in df and df["run"].notna().any() else None,
                         columns=scen_col, values=metric, aggfunc="mean") if "run" in df else None
    if piv is None or isinstance(piv, pd.Series):
        piv = df.groupby(scen_col)[metric].mean().to_frame().T
    return piv.sort_index(axis=1).values


def boot_ci(M, stat):
    """Stratified bootstrap over training seeds (rows) and test scenarios (columns)."""
    R, S = M.shape
    vals = []
    for _ in range(B):
        r = RNG.integers(R, size=R)
        s = RNG.integers(S, size=S)
        vals.append(stat(M[np.ix_(r, s)]))
    return np.percentile(vals, [2.5, 97.5])


def paired(Ma, Mb):
    """a minus b, per scenario averaged over seeds; bootstrap CI over scenarios and seeds; d_z; t-test p."""
    from scipy import stats
    a, b = np.nanmean(Ma, 0), np.nanmean(Mb, 0)
    d = a - b
    S = len(d)
    vals = []
    for _ in range(B):
        s = RNG.integers(S, size=S)
        ra = RNG.integers(Ma.shape[0], size=Ma.shape[0])
        rb = RNG.integers(Mb.shape[0], size=Mb.shape[0])
        vals.append(np.nanmean(Ma[np.ix_(ra, s)], 0).mean() - np.nanmean(Mb[np.ix_(rb, s)], 0).mean())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    p = stats.ttest_rel(a, b).pvalue if d.std(ddof=1) > 0 else 1.0
    p_imp = float(np.mean(a > b) + 0.5 * np.mean(a == b))
    return d.mean(), lo, hi, dz, p, p_imp


def holm(ps):
    ps = np.asarray(ps)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    m, running = len(ps), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * ps[i]))
        adj[i] = running
    return adj


def load():
    data = {}
    for name, path in ALGOS.items():
        p = os.path.join(ROOT, path)
        if os.path.exists(p):
            d = pd.read_csv(p)
            data[name] = d[d.policy == "mappo"].copy()
            if name == "MAPPO":
                for key, label in RULES.items():
                    data[label] = d[d.policy == key].copy().assign(run=0)
    oracle = os.path.join(ROOT, "results", "oracle", "eval_episodes.csv")
    if os.path.exists(oracle):
        data["Full-information planner"] = pd.read_csv(oracle).assign(run=0)
    return data


def main():
    data = load()
    learned = [n for n in ALGOS if n in data]
    names = list(data)
    L = ["# RL results with robust statistics", "",
         "Following Agarwal et al. (2021): interval estimates instead of single numbers. 95% CIs from a stratified "
         f"bootstrap ({B} resamples of test scenarios and training seeds). IQM = interquartile mean (mean of the "
         "middle 50% of episodes). Paired comparisons: per test scenario, averaged over each algorithm's 3 "
         "seeds; d_z = mean difference / sd of differences (0.2 small, 0.5 medium, 0.8 large); p-values "
         "Holm-Bonferroni corrected within each table; P(MAPPO better) = probability of improvement on a random "
         "test scenario.", ""]
    fig_rows = []
    for metric, (mname, fmt) in METRICS.items():
        L += [f"## {mname} - standard test (100 unseen scenarios per simulated human)", "",
              "| Policy | Human | Mean [95% CI] | IQM [95% CI] |", "|---|---|---|---|"]
        for tm in TEAMMATES:
            for n in names:
                d = data[n][data[n].teammate == tm]
                if d.empty:
                    continue
                M = matrix(d, metric)
                lo, hi = boot_ci(M, np.nanmean)
                ilo, ihi = boot_ci(M, iqm)
                f = lambda v: fmt.format(v)
                L.append(f"| {'**' + n + '**' if n == 'MAPPO' else n} | {tm} | {f(np.nanmean(M))} [{f(lo)}, {f(hi)}] | "
                         f"{f(iqm(M))} [{f(ilo)}, {f(ihi)}] |")
                if metric == "score":
                    fig_rows.append((n, tm, iqm(M), ilo, ihi))
        L += ["", f"Paired comparison with MAPPO ({mname}; difference = other minus MAPPO):", "",
              "| Other | Human | Difference [95% CI] | d_z | p (Holm) | P(MAPPO better) |", "|---|---|---|---|---|---|"]
        rows = []
        for tm in TEAMMATES:
            ref = matrix(data["MAPPO"][data["MAPPO"].teammate == tm], metric)
            for n in names:
                if n == "MAPPO":
                    continue
                d = data[n][data[n].teammate == tm]
                if d.empty:
                    continue
                diff, lo, hi, dz, p, p_imp = paired(matrix(d, metric), ref)
                rows.append([n, tm, diff, lo, hi, dz, p, 1 - p_imp])
        adj = holm([r[6] for r in rows])
        for r, pa in zip(rows, adj):
            f = (lambda v: f"{v:+.1f}") if metric == "score" else (lambda v: f"{v * 100:+.1f} pts")
            L.append(f"| {r[0]} | {r[1]} | {f(r[2])} [{f(r[3])}, {f(r[4])}] | {r[5]:+.2f} | {pa:.3g} | {r[7]:.0%} |")
        L.append("")

    # unseen learners
    path = os.path.join(ROOT, "results", "robustness_unseen_learners.csv")
    if os.path.exists(path):
        u = pd.read_csv(path)
        L += ["## Team score with learners never seen in training (4 types x 50 scenarios, pooled)", "",
              "| Policy | Mean [95% CI] | IQM [95% CI] | Difference from MAPPO [95% CI] | d_z | p (Holm) | "
              "P(MAPPO better) |", "|---|---|---|---|---|---|---|"]
        u["scen"] = u.learner + "_" + u.seed.astype(str)
        mats = {n: u[u.policy == n].pivot_table(index="run", columns="scen", values="score").sort_index(axis=1).values
                for n in dict.fromkeys(u.policy)}
        rows = []
        for n, M in mats.items():
            lo, hi = boot_ci(M, np.nanmean)
            ilo, ihi = boot_ci(M, iqm)
            if n == "MAPPO":
                rows.append([n, M, lo, hi, ilo, ihi, None])
            else:
                rows.append([n, M, lo, hi, ilo, ihi, paired(M, mats["MAPPO"])])
        adj = iter(holm([r[6][4] for r in rows if r[6] is not None]))
        for n, M, lo, hi, ilo, ihi, pr in rows:
            base = f"| {'**' + n + '**' if n == 'MAPPO' else n} | {np.nanmean(M):.1f} [{lo:.1f}, {hi:.1f}] | {iqm(M):.1f} [{ilo:.1f}, {ihi:.1f}] |"
            if pr is None:
                L.append(base + " - | - | - | - |")
            else:
                L.append(base + f" {pr[0]:+.1f} [{pr[1]:+.1f}, {pr[2]:+.1f}] | {pr[3]:+.2f} | {next(adj):.3g} | {1 - pr[5]:.0%} |")
            fig_rows.append((n, "unseen", iqm(M), ilo, ihi))
        L.append("")

    # normalised score against the full-information planner
    if "Full-information planner" in data:
        L += ["## How close to the full-information planner? (standard test)", "",
              "Normalised score = (score - random target) / (full-information planner - random target): 0 = random, "
              "100% = the planner that knows every victim in advance. Mean over the 3 simulated humans. The planner "
              "is a strong reference, not the optimum: it knows the future but does not model how the human will "
              "act, so learned policies can exceed 100% with an active human; with the idle human (the robots do "
              "everything) it is ahead of every learned policy.", "",
              "| Policy | Normalised score [95% CI] |", "|---|---|"]
        for n in names:
            if n in ("Random target",):
                continue
            vals = []
            for _ in range(B // 4):
                per = []
                for tm in TEAMMATES:
                    M = matrix(data[n][data[n].teammate == tm], "score")
                    R = matrix(data["Random target"][data["Random target"].teammate == tm], "score")
                    O = matrix(data["Full-information planner"][data["Full-information planner"].teammate == tm], "score")
                    s = RNG.integers(M.shape[1], size=M.shape[1])
                    r = RNG.integers(M.shape[0], size=M.shape[0])
                    m, rr, oo = M[np.ix_(r, s)].mean(), R[:, s].mean(), O[:, s].mean()
                    per.append((m - rr) / (oo - rr))
                vals.append(np.mean(per))
            point = np.mean([(matrix(data[n][data[n].teammate == tm], "score").mean() -
                              matrix(data["Random target"][data["Random target"].teammate == tm], "score").mean()) /
                             (matrix(data["Full-information planner"][data["Full-information planner"].teammate == tm], "score").mean() -
                              matrix(data["Random target"][data["Random target"].teammate == tm], "score").mean())
                             for tm in TEAMMATES])
            lo, hi = np.percentile(vals, [2.5, 97.5])
            L.append(f"| {n} | {point:.0%} [{lo:.0%}, {hi:.0%}] |")
        L.append("")
    open(os.path.join(ROOT, "results", "rl_statistics.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    figure(fig_rows)


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [g for g in ("greedy", "noisy", "idle", "unseen") if any(r[1] == g for r in rows)]
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4.2), sharey=True)
    colours = {"MAPPO": "#1f5aa6", "IPPO": "#2e8b57", "MAA2C": "#d98c1f", "QMIX": "#8e44ad",
               "Greedy rule": "#777777", "Deadline-aware rule": "#aaaaaa", "Random target": "#cccccc",
               "Full-information planner": "#c0392b"}
    order = list(dict.fromkeys(r[0] for r in rows))
    for ax, g in zip(np.atleast_1d(axes), groups):
        rs = {r[0]: r for r in rows if r[1] == g}
        for k, n in enumerate(order):
            if n in rs:
                _, _, v, lo, hi = rs[n]
                ax.errorbar(v, k, xerr=[[v - lo], [hi - v]], fmt="o", color=colours.get(n, "#333"), capsize=4, ms=6)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.invert_yaxis()
        ax.set_title({"unseen": "unseen learners"}.get(g, f"{g} simulated human"))
        ax.set_xlabel("team score (IQM, 95% CI)")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Interquartile mean team score with 95% stratified-bootstrap confidence intervals")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "rl_statistics.png"), dpi=130)


if __name__ == "__main__":
    main()
