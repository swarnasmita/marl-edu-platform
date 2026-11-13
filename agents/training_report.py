"""Training diagnostics for one or more MAPPO runs (e.g. one per seed).

    python agents/training_report.py runs/mappo_seed0 runs/mappo_seed1 runs/mappo_seed2

Convergence episode: the first training episode at which the moving average of
the training return (10 iterations = 160 episodes) reaches 95% of its final
plateau (the mean over the last 15 iterations).

Writes results/training_report.md and results/training_diagnostics.png
"""
import csv
import os
import sys

import numpy as np

WINDOW = 10
PLATEAU_ITERS = 15


def load(run):
    rows = list(csv.DictReader(open(os.path.join(run, "log.csv"))))
    f = lambda k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
    return dict(episodes=f("episodes"), train=f("train_score"), sev=f("train_sev"), pg=f("pg_loss"),
                v=f("v_loss"), ent=f("entropy"), val_g=f("val_greedy"), val_i=f("val_idle"), minutes=f("minutes"))


def moving(x, k):
    return np.array([np.nanmean(x[max(0, i - k + 1):i + 1]) for i in range(len(x))])


def convergence(d):
    smooth = moving(d["train"], WINDOW)
    plateau = np.nanmean(d["train"][-PLATEAU_ITERS:])
    idx = np.flatnonzero(smooth >= 0.95 * plateau)
    return (float(d["episodes"][idx[0]]) if len(idx) else float("nan")), float(plateau)


def main(runs, out="results"):
    os.makedirs(out, exist_ok=True)
    data = {r: load(r) for r in runs}
    lines = ["# MAPPO training report", "",
             f"Convergence episode = first episode where the {WINDOW}-iteration moving average of the training return "
             f"reaches 95% of its plateau (mean of the last {PLATEAU_ITERS} iterations).", "",
             "| Run | Episodes | Training time (min) | Plateau return | Convergence episode | "
             "Best validation (greedy / idle teammate) | Final entropy |", "|---|---|---|---|---|---|---|"]
    conv, plat = [], []
    for r, d in data.items():
        c, p = convergence(d)
        conv.append(c)
        plat.append(p)
        vm = (d["val_g"] + d["val_i"]) / 2
        b = int(np.nanargmax(vm))
        lines.append(f"| {os.path.basename(r.rstrip('/'))} | {int(d['episodes'][-1])} | {d['minutes'][-1]:.1f} | "
                     f"{p:.1f} | {c:.0f} | {d['val_g'][b]:.1f} / {d['val_i'][b]:.1f} (episode {int(d['episodes'][b])}) | "
                     f"{d['ent'][-1]:.3f} |")
    if len(runs) > 1:
        lines += ["", f"**Across {len(runs)} seeds:** convergence at episode {np.nanmean(conv):.0f} ± {np.nanstd(conv):.0f}, "
                      f"plateau return {np.mean(plat):.1f} ± {np.std(plat):.1f}."]
    open(os.path.join(out, "training_report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    ep = next(iter(data.values()))["episodes"]

    def band(ax, key, label, colour, smooth=True):
        arr = np.array([moving(d[key], WINDOW) if smooth else d[key] for d in data.values()])
        m = np.nanmean(arr, 0)
        ax.plot(ep, m, color=colour, lw=2, label=label)
        if len(arr) > 1:
            sd = np.nanstd(arr, 0)
            ax.fill_between(ep, m - sd, m + sd, color=colour, alpha=0.2)

    band(axes[0, 0], "train", "training return (moving average)", "#1f5aa6")
    for c in conv:
        axes[0, 0].axvline(c, color="#1f5aa6", ls=":", lw=1)
    axes[0, 0].set_title("Training return" + (" (mean ± sd over seeds)" if len(runs) > 1 else ""))
    for key, col, name in (("val_g", "#1f5aa6", "greedy teammate"), ("val_i", "#c0392b", "idle teammate")):
        arr = np.array([d[key] for d in data.values()])
        mask = np.isfinite(arr[0])
        m = np.nanmean(arr[:, mask], 0)
        axes[0, 1].plot(ep[mask], m, "o-", color=col, label=name)
        if len(runs) > 1:
            sd = np.nanstd(arr[:, mask], 0)
            axes[0, 1].fill_between(ep[mask], m - sd, m + sd, color=col, alpha=0.2)
    axes[0, 1].set_title("Validation score (20 fixed scenarios)")
    band(axes[1, 0], "ent", "policy entropy", "#7d3c98", smooth=False)
    axes[1, 0].set_title("Policy entropy (exploration)")
    band(axes[1, 1], "v", "value loss", "#d35400", smooth=False)
    ax2 = axes[1, 1].twinx()
    arr = np.array([d["pg"] for d in data.values()])
    ax2.plot(ep, np.nanmean(arr, 0), color="#27ae60", lw=1, alpha=0.8, label="policy loss")
    ax2.set_ylabel("policy loss", color="#27ae60")
    axes[1, 1].set_title("Critic value loss and actor policy loss")
    for ax in axes.ravel():
        ax.set_xlabel("training episodes")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training_diagnostics.png"), dpi=130)
    print("wrote", os.path.join(out, "training_report.md"), "and training_diagnostics.png")


if __name__ == "__main__":
    runs = sys.argv[1:] or ["runs/mappo_seed0"]
    main(runs)
