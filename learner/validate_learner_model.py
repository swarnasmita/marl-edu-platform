"""Does the learner model tell skilled and unskilled learners apart?

Simulated learners play robot_0 alongside the trained MAPPO blue robots:
  expert          chooses exactly what the trained agent would (upper reference)
  greedy          severity / distance; ignores deadlines and teammates' plans
  noisy 30% / 60% greedy, but 30% / 60% of choices are random
  random          any waiting victim at random
Plus 4 improving learners whose random choices drop from 90% to 0% over 20
episodes, each with one learner model kept across episodes (as for a real person).

    python learner/validate_learner_model.py
Writes results/learner_model_validation.md and results/learner_mastery_curve.png
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

import agents.hrl as H
from envs.disaster_env import DisasterResponseEnv
from learner.learner_model import CONCEPTS, LearnerModel

CKPT = os.path.join(ROOT, "agents", "checkpoints", "mappo_disaster.pt")
SEEDS = list(range(10000, 10030))


class ExpertMate:
    kind = "expert"

    def __init__(self, expert):
        self.expert = expert

    def choose(self, env, ctrl):
        p = self.expert.probs(0)
        return int(np.argmax(p[:H.WAIT])) if p[:H.WAIT].max() > 0 else H.WAIT


def play(env, ctrl, blue, mate, model, seed):
    env.reset(seed=seed)
    ctrl.reset()
    while env.agents:
        _, _, _, _, info = env.step(ctrl.actions(blue, mate))
        ev = info["robot_1"]["events"]
        ctrl.notify(ev)
        model.observe(ev)
    return env.score


def main():
    out = os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)
    env = DisasterResponseEnv(observe=False)
    env.reset(seed=0)
    ctrl = H.HierarchicalController(env)
    blue = H.MappoChooser(env, ctrl, CKPT)
    model = LearnerModel(env, ctrl, expert=blue)

    learners = {
        "expert": lambda s: ExpertMate(blue),
        "greedy": lambda s: H.Teammate("greedy", np.random.default_rng(s), noise=0.0),
        "noisy 30%": lambda s: H.Teammate("greedy", np.random.default_rng(s), noise=0.3),
        "noisy 60%": lambda s: H.Teammate("greedy", np.random.default_rng(s), noise=0.6),
        "random": lambda s: H.Teammate("greedy", np.random.default_rng(s), noise=1.0),
    }
    rows = []
    for name, make in learners.items():
        acc = {c: [0, 0] for c in CONCEPTS}
        quality, mastery, scores = [], {c: [] for c in CONCEPTS}, []
        for s in SEEDS:
            model.reset()
            scores.append(play(env, ctrl, blue, make(s), model, s))
            for c, v in model.evidence.items():
                acc[c][0] += sum(v)
                acc[c][1] += len(v)
                if v:
                    mastery[c].append(model.mastery[c])
            quality += model.quality
        rows.append((name, np.mean(scores), np.mean(quality) if quality else np.nan, acc,
                     {c: np.mean(m) if m else np.nan for c, m in mastery.items()}))
        print(f"{name:10s} score {np.mean(scores):6.1f} | decision quality {rows[-1][2]:.2f} | " +
              " ".join(f"{c} {a[0]}/{a[1]}" for c, a in acc.items()), flush=True)

    # improving learners: one model per learner kept across episodes (as for a real person)
    n_learners, n_eps = 4, 20
    m_hist = np.zeros((n_learners, n_eps, len(CONCEPTS)))
    q_hist = np.full((n_learners, n_eps), np.nan)
    noise_sched = [0.9 * (1 - k / (n_eps - 1)) for k in range(n_eps)]
    for L in range(n_learners):
        model.reset()
        for k, noise in enumerate(noise_sched):
            model.new_episode()
            n_before = len(model.quality)
            seed = 20000 + 100 * L + k
            play(env, ctrl, blue, H.Teammate("greedy", np.random.default_rng(seed), noise=noise), model, seed)
            m_hist[L, k] = [model.mastery[c] for c in CONCEPTS]
            if len(model.quality) > n_before:
                q_hist[L, k] = np.mean(model.quality[n_before:])
        print(f"improving learner {L + 1}: final mastery " +
              " ".join(f"{c} {model.mastery[c]:.2f}" for c in CONCEPTS), flush=True)
    curve = [(noise_sched[k], {c: m_hist[:, k, i].mean() for i, c in enumerate(CONCEPTS)},
              np.nanmean(q_hist[:, max(0, k - 2):k + 1])) for k in range(n_eps)]
    print("averaged over learners (random share -> mastery, decision quality over last 3 episodes):")
    for noise, m, q in curve:
        print(f"  random {noise:4.0%}: " + " ".join(f"{c} {v:.2f}" for c, v in m.items()) + f" | quality {q:.2f}")

    lines = ["# Learner model validation (simulated learners, 30 scenarios each)", "",
             "Blue robots: trained MAPPO. Correct / assessed choices per concept, and mean BKT mastery at the end "
             "of an episode (only episodes where the concept was tested).", "",
             "| Learner | Team score | Decision quality | " + " | ".join(CONCEPTS[c].split(" (")[0] for c in CONCEPTS) + " |",
             "|---|---|---|" + "---|" * len(CONCEPTS)]
    for name, score, q, acc, mas in rows:
        cells = []
        for c in CONCEPTS:
            k, n = acc[c]
            cells.append(f"{100 * k / n:.0f}% of {n} (mastery {mas[c]:.2f})" if n else "not tested")
        lines.append(f"| {name} | {score:.1f} | {q:.2f} | " + " | ".join(cells) + " |")
    open(os.path.join(out, "learner_model_validation.md"), "w").write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(1, len(curve) + 1)
    for c in CONCEPTS:
        ax.plot(x, [m[c] for _, m, _ in curve], "o-", label=CONCEPTS[c])
    ax.plot(x, [q for _, _, q in curve], "k--", lw=1.5, label="decision quality (last 3 episodes)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n{n:.0%}" if k % 2 else "" for k, (n, _, _) in zip(x, curve)], fontsize=8)
    ax.set_xlabel("episode  /  share of random choices by the simulated learner")
    ax.set_ylabel("estimated mastery (BKT)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Improving simulated learners (mean of {n_learners}): does the learner model notice?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "learner_mastery_curve.png"), dpi=130)
    print("wrote results/learner_model_validation.md and results/learner_mastery_curve.png")


if __name__ == "__main__":
    main()
