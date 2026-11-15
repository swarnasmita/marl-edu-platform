"""Check the pedagogical agent inside the real game (MuJoCo), not only in the fast tutoring simulator.

Simulated students (same model as in tutor_env.py) play the learner's robot in real Disaster Response episodes
with the trained blue robots. At every real decision of the student, the tutor sees the live situation and the
learner-model state, and may hint / ask before the choice and explain after it - exactly as in the platform.
Here the situations come from actual play (the student's own choices shape what happens next), with the real
time pressure and the blue robots moving.

    python learner/validate_tutor_in_game.py --students 20 --episodes 5
Writes results/tutor_in_game.md
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from learner.tutor_env import CONCEPTS, PROFILES, RULES, TutorEnv, tested   # noqa: E402

CKPT = os.path.join(ROOT, "agents", "checkpoints", "mappo_disaster.pt")
TUTOR = os.path.join(ROOT, "learner", "checkpoints", "tutor.pt")
_W = {}


def _init():
    import torch
    torch.set_num_threads(1)
    import agents.hrl as H
    from envs.disaster_env import DisasterResponseEnv
    from learner.train_tutor import PolicyTutor, load
    env = DisasterResponseEnv(observe=False)
    env.reset(seed=0)
    ctrl = H.HierarchicalController(env)
    tutors = dict(RULES)
    tutors["RL tutor (PPO)"] = PolicyTutor(load(TUTOR))
    _W.update(env=env, ctrl=ctrl, blue=H.MappoChooser(env, ctrl, CKPT), tutors=tutors)


def _student(job):
    """One simulated student, several game episodes with one tutor."""
    tutor_name, profile, k, n_eps = job
    import agents.hrl as H
    from learner.learner_model import judge_choice, situation
    env, ctrl, blue, tutor = _W["env"], _W["ctrl"], _W["blue"], _W["tutors"][tutor_name]
    seed = 700_000 + 1000 * list(PROFILES).index(profile) + k
    T = TutorEnv(np.random.default_rng(seed), profile=profile, lr=1.0, impatience=1.0)   # holds the student + tutor state
    K0 = T.student.K.copy()
    scores, last_report = [], [-10 ** 6]

    class StudentMate(H.Teammate):
        def choose(self, env_, ctrl_):
            snap = situation(env, ctrl, 0, blue, last_report[0])
            opts = snap["options"]
            if len(opts) < 2:
                return super().choose(env_, ctrl_)
            judged = {j: judge_choice(opts, j, snap["probs"], snap["since_report"]) for j in opts}
            s = dict(options=opts, results={j: r for j, (r, _, _) in judged.items()},
                     quality={j: q for j, (_, _, q) in judged.items()})
            T.s, T.tested = s, [CONCEPTS.index(c) for c in tested(s)]
            T.boost, T.hinted, T.result = {}, set(), {}
            T.step(tutor(T))                    # before the choice: quiet / hint / ask; the student then chooses
            choice = T.chosen
            T.step(tutor(T))                    # after the choice: quiet / explain
            return choice

    for e in range(n_eps):
        env.reset(seed=seed * 10 + e)
        ctrl.reset()
        mate = StudentMate("greedy", np.random.default_rng(seed + e))
        while env.agents:
            _, _, _, _, info = env.step(ctrl.actions(blue, mate))
            ev = info["robot_1"]["events"]
            ctrl.notify(ev)
            if any(kd == "report" for kd, _, _ in ev):
                last_report[0] = env.t
        scores.append(env.score)
    return dict(tutor=tutor_name, profile=profile, student=k, gain=float(T.student.K.mean() - K0.mean()),
                final_k=float(T.student.K.mean()), interventions=T.log["interventions"], seconds=T.log["seconds"],
                attention=float(T.student.att), decisions=T.k, score=float(np.mean(scores)),
                correct_last=float(np.mean(T.log["correct"][-10:])) if T.log["correct"] else np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--students", type=int, default=20)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--tutors", nargs="+", default=["RL tutor (PPO)", "Explain mistakes", "Mastery-threshold rule",
                                                   "No help"])
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 2)))
    args = p.parse_args()
    jobs = [(t, prof, k, args.episodes) for t in args.tutors for prof in PROFILES for k in range(args.students)]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        rows = pool.map(_student, jobs, chunksize=1)
    import pandas as pd
    from learner.train_tutor import paired_p
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(ROOT, "results", "tutor_in_game.csv"), index=False)
    L = ["# Pedagogical agent inside the real game", "",
         f"Simulated students played the learner's robot in real MuJoCo episodes ({args.episodes} episodes each, "
         f"{args.students} students per profile, the same students for every tutor), with the trained blue robots. "
         "The tutor acted at every real decision of the student.", "",
         "| Tutor | Learning gain | Decisions | Interventions | Interruption (s) | Attention at end | Team score |",
         "|---|---|---|---|---|---|---|"]
    for t in args.tutors:
        x = d[d.tutor == t]
        L.append(f"| {t} | {x.gain.mean():.3f} | {x.decisions.mean():.0f} | {x.interventions.mean():.1f} | "
                 f"{x.seconds.mean():.0f} | {x.attention.mean():.2f} | {x.score.mean():.1f} |")
    L += ["", "## Learning gain by profile", "", "| Tutor | " + " | ".join(PROFILES) + " |",
          "|---" * (len(PROFILES) + 1) + "|"]
    for t in args.tutors:
        L.append(f"| {t} | " + " | ".join(f"{d[(d.tutor == t) & (d.profile == p)].gain.mean():.3f}" for p in PROFILES) + " |")
    ref = d[d.tutor == "RL tutor (PPO)"].sort_values(["profile", "student"])
    L += ["", "## RL tutor vs each rule (paired over the same students)", "",
          "| Rule | Gain difference (RL - rule) | p | Interventions difference | Team score difference | p |",
          "|---|---|---|---|---|---|"]
    for t in args.tutors:
        if t == "RL tutor (PPO)":
            continue
        o = d[d.tutor == t].sort_values(["profile", "student"])
        g, pg = paired_p(ref.gain.values, o.gain.values)
        iv, _ = paired_p(ref.interventions.values, o.interventions.values)
        sc, ps = paired_p(ref.score.values, o.score.values)
        L.append(f"| {t} | {g:+.3f} | {pg:.4f} | {iv:+.1f} | {sc:+.1f} | {ps:.3f} |")
    open(os.path.join(ROOT, "results", "tutor_in_game.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
