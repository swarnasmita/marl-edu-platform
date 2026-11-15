"""Record real decision situations from the Disaster Response game for training the pedagogical agent.

Each time the learner's robot (robot_0) has to choose a victim, the situation is recorded exactly as the learner
model sees it (learner_model.situation), and every possible choice is judged in advance (judge_choice): which
concepts that choice would show as understood / not understood, the explanation, and its decision quality.
The learner slot is played by simulated players of mixed skill so that the situations are varied.

    python learner/collect_situations.py --episodes 240
Writes learner/data/situations.pkl
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CKPT = os.path.join(ROOT, "agents", "checkpoints", "mappo_disaster.pt")
NOISE = (0.0, 0.3, 0.6, 1.0)
_W = {}


def _init():
    import torch
    torch.set_num_threads(1)
    import agents.hrl as H
    from envs.disaster_env import DisasterResponseEnv
    env = DisasterResponseEnv(observe=False)
    env.reset(seed=0)
    ctrl = H.HierarchicalController(env)
    _W.update(env=env, ctrl=ctrl, blue=H.MappoChooser(env, ctrl, CKPT))


def _episode(seed):
    import agents.hrl as H
    from learner.learner_model import judge_choice, situation
    env, ctrl, blue = _W["env"], _W["ctrl"], _W["blue"]
    rng = np.random.default_rng(seed)
    out, last_report = [], [-10 ** 6]

    class Recorder(H.Teammate):
        def choose(self, env, ctrl):
            snap = situation(env, ctrl, 0, blue, last_report[0])
            opts = snap["options"]
            if len(opts) >= 2:
                judged = {j: judge_choice(opts, j, snap["probs"], snap["since_report"]) for j in opts}
                out.append(dict(options=opts, since_report=snap["since_report"], t=snap["t"],
                                results={j: r for j, (r, _, _) in judged.items()},
                                notes={j: n for j, (_, n, _) in judged.items()},
                                quality={j: q for j, (_, _, q) in judged.items()}))
            return super().choose(env, ctrl)

    env.reset(seed=seed)
    ctrl.reset()
    mate = Recorder("greedy", rng, noise=float(NOISE[seed % len(NOISE)]))
    while env.agents:
        _, _, _, _, info = env.step(ctrl.actions(blue, mate))
        ev = info["robot_1"]["events"]
        ctrl.notify(ev)
        if any(k == "report" for k, _, _ in ev):
            last_report[0] = env.t
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=240)
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 2)))
    args = p.parse_args()
    seeds = list(range(20000, 20000 + args.episodes))      # not used anywhere else (train / val / test differ)
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        eps = pool.map(_episode, seeds)
    data = [dict(s, episode=seed) for seed, ep in zip(seeds, eps) for s in ep]
    os.makedirs(os.path.join(ROOT, "learner", "data"), exist_ok=True)
    path = os.path.join(ROOT, "learner", "data", "situations.pkl")
    pickle.dump(data, open(path, "wb"))
    tested = {}
    for s in data:
        for c in set().union(*[set(r) for r in s["results"].values()]):
            tested[c] = tested.get(c, 0) + 1
    print(f"{len(data)} situations from {len(eps)} episodes -> {path}")
    print("situations testing each concept:", tested)


if __name__ == "__main__":
    main()
