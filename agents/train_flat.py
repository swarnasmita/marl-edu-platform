"""Ablation: flat (non-hierarchical) MAPPO.

The blue robots are controlled directly by MAPPO at every simulation step (20 per second): each robot picks one
of 9 moves (stop or one of 8 directions) from the environment's own observation (90 numbers: lidar, victims,
teammates, carried victim, time; envs/disaster_env.py). There is no "which victim" decision and no route
planner - the network must learn to navigate, pick up and deliver by itself.
Everything else matches the hierarchical MAPPO: shared actor, centralised critic (both robots' observations),
team reward (scaled by 0.1), gamma 0.999 per step, GAE 0.95, PPO clip 0.2, entropy 0.01 -> 0.001, learning
rate 3e-4, 2,400 training episodes (150 iterations x 16), the same simulated humans and validation scenarios.

    python agents/train_flat.py --seed 0 --out runs/flat_seed0
    python agents/train_flat.py --evaluate runs/flat_seed0/best.pt runs/flat_seed1/best.pt ... --out results/ablations/flat
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.train_mappo import REWARD_SCALE, TEAMMATE_PROBS, VAL_SEEDS   # noqa: E402

GAMMA, LAMBDA = 0.999, 0.95
BLUE = ("robot_1", "robot_2")
TEST_SEEDS = list(range(10000, 10100))
TEAMMATES = ("greedy", "noisy", "idle")
_W = {}


def build(obs_dim=90, n_act=9, h=128):
    import torch.nn as nn

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, n_act))

        def forward(self, x):
            return self.net(x)

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2 * obs_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return Actor, Critic


def _init():
    import torch
    torch.set_num_threads(1)
    import agents.hrl as H
    from envs.disaster_env import DisasterResponseEnv
    env = DisasterResponseEnv(observe=True)
    env.reset(seed=0)
    ctrl = H.HierarchicalController(env)
    Actor, Critic = build()
    _W.update(env=env, ctrl=ctrl, actor=Actor(), critic=Critic(), torch=torch)


def _episode(job):
    """mode 'train': sample, return per-robot trajectories; 'eval': greedy actions, return metrics only."""
    a_state, c_state, seed, teammate, mode = job
    import agents.hrl as H
    from agents.train_mappo import episode_metrics
    torch, env, ctrl, actor, critic = _W["torch"], _W["env"], _W["ctrl"], _W["actor"], _W["critic"]
    actor.load_state_dict(a_state)
    if c_state is not None:
        critic.load_state_dict(c_state)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    ctrl.reset()
    mate = H.Teammate(teammate, rng)
    traj = {b: dict(obs=[], cobs=[], act=[], logp=[], val=[], rew=[]) for b in BLUE}
    gen = torch.Generator().manual_seed(int(seed) % (2 ** 31))
    while env.agents:
        acts = ctrl.actions(lambda i: H.WAIT, mate)      # only the human's move is used from the controller
        env.pickup_only[1] = env.pickup_only[2] = None   # flat robots pick up whoever they reach
        o = np.stack([obs[b] for b in BLUE]).astype(np.float32)
        with torch.no_grad():
            logits = actor(torch.from_numpy(o))
            if mode == "train":
                a = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(-1)
                lp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(-1)
                co = np.stack([np.concatenate([o[0], o[1]]), np.concatenate([o[1], o[0]])]).astype(np.float32)
                v = critic(torch.from_numpy(co))
            else:
                a = logits.argmax(-1)
        for k, b in enumerate(BLUE):
            acts[b] = int(a[k])
        obs, rew, _, _, info = env.step(acts)
        ctrl.notify(info["robot_1"]["events"])
        if mode == "train":
            for k, b in enumerate(BLUE):
                t = traj[b]
                t["obs"].append(o[k])
                t["cobs"].append(co[k])
                t["act"].append(int(a[k]))
                t["logp"].append(float(lp[k]))
                t["val"].append(float(v[k]))
                t["rew"].append(rew[b] * REWARD_SCALE)
        if not env.agents:
            break
    m = episode_metrics(env, ctrl)
    if mode != "train":
        return m, None
    out = []
    for b in BLUE:
        t = traj[b]
        r, v = np.array(t["rew"], np.float32), np.array(t["val"], np.float32)
        adv = np.zeros_like(r)
        last = 0.0
        for s in reversed(range(len(r))):
            nv = v[s + 1] if s + 1 < len(r) else 0.0
            delta = r[s] + GAMMA * nv - v[s]
            last = delta + GAMMA * LAMBDA * last
            adv[s] = last
        out.append(dict(obs=np.array(t["obs"], np.float16), cobs=np.array(t["cobs"], np.float16),
                        act=np.array(t["act"], np.int8), logp=np.array(t["logp"], np.float32),
                        adv=adv, ret=adv + v))
    return m, out


def validate(pool, actor, seeds):
    st = {k: v.clone() for k, v in actor.state_dict().items()}
    jobs = [(st, None, s, tm, "eval") for s in seeds for tm in ("greedy", "idle")]
    res = pool.map(_episode, jobs)
    return {tm: float(np.mean([r[0]["score"] for r, j in zip(res, jobs) if j[3] == tm])) for tm in ("greedy", "idle")}


def train(args):
    import torch
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    Actor, Critic = build()
    actor, critic = Actor(), Critic()
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    os.makedirs(args.out, exist_ok=True)
    kinds, probs = list(TEAMMATE_PROBS), list(TEAMMATE_PROBS.values())
    val_seeds = VAL_SEEDS[:args.val_scenarios]
    log = open(os.path.join(args.out, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["iteration", "episodes", "train_score", "train_sev", "pg_loss", "v_loss", "entropy",
                "val_greedy", "val_idle", "val_sev_greedy", "minutes"])
    best, episodes, t0 = -1e9, 0, time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        for it in range(1, args.iterations + 1):
            ent_coef = 0.01 + (0.001 - 0.01) * (it - 1) / max(1, args.iterations - 1)
            a_st = {k: v.clone() for k, v in actor.state_dict().items()}
            c_st = {k: v.clone() for k, v in critic.state_dict().items()}
            jobs = [(a_st, c_st, args.seed * 1_000_000 + episodes + k, str(rng.choice(kinds, p=probs)), "train")
                    for k in range(args.episodes_per_iter)]
            res = pool.map(_episode, jobs)
            episodes += len(jobs)
            data = [t for _, ts in res for t in ts]
            cat = lambda k, dt=torch.float32: torch.as_tensor(np.concatenate([d[k] for d in data]), dtype=dt)
            O, CO, A, LP, ADV, RET = cat("obs"), cat("cobs"), cat("act", torch.long), cat("logp"), cat("adv"), cat("ret")
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            n = len(A)
            for _ in range(4):
                perm = torch.randperm(n)
                for s in range(0, n, 4096):
                    idx = perm[s:s + 4096]
                    dist = torch.distributions.Categorical(logits=actor(O[idx]))
                    ratio = torch.exp(dist.log_prob(A[idx]) - LP[idx])
                    pg = -torch.min(ratio * ADV[idx], torch.clamp(ratio, 0.8, 1.2) * ADV[idx]).mean()
                    vl = ((critic(CO[idx]) - RET[idx]) ** 2).mean()
                    ent = dist.entropy().mean()
                    loss = pg + 0.5 * vl - ent_coef * ent
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                    opt.step()
            row = [it, episodes, np.mean([m["score"] for m, _ in res]), np.mean([m["sev"] for m, _ in res]),
                   pg.item(), vl.item(), ent.item(), "", "", "", (time.time() - t0) / 60]
            if it % args.eval_every == 0 or it == args.iterations:
                v = validate(pool, actor, val_seeds)
                row[7:9] = [v["greedy"], v["idle"]]
                mean_val = (v["greedy"] + v["idle"]) / 2
                if mean_val > best:
                    best = mean_val
                    torch.save({"flat": True, "actor": actor.state_dict(), "critic": critic.state_dict(),
                                "iteration": it, "val": v}, os.path.join(args.out, "best.pt"))
                print(f"it {it:4d} | ep {episodes:5d} | train {row[2]:6.1f} | val greedy-mate {v['greedy']:6.1f} "
                      f"idle-mate {v['idle']:6.1f} | ent {row[6]:.2f} | {row[10]:.1f} min", flush=True)
            w.writerow(row)
            log.flush()
        torch.save({"flat": True, "actor": actor.state_dict(), "critic": critic.state_dict()},
                   os.path.join(args.out, "last.pt"))
    print("done; best validation (mean of both teammates):", round(best, 1), "->", os.path.join(args.out, "best.pt"))


def evaluate(args):
    """Same 100 test scenarios x 3 simulated humans as agents/evaluate.py; writes eval_episodes.csv."""
    import torch
    rows = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        for run, ck in enumerate(args.evaluate):
            st = torch.load(ck, map_location="cpu", weights_only=False)["actor"]
            jobs = [(st, None, s, tm, "eval") for tm in TEAMMATES for s in TEST_SEEDS]
            res = pool.map(_episode, jobs, chunksize=4)
            for (_, _, s, tm, _), (m, _) in zip(jobs, res):
                rows.append(dict(policy="mappo", run=run, teammate=tm, seed=s, **m))
            print("evaluated", ck, flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "eval_episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for tm in TEAMMATES:
        print(tm, "score", round(np.mean([r["score"] for r in rows if r["teammate"] == tm]), 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--episodes-per-iter", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--val-scenarios", type=int, default=len(VAL_SEEDS))
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 2)))
    p.add_argument("--evaluate", nargs="+", default=None, help="checkpoints to evaluate on the test scenarios")
    args = p.parse_args()
    if args.evaluate:
        args.out = args.out or "results/ablations/flat"
        evaluate(args)
    else:
        args.out = args.out or f"runs/flat_seed{args.seed}"
        train(args)


if __name__ == "__main__":
    main()
