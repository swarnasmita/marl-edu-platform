"""Train the blue robots with hierarchical QMIX (value-based baseline for the algorithm comparison).

Same task, features, network size, reward, number of episodes, simulated teammates and validation as
agents/train_mappo.py; only the learning algorithm differs.

- Agent network (shared by R1 and R2): the same deep-set network as the MAPPO actor, but its 11 outputs are
  Q-values (one per victim + wait) instead of action preferences. The robot picks the highest Q-value
  (epsilon-greedy while training).
- Mixing network: combines Q_1 and Q_2 into the team value Q_tot with non-negative weights produced by
  hypernetworks from the global state (the critic's view of the whole team), as in QMIX (Rashid et al., 2018).
- Asynchronous decisions (macro-actions): R1 and R2 choose targets at different moments. A team transition is
  recorded whenever either robot decides; a robot that is still busy keeps its running choice (and the view it
  had when it made it). The target uses the next decision moment:
      y = R + gamma^(steps elapsed) * Q_tot'(s', q'_1, q'_2)
  where a robot deciding at s' takes its best action (double Q-learning: chosen by the online network, valued by
  the target network) and a busy robot keeps its running choice (cf. macro-action MARL, Xiao et al., 2020).
- Replay buffer, epsilon-greedy exploration (1.0 -> 0.05 over the first half of training), target network
  copied every iteration.

    python agents/train_qmix.py --seed 0 --out runs/qmix_seed0
Outputs: best.pt (agent network saved as "actor", so evaluate.py and the viewer can use it), last.pt, log.csv,
learning_curve.png. In log.csv, "entropy" holds epsilon and pg_loss / v_loss hold the TD loss.
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from agents.train_mappo import (DUP_PENALTY, GAMMA, REWARD_SCALE, TEAMMATE_PROBS, VAL_SEEDS, _W, _init_worker,
                                episode_metrics, evaluate, plot)

BUFFER = 30_000           # team transitions kept (about 500 episodes)
LEARN_START = 2_000       # transitions collected before the first update
EPS_START, EPS_END, EPS_FRACTION = 1.0, 0.05, 0.5


def _episode_q(args):
    """One training episode with epsilon-greedy Q-values. Returns (metrics, list of team transitions)."""
    q_state, seed, teammate, eps = args
    import agents.hrl as H
    torch, env, ctrl, qnet = _W["torch"], _W["env"], _W["ctrl"], _W["actor"]
    qnet.load_state_dict(q_state)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    ctrl.reset()
    mate = H.Teammate(teammate, rng)

    events = []                     # one per env step at which at least one blue robot decided
    cur = {}                        # robot -> (V, G, mask, action) of its running choice

    def choose(i):
        V, G, mask = H.actor_features(env, ctrl, i)
        if rng.random() < eps:
            a = int(rng.choice(np.flatnonzero(mask)))
        else:
            with torch.no_grad():
                a = int(torch.argmax(qnet(torch.from_numpy(V), torch.from_numpy(G), torch.from_numpy(mask))))
        if not events or events[-1]["t"] != env.t:          # first decision this step: new team transition
            Vs, Gs = H.critic_features(env, ctrl, 1, teammate)
            ev = dict(t=env.t, Vs=Vs, Gs=Gs, R=0.0, disc=1.0, done=0.0)
            for k in H.RL_ROBOTS:
                if k in cur:
                    ev[k] = cur[k]
                    ev[f"d{k}"] = 0.0
            events.append(ev)
        ev = events[-1]
        other = [k for k in H.RL_ROBOTS if k != i][0]
        if a != H.WAIT and ctrl.target[other] == a:
            ev["R"] -= DUP_PENALTY * REWARD_SCALE
        cur[i] = (V, G, mask, a)
        ev[i] = cur[i]
        ev[f"d{i}"] = 1.0
        return a

    while env.agents:
        acts = ctrl.actions(choose, mate)
        _, rew, _, _, info = env.step(acts)
        ctrl.notify(info["robot_1"]["events"])
        if events:
            ev = events[-1]
            ev["R"] += ev["disc"] * rew["robot_1"] * REWARD_SCALE
            ev["disc"] *= GAMMA

    events = [e for e in events if all(k in e for k in H.RL_ROBOTS)]   # both robots have a running choice
    out = []
    for n, ev in enumerate(events):
        nxt = events[n + 1] if n + 1 < len(events) else ev
        done = 0.0 if n + 1 < len(events) else 1.0
        out.append(dict(Vs=ev["Vs"], Gs=ev["Gs"], R=ev["R"], disc=ev["disc"], done=done,
                        V1=ev[1][0], G1=ev[1][1], M1=ev[1][2], a1=ev[1][3],
                        V2=ev[2][0], G2=ev[2][1], M2=ev[2][2], a2=ev[2][3],
                        nVs=nxt["Vs"], nGs=nxt["Gs"],
                        nV1=nxt[1][0], nG1=nxt[1][1], nM1=nxt[1][2], na1=nxt[1][3], nd1=nxt["d1"],
                        nV2=nxt[2][0], nG2=nxt[2][1], nM2=nxt[2][2], na2=nxt[2][3], nd2=nxt["d2"]))
    return episode_metrics(env, ctrl), out


def build_mixer():
    import torch
    import torch.nn as nn
    from agents.hrl import CRITIC_GLOBAL_DIM, CRITIC_VICTIM_DIM

    class Mixer(nn.Module):
        """QMIX mixing network. The global state is embedded with the same deep-set body as the critic;
        hypernetworks turn that embedding into non-negative mixing weights (so Q_tot rises with each Q_i)."""

        def __init__(self, h=64, m=32):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(CRITIC_VICTIM_DIM, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
            self.ctx = nn.Sequential(nn.Linear(h + CRITIC_GLOBAL_DIM, h), nn.ReLU())
            self.w1, self.b1 = nn.Linear(h, 2 * m), nn.Linear(h, m)
            self.w2 = nn.Linear(h, m)
            self.b2 = nn.Sequential(nn.Linear(h, m), nn.ReLU(), nn.Linear(m, 1))
            self.m = m

        def forward(self, qs, Vs, Gs):             # qs: (B, 2)
            valid = Vs[..., :1]
            e = self.enc(Vs) * valid
            c = self.ctx(torch.cat([e.sum(-2) / valid.sum(-2).clamp(min=1.0), Gs], -1))
            w1 = torch.abs(self.w1(c)).view(-1, 2, self.m)
            hidden = torch.nn.functional.elu(torch.bmm(qs.unsqueeze(1), w1).squeeze(1) + self.b1(c))
            return (hidden * torch.abs(self.w2(c))).sum(-1) + self.b2(c).squeeze(-1)

    return Mixer


def td_update(qnet, qtarget, mixer, mtarget, opt, buf, rng, args):
    import torch
    idx = rng.choice(len(buf), size=min(args.minibatch, len(buf)), replace=False)
    batch = [buf[k] for k in idx]
    T = lambda k, dt=torch.float32: torch.as_tensor(np.stack([b[k] for b in batch]), dtype=dt)
    L = lambda k: torch.as_tensor([b[k] for b in batch], dtype=torch.long)
    F = lambda k: torch.as_tensor([b[k] for b in batch], dtype=torch.float32)

    q1 = qnet(T("V1"), T("G1"), T("M1", torch.bool)).gather(1, L("a1")[:, None]).squeeze(1)
    q2 = qnet(T("V2"), T("G2"), T("M2", torch.bool)).gather(1, L("a2")[:, None]).squeeze(1)
    q_tot = mixer(torch.stack([q1, q2], 1), T("Vs"), T("Gs"))
    with torch.no_grad():
        nexts = []
        for k in (1, 2):
            V, G, M = T(f"nV{k}"), T(f"nG{k}"), T(f"nM{k}", torch.bool)
            best = qnet(V, G, M).argmax(1)                       # double Q: online network picks ...
            a = torch.where(F(f"nd{k}") > 0, best, L(f"na{k}"))   # ... busy robots keep their running choice
            nexts.append(qtarget(V, G, M).gather(1, a[:, None]).squeeze(1))   # ... target network values it
        y = F("R") + F("disc") * (1 - F("done")) * mtarget(torch.stack(nexts, 1), T("nVs"), T("nGs"))
    loss = ((q_tot - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(qnet.parameters()) + list(mixer.parameters()), 10.0)
    opt.step()
    return loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default: runs/qmix_seed<seed>")
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--episodes-per-iter", type=int, default=16)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2))))
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--updates-per-iter", type=int, default=50)
    p.add_argument("--val-scenarios", type=int, default=len(VAL_SEEDS))
    args = p.parse_args()
    args.out = args.out or f"runs/qmix_seed{args.seed}"
    val_seeds = VAL_SEEDS[:args.val_scenarios]

    import copy

    import torch
    from agents.hrl import MODEL_VERSION, build_networks
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    QNet, _ = build_networks()
    qnet, mixer = QNet(), build_mixer()()
    qtarget, mtarget = copy.deepcopy(qnet), copy.deepcopy(mixer)
    opt = torch.optim.Adam(list(qnet.parameters()) + list(mixer.parameters()), lr=args.lr)
    kinds, probs = list(TEAMMATE_PROBS), list(TEAMMATE_PROBS.values())
    buf = []

    def save(path, it, episodes, v=None):
        torch.save({"version": MODEL_VERSION, "algo": "qmix", "actor": qnet.state_dict(), "critic": mixer.state_dict(),
                    "iteration": it, "episodes": episodes, "val": v}, path)

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker) as pool:
        print(f"algo=qmix; workers={args.workers}; evaluating rule baselines on validation scenarios...", flush=True)
        baselines = {m: evaluate(pool, None, None, val_seeds, mode=m) for m in ("greedy", "feasible")}
        for m, b in baselines.items():
            print(f"  {m:9s} rule: score {b['greedy']['score']:.1f} (greedy mate) / {b['idle']['score']:.1f} (idle mate)",
                  flush=True)
        log_path = os.path.join(args.out, "log.csv")
        fields = ["iteration", "episodes", "train_score", "train_sev", "pg_loss", "v_loss", "entropy",
                  "val_greedy", "val_idle", "val_sev_greedy", "minutes"]
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()
        best, episodes, t0 = -1e9, 0, time.time()
        for it in range(1, args.iterations + 1):
            eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (it - 1) / (EPS_FRACTION * args.iterations))
            q_state = {k: v.clone() for k, v in qnet.state_dict().items()}
            jobs = [(q_state, args.seed * 1_000_000 + episodes + k, str(rng.choice(kinds, p=probs)), eps)
                    for k in range(args.episodes_per_iter)]
            results = pool.map(_episode_q, jobs)
            episodes += len(jobs)
            for _, trans in results:
                buf.extend(trans)
            del buf[:max(0, len(buf) - BUFFER)]
            losses = []
            if len(buf) >= LEARN_START:
                for _ in range(args.updates_per_iter):
                    losses.append(td_update(qnet, qtarget, mixer, mtarget, opt, buf, rng, args))
                qtarget.load_state_dict(qnet.state_dict())
                mtarget.load_state_dict(mixer.state_dict())
            td = float(np.mean(losses)) if losses else float("nan")
            row = dict(iteration=it, episodes=episodes, train_score=np.mean([r["score"] for r, _ in results]),
                       train_sev=np.mean([r["sev"] for r, _ in results]), pg_loss=td, v_loss=td, entropy=eps,
                       val_greedy="", val_idle="", val_sev_greedy="", minutes=(time.time() - t0) / 60)
            if it % args.eval_every == 0 or it == args.iterations:
                v = evaluate(pool, qnet, None, val_seeds)
                row.update(val_greedy=v["greedy"]["score"], val_idle=v["idle"]["score"],
                           val_sev_greedy=v["greedy"]["sev"])
                mean_val = (v["greedy"]["score"] + v["idle"]["score"]) / 2
                if mean_val > best:
                    best = mean_val
                    save(os.path.join(args.out, "best.pt"), it, episodes, v)
                print(f"it {it:4d} | ep {episodes:5d} | train {row['train_score']:6.1f} | val greedy-mate "
                      f"{v['greedy']['score']:6.1f} idle-mate {v['idle']['score']:6.1f} | eps {eps:.2f} | "
                      f"td {td:.3f} | {row['minutes']:.1f} min", flush=True)
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fields).writerow(row)
        save(os.path.join(args.out, "last.pt"), args.iterations, episodes)
    plot(log_path, baselines, os.path.join(args.out, "learning_curve.png"), name="QMIX")
    print("done; best validation (mean of both teammates):", round(best, 1), "->", os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
