"""Train the blue robots (robot_1, robot_2) with hierarchical MAPPO.

- Actor (shared by both robots, decentralised): sees only its own robot's view
  (actor_features) and picks a victim to go for, or "wait".
- Critic (centralised, training only): sees the whole team's situation
  (critic_features) and estimates the team's future reward.
- robot_0 is played by simulated teammates of different skill (greedy, noisy,
  idle, wander) so the agents learn to work with any learner.
- Version 2 fixes found by the behaviour tests: victims that can no longer be
  reached in time are not offered while a feasible one exists, a small penalty
  for duplicating the other blue robot's target, and a feature telling each
  robot which victim the human appears to be heading for.
- Decisions happen at irregular times (semi-MDP), so rewards between decisions
  are discounted per 0.05 s step and bootstrapping uses gamma ** (steps elapsed).

Usage (laptop or Kaggle):
    python agents/train_mappo.py --seed 0 --out runs/mappo_seed0
Outputs in --out: best.pt (best on validation), last.pt, log.csv, learning_curve.png
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

GAMMA = 0.999          # per 0.05 s step (about 0.98 per second)
LAMBDA = 0.95          # GAE, per decision
REWARD_SCALE = 0.1
DUP_PENALTY = 1.0      # fix 2: choosing the victim the other blue robot is already heading for (reward units)
TEAMMATE_PROBS = {"greedy": 0.4, "noisy": 0.3, "idle": 0.15, "wander": 0.15}
VAL_SEEDS = list(range(5000, 5020))      # validation scenarios (test scenarios are 10000+ in evaluate.py)

_W = {}                # per-worker state


# --------------------------------------------------------------------------- worker side
def _init_worker():
    import torch
    torch.set_num_threads(1)
    from agents.hrl import HierarchicalController, build_networks
    from envs.disaster_env import DisasterResponseEnv
    env = DisasterResponseEnv(observe=False)
    env.reset(seed=0)
    ctrl = HierarchicalController(env)
    for p in env.spawn_points:                   # pre-compute route maps to every victim location
        ctrl.planner.field(p[:2])
    Actor, Critic = build_networks()
    _W.update(env=env, ctrl=ctrl, actor=Actor(), critic=Critic(), torch=torch)


def _episode(args):
    """Run one episode. mode = 'train' (sample actions, record transitions) or a rule/eval mode."""
    actor_state, critic_state, seed, teammate, mode = args
    import agents.hrl as H
    from envs.disaster_env import N_VICTIMS, RESCUED
    torch, env, ctrl = _W["torch"], _W["env"], _W["ctrl"]
    actor, critic = _W["actor"], _W["critic"]
    if actor_state is not None:
        actor.load_state_dict(actor_state)
        critic.load_state_dict(critic_state)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    ctrl.reset()
    mate = H.Teammate(teammate, rng)

    trajs = {i: [] for i in H.RL_ROBOTS}
    pending = {i: None for i in H.RL_ROBOTS}

    def choose(i):
        if mode == "greedy":
            return H.greedy_choice(env, ctrl, i)
        if mode == "feasible":
            return H.greedy_choice(env, ctrl, i, feasible_only=True)
        if mode == "random":
            return H.random_choice(env, ctrl, i, rng)
        V, G, mask = H.actor_features(env, ctrl, i)
        with torch.no_grad():
            logits = actor(torch.from_numpy(V), torch.from_numpy(G), torch.from_numpy(mask))
        if mode == "eval":
            return int(torch.argmax(logits))
        dist = torch.distributions.Categorical(logits=logits)
        a = int(dist.sample())
        Vc, Gc = H.critic_features(env, ctrl, i, teammate)
        with torch.no_grad():
            value = float(critic(torch.from_numpy(Vc), torch.from_numpy(Gc)))
        if pending[i] is not None:
            pending[i]["next_value"] = value
            trajs[i].append(pending[i])
        other = [k for k in H.RL_ROBOTS if k != i][0]
        penalty = DUP_PENALTY * REWARD_SCALE if (a != H.WAIT and ctrl.target[other] == a) else 0.0
        pending[i] = dict(V=V, G=G, mask=mask, Vc=Vc, Gc=Gc, a=a, logp=float(dist.log_prob(torch.tensor(a))),
                          value=value, R=-penalty, disc=1.0, next_value=0.0, done=0.0)
        return a

    while env.agents:
        acts = ctrl.actions(choose, mate)
        _, rew, _, _, info = env.step(acts)
        ctrl.notify(info["robot_1"]["events"])
        r = rew["robot_1"] * REWARD_SCALE
        for i in H.RL_ROBOTS:
            if pending[i] is not None:
                pending[i]["R"] += pending[i]["disc"] * r
                pending[i]["disc"] *= GAMMA

    stats = episode_metrics(env, ctrl)
    if mode != "train":
        return stats, None

    for i in H.RL_ROBOTS:                         # close the last decision of each robot
        if pending[i] is not None:
            pending[i]["done"] = 1.0
            trajs[i].append(pending[i])
    batch = []
    for i in H.RL_ROBOTS:                         # semi-MDP GAE over each robot's decisions
        adv = 0.0
        for t in reversed(trajs[i]):
            nonterm = 1.0 - t["done"]
            delta = t["R"] + t["disc"] * t["next_value"] * nonterm - t["value"]
            adv = delta + t["disc"] * LAMBDA * nonterm * adv
            t["adv"], t["ret"] = adv, adv + t["value"]
            batch.append(t)
    return stats, batch


def episode_metrics(env, ctrl):
    """Task and coordination metrics for one finished episode."""
    from envs.disaster_env import DT, N_VICTIMS, RESCUED
    sev, status = env.severity, env.status
    total = float(np.sum(sev))
    saved = float(sum(sev[j] for j in range(N_VICTIMS) if status[j] == RESCUED))
    crit = [j for j in range(N_VICTIMS) if sev[j] == 3]
    picked = [j for j in range(N_VICTIMS) if env.pickup_time[j] >= 0]
    def wait_s(level):          # seconds from being reported to being picked up
        w = [(env.pickup_time[j] - env.report_time[j]) * DT for j in picked if sev[j] == level]
        return float(np.mean(w)) if w else float("nan")
    by = [int(np.sum(env.rescued_by == k)) for k in range(3)]
    blue_decisions = ctrl.decisions[1] + ctrl.decisions[2]
    return dict(
        score=float(env.score), sev=saved / total, success=float(saved / total >= 0.8),
        rescued=int(env.rescued_count), lost=int(env.lost_count),
        crit_lost=int(sum(1 for j in crit if status[j] == 3)),
        crit_survival=float(np.mean([status[j] == RESCUED for j in crit])),
        wait_critical=wait_s(3), wait_moderate=wait_s(2), wait_minor=wait_s(1),
        decisions=int(blue_decisions),
        duplicate_rate=float((ctrl.duplicate_choices[1] + ctrl.duplicate_choices[2]) / max(blue_decisions, 1)),
        wasted_trips=int(ctrl.wasted_trips[1] + ctrl.wasted_trips[2]),
        dup_time=float(ctrl.dup_steps * DT),
        idle_time=float((ctrl.idle_steps[1] + ctrl.idle_steps[2]) / 2 * DT),
        share_learner=by[0] / max(sum(by), 1), share_blue=(by[1] + by[2]) / max(sum(by), 1),
        balance=float(min(by[1], by[2]) / max(by[1], by[2], 1)),
    )


# --------------------------------------------------------------------------- learner side
def ppo_update(actor, critic, opt, batch, args, ent_coef):
    import torch
    cat = lambda k, dt=torch.float32: torch.as_tensor(np.stack([t[k] for t in batch]), dtype=dt)
    V, G, M, Vc, Gc = cat("V"), cat("G"), cat("mask", torch.bool), cat("Vc"), cat("Gc")
    A = torch.as_tensor([t["a"] for t in batch])
    old_logp = torch.as_tensor([t["logp"] for t in batch])
    adv = torch.as_tensor([t["adv"] for t in batch], dtype=torch.float32)
    ret = torch.as_tensor([t["ret"] for t in batch], dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(batch)
    stats = []
    for _ in range(args.epochs):
        perm = torch.randperm(n)
        for s in range(0, n, args.minibatch):
            idx = perm[s:s + args.minibatch]
            dist = torch.distributions.Categorical(logits=actor(V[idx], G[idx], M[idx]))
            ratio = torch.exp(dist.log_prob(A[idx]) - old_logp[idx])
            pg = -torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[idx]).mean()
            vloss = ((critic(Vc[idx], Gc[idx]) - ret[idx]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + 0.5 * vloss - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
            opt.step()
            stats.append((pg.item(), vloss.item(), ent.item()))
    return np.mean(stats, axis=0)


def evaluate(pool, actor, critic, seeds, mode="eval"):
    a_state = {k: v.clone() for k, v in actor.state_dict().items()} if actor is not None else None
    c_state = {k: v.clone() for k, v in critic.state_dict().items()} if critic is not None else None
    jobs = [(a_state, c_state, s, tm, mode) for s in seeds for tm in ("greedy", "idle")]
    res = pool.map(_episode, jobs)
    out = {}
    for tm in ("greedy", "idle"):
        rs = [r[0] for r, j in zip(res, jobs) if j[3] == tm]
        out[tm] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    return out


def plot(log_path, baselines, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(log_path)))
    ep = np.array([float(r["episodes"]) for r in rows])
    tr = np.array([float(r["train_score"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    k = max(1, len(tr) // 25)
    smooth = np.convolve(tr, np.ones(k) / k, mode="valid")
    axes[0].plot(ep, tr, color="#9db4d6", lw=1, label="training (sampled actions, mixed teammates)")
    axes[0].plot(ep[k - 1:], smooth, color="#1f5aa6", lw=2, label=f"moving average ({k})")
    axes[0].set_title("Training return per episode")
    vrows = [r for r in rows if r["val_greedy"]]
    ve = [float(r["episodes"]) for r in vrows]
    for key, col, name in (("val_greedy", "#1f5aa6", "with greedy teammate"), ("val_idle", "#c0392b", "with idle teammate")):
        axes[1].plot(ve, [float(r[key]) for r in vrows], "o-", color=col, label=f"MAPPO {name}")
        tm = key.split("_")[1]
        axes[1].axhline(baselines["greedy"][tm]["score"], color=col, ls="--", lw=1, label=f"greedy rule {name}")
        axes[1].axhline(baselines["feasible"][tm]["score"], color=col, ls=":", lw=1, label=f"deadline-aware rule {name}")
    axes[1].set_title("Validation score (20 fixed scenarios, best action)")
    for ax in axes:
        ax.set_xlabel("training episodes")
        ax.set_ylabel("team score")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/mappo_seed0")
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--episodes-per-iter", type=int, default=16)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2))))
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--val-scenarios", type=int, default=len(VAL_SEEDS))
    args = p.parse_args()
    val_seeds = VAL_SEEDS[:args.val_scenarios]

    import torch
    from agents.hrl import MODEL_VERSION as H_VERSION_, build_networks
    global H_VERSION
    H_VERSION = H_VERSION_
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    Actor, Critic = build_networks()
    actor, critic = Actor(), Critic()
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)
    kinds, probs = list(TEAMMATE_PROBS), list(TEAMMATE_PROBS.values())

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker) as pool:
        print(f"workers={args.workers}; evaluating rule baselines on validation scenarios...", flush=True)
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
            a_state = {k: v.clone() for k, v in actor.state_dict().items()}
            c_state = {k: v.clone() for k, v in critic.state_dict().items()}
            jobs = [(a_state, c_state, args.seed * 1_000_000 + episodes + k, str(rng.choice(kinds, p=probs)), "train")
                    for k in range(args.episodes_per_iter)]
            results = pool.map(_episode, jobs)
            episodes += len(jobs)
            batch = [t for _, b in results for t in b]
            frac = it / args.iterations
            ent_coef = 0.02 * (1 - frac) + 0.003 * frac
            pg, vl, ent = ppo_update(actor, critic, opt, batch, args, ent_coef)
            row = dict(iteration=it, episodes=episodes, train_score=np.mean([r["score"] for r, _ in results]),
                       train_sev=np.mean([r["sev"] for r, _ in results]), pg_loss=pg, v_loss=vl, entropy=ent,
                       val_greedy="", val_idle="", val_sev_greedy="", minutes=(time.time() - t0) / 60)
            if it % args.eval_every == 0 or it == args.iterations:
                v = evaluate(pool, actor, critic, val_seeds)
                row.update(val_greedy=v["greedy"]["score"], val_idle=v["idle"]["score"],
                           val_sev_greedy=v["greedy"]["sev"])
                mean_val = (v["greedy"]["score"] + v["idle"]["score"]) / 2
                if mean_val > best:
                    best = mean_val
                    torch.save({"version": H_VERSION, "actor": actor.state_dict(), "critic": critic.state_dict(), "iteration": it,
                                "episodes": episodes, "val": v}, os.path.join(args.out, "best.pt"))
                print(f"it {it:4d} | ep {episodes:5d} | train {row['train_score']:6.1f} | val greedy-mate "
                      f"{v['greedy']['score']:6.1f} idle-mate {v['idle']['score']:6.1f} | ent {ent:.2f} | "
                      f"{row['minutes']:.1f} min", flush=True)
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fields).writerow(row)
        torch.save({"version": H_VERSION, "actor": actor.state_dict(), "critic": critic.state_dict(),
                    "iteration": args.iterations,
                    "episodes": episodes}, os.path.join(args.out, "last.pt"))
    plot(log_path, baselines, os.path.join(args.out, "learning_curve.png"))
    print("done; best validation (mean of both teammates):", round(best, 1), "->", os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
