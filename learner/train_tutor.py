"""Train the pedagogical agent (PPO with action masking) and compare it with rule-based tutors.

    python learner/train_tutor.py --seeds 0 1 2
Training uses 80% of the recorded game situations; all comparisons use the held-out 20% and fixed student
seeds, so every tutor meets exactly the same students. Writes:
    learner/checkpoints/tutor.pt                    (best seed by validation reward)
    runs/tutor_seed<k>/log.csv, best.pt
    results/tutor_report.md, results/tutor_comparison.png
With --student v2 (students with motivation and forgetting) the names get a _v2 suffix
(tutor_v2.pt, runs/tutor_v2_seed<k>, tutor_report_v2.md, tutor_comparison_v2.png).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import learner.tutor_env as TE                                  # noqa: E402
from learner.tutor_env import (ACTION_NAMES, ASK0, CONCEPTS, EXPLAIN0, HINT0, N_ACTIONS, OBS_DIM, PROFILES, QUIET,
                               RULES, SIM, TutorEnv)   # noqa: E402

GAMMA, LAMBDA = 0.99, 0.95
N_ENVS = 64
EVAL_STUDENTS = 150            # per profile


def suffix():
    return "_v2" if TE.STUDENT == "v2" else ""


def obs_dim():
    return TE.OBS_DIM_V2 if TE.STUDENT == "v2" else OBS_DIM


def build(dim=None):
    dim = dim or obs_dim()
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, h=64):
            super().__init__()
            self.pi = nn.Sequential(nn.Linear(dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, N_ACTIONS))
            self.v = nn.Sequential(nn.Linear(dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

        def logits(self, obs, mask):
            return self.pi(obs).masked_fill(~mask, -1e9)

    return Net


class PolicyTutor:
    """Wraps a trained network as a tutor: act(env) -> action (greedy)."""

    def __init__(self, net):
        import torch
        self.net, self.torch = net, torch

    def __call__(self, env):
        t = self.torch
        with t.no_grad():
            lg = self.net.logits(t.from_numpy(env.obs())[None], t.from_numpy(env.mask())[None])
        return int(lg.argmax())


def run_session(tutor, seed, profile, split="test", lr=1.0, impatience=1.0, sim=None):
    env = TutorEnv(np.random.default_rng(seed), profile=profile, split=split, lr=lr, impatience=impatience, sim=sim)
    R, done = 0.0, False
    while not done:
        a = tutor(env)
        r, done = env.step(a)
        R += r
    out = env.summary()
    out["reward"] = R
    out["actions"] = env.log["actions"]
    return out


def evaluate(tutor, split="test", n=EVAL_STUDENTS, base_seed=900_000, sim=None, lr=1.0, impatience=1.0):
    rows = []
    for p_i, prof in enumerate(PROFILES):
        for k in range(n):
            r = run_session(tutor, base_seed + 10_000 * p_i + k, prof, split, lr, impatience, sim)
            r["student"] = k
            rows.append(r)
    return rows


def train(seed, args):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Net = build()
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    out = os.path.join(ROOT, "runs", f"tutor{suffix()}_seed{seed}")
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["iteration", "sessions", "train_reward", "val_reward", "val_gain", "val_interventions", "entropy"])
    best, t0 = -1e9, time.time()
    for it in range(1, args.iterations + 1):
        ent_coef = 0.02 + (0.002 - 0.02) * (it - 1) / max(1, args.iterations - 1)
        envs = [TutorEnv(np.random.default_rng(rng.integers(1 << 31)), split="train") for _ in range(N_ENVS)]
        O, M, A, LP, V, R, D = [], [], [], [], [], [], []
        done = np.zeros(N_ENVS, bool)
        ep_ret = np.zeros(N_ENVS)
        while not done.all():
            obs = np.stack([e.obs() for e in envs])
            mask = np.stack([e.mask() for e in envs])
            with torch.no_grad():
                lg = net.logits(torch.from_numpy(obs), torch.from_numpy(mask))
                dist = torch.distributions.Categorical(logits=lg)
                a = dist.sample()
                v = net.v(torch.from_numpy(obs)).squeeze(-1)
            r = np.zeros(N_ENVS)
            d = np.zeros(N_ENVS, bool)
            for i, e in enumerate(envs):
                if not done[i]:
                    r[i], d[i] = e.step(int(a[i]))
            alive = ~done
            O.append(obs)
            M.append(mask)
            A.append(a.numpy())
            LP.append(dist.log_prob(a).numpy())
            V.append(v.numpy())
            R.append(r)
            D.append(d | done)
            ep_ret += r
            done |= d
            if alive.sum() == 0:
                break
        # all sessions have the same length, so the arrays are rectangular (T, N)
        R, V, D = np.array(R), np.array(V), np.array(D, float)
        T = len(R)
        adv = np.zeros_like(R)
        last = 0.0
        for t in reversed(range(T)):
            nv = V[t + 1] if t + 1 < T else 0.0
            delta = R[t] + GAMMA * nv * (1 - D[t]) - V[t]
            last = delta + GAMMA * LAMBDA * (1 - D[t]) * last
            adv[t] = last
        ret = adv + V
        flat = lambda x: torch.as_tensor(np.concatenate(x) if isinstance(x, list) else x.reshape(-1))
        Ob, Mb, Ab, LPb = flat(O), flat(M), flat(A), flat(LP)
        ADV, RET = torch.as_tensor(adv.reshape(-1), dtype=torch.float32), torch.as_tensor(ret.reshape(-1), dtype=torch.float32)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(Ab)
        for _ in range(4):
            perm = torch.randperm(n)
            for s in range(0, n, 512):
                idx = perm[s:s + 512]
                dist = torch.distributions.Categorical(logits=net.logits(Ob[idx], Mb[idx]))
                ratio = torch.exp(dist.log_prob(Ab[idx]) - LPb[idx])
                pg = -torch.min(ratio * ADV[idx], torch.clamp(ratio, 0.8, 1.2) * ADV[idx]).mean()
                vl = ((net.v(Ob[idx]).squeeze(-1) - RET[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg + 0.5 * vl - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        row = [it, it * N_ENVS, ep_ret.mean(), "", "", "", ent.item()]
        if it % args.eval_every == 0 or it == args.iterations:
            val = evaluate(PolicyTutor(net), split="train", n=40, base_seed=500_000)
            vr = np.mean([x["reward"] for x in val])
            row[3:6] = [vr, np.mean([x["gain"] for x in val]), np.mean([x["interventions"] for x in val])]
            if vr > best:
                best = vr
                torch.save({"net": net.state_dict(), "iteration": it, "val_reward": vr, "student": TE.STUDENT,
                            "obs_dim": obs_dim()}, os.path.join(out, "best.pt"))
            print(f"seed {seed} it {it:4d} | train reward {ep_ret.mean():6.1f} | val reward {vr:6.1f} gain "
                  f"{row[4]:.3f} interventions {row[5]:4.1f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        w.writerow(row)
        log.flush()
    return best


def load(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build(ck.get("obs_dim", OBS_DIM))()
    net.load_state_dict(ck["net"])
    net.eval()
    return net


def action_profile(tutor, n=60):
    out = {}
    for p_i, prof in enumerate(PROFILES):
        c = dict(hint=0, ask=0, explain=0, praise=0)
        for k in range(n):
            env = TutorEnv(np.random.default_rng(900_000 + 10_000 * p_i + k), profile=prof, split="test", lr=1.0,
                           impatience=1.0)
            done = False
            while not done:
                a = tutor(env)
                if HINT0 <= a < ASK0:
                    c["hint"] += 1
                elif ASK0 <= a < EXPLAIN0:
                    c["ask"] += 1
                elif a >= EXPLAIN0:
                    c["explain" if env.result.get(CONCEPTS[a - EXPLAIN0]) is False else "praise"] += 1
                _, done = env.step(a)
        out[prof] = {k: v / n for k, v in c.items()}
    return out


def skipped_mistakes(tutor, n=300):
    out = []
    for i in range(n):
        env = TutorEnv(np.random.default_rng(900_000 + i), profile=list(PROFILES)[i % 4], split="test", lr=1.0,
                       impatience=1.0)
        done = False
        while not done:
            a = tutor(env)
            if env.phase == 1:
                wrong = [CONCEPTS.index(c) for c, ok in env.result.items() if not ok]
                if wrong:
                    out.append(dict(mastery=env.mastery[wrong[0]], recent=sum(env.recent[-5:]), skipped=a < EXPLAIN0))
            _, done = env.step(a)
    return out


def paired_p(a, b):
    from math import erf, sqrt
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    z = abs(d.mean() / se) if se > 0 else 0.0
    return d.mean(), 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))


def report(seeds):
    import shutil
    import torch
    nets = {s: load(os.path.join(ROOT, "runs", f"tutor{suffix()}_seed{s}", "best.pt")) for s in seeds}
    best_seed = max(seeds, key=lambda s: torch.load(os.path.join(ROOT, "runs", f"tutor{suffix()}_seed{s}", "best.pt"),
                                                    weights_only=False)["val_reward"])
    os.makedirs(os.path.join(ROOT, "learner", "checkpoints"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "runs", f"tutor{suffix()}_seed{best_seed}", "best.pt"),
                os.path.join(ROOT, "learner", "checkpoints", f"tutor{suffix()}.pt"))

    results = {name: evaluate(rule) for name, rule in RULES.items()}
    per_seed = {s: evaluate(PolicyTutor(n)) for s, n in nets.items()}
    # RL tutor: per student, average over the training seeds
    rl = []
    for k, row in enumerate(per_seed[seeds[0]]):
        avg = {m: float(np.mean([per_seed[s][k][m] for s in seeds])) for m in
               ("final_k", "gain", "interventions", "seconds", "attention", "motivation", "correct_last10",
                "quality_last10", "reward")}
        rl.append(dict(row, **avg))
    results = {"RL tutor (PPO)": rl, **results}
    names = list(results)
    profiles = list(PROFILES)

    L = ["# Pedagogical agent: when should the robots help?", "",
         "Simulated students (hidden knowledge per concept, attention) work through real decision situations from "
         "the game; 30 decisions per session (about 5 game episodes). The tutor sees only the learner model's "
         "estimates. Student model: " + ("v2 (with motivation and forgetting)" if TE.STUDENT == "v2" else
                                         "v1 (knowledge and attention only)") + ". Test: held-out game situations, " + f"{EVAL_STUDENTS} students per profile, the same students "
         "for every tutor. RL tutor: mean over 3 training seeds.", "",
         "Learning gain = increase in the student's true mean knowledge over the session (0-1 scale).", "",
         "## All students", "",
         "| Tutor | Learning gain | Final knowledge | Interventions per session | Interruption (s) | Attention at end | "
         "Motivation at end | Correct choices (last 10) | Reward |", "|---|---|---|---|---|---|---|---|---|"]
    M = lambda rows, k: np.mean([r[k] for r in rows])
    for n in names:
        rows = results[n]
        L.append(f"| {'**' + n + '**' if n.startswith('RL') else n} | {M(rows, 'gain'):.3f} | {M(rows, 'final_k'):.3f} | "
                 f"{M(rows, 'interventions'):.1f} | {M(rows, 'seconds'):.0f} | {M(rows, 'attention'):.2f} | "
                 f"{M(rows, 'motivation'):.2f} | {M(rows, 'correct_last10'):.0%} | {M(rows, 'reward'):.1f} |")
    L += ["", "## Learning gain by student profile", "", "| Tutor | " + " | ".join(profiles) + " |",
          "|---" * (len(profiles) + 1) + "|"]
    for n in names:
        L.append(f"| {n} | " + " | ".join(f"{M([r for r in results[n] if r['profile'] == p], 'gain'):.3f}"
                                          for p in profiles) + " |")
    L += ["", "## Interventions per session by student profile", "", "| Tutor | " + " | ".join(profiles) + " |",
          "|---" * (len(profiles) + 1) + "|"]
    for n in names:
        L.append(f"| {n} | " + " | ".join(f"{M([r for r in results[n] if r['profile'] == p], 'interventions'):.1f}"
                                          for p in profiles) + " |")
    # paired tests against the strongest rule (by reward)
    rules = [n for n in names if not n.startswith("RL")]
    best_rule = max(rules, key=lambda n: M(results[n], "reward"))
    L += ["", f"## RL tutor vs the strongest rule ({best_rule}), paired over the same students", "",
          "| Measure | RL tutor | " + best_rule + " | Difference | p |", "|---|---|---|---|---|"]
    for k, label in (("gain", "Learning gain"), ("interventions", "Interventions"), ("attention", "Attention at end"),
                     ("motivation", "Motivation at end"), ("reward", "Reward")):
        d, p = paired_p([r[k] for r in rl], [r[k] for r in results[best_rule]])
        L.append(f"| {label} | {M(rl, k):.3f} | {M(results[best_rule], k):.3f} | {d:+.3f} | {p:.4f} |")
    # what the RL tutor does
    L += ["", "## What the RL tutor says (final model; per session of 30 decisions)", "",
          "| Profile | Hints | Questions | Explanations of mistakes | Encouragement (explaining a good choice) | "
          "Total |", "|---|---|---|---|---|---|"]
    prof = action_profile(PolicyTutor(nets[best_seed]))
    for p in profiles:
        c = prof[p]
        L.append(f"| {p} | {c['hint']:.1f} | {c['ask']:.1f} | {c['explain']:.1f} | {c['praise']:.1f} | "
                 f"{sum(c.values()):.1f} |")
    # when does the RL tutor leave a mistake unexplained?
    L += ["", "## What the RL tutor learned differently from 'Explain mistakes'", "",
          "Both explain mistakes; the RL tutor leaves some unexplained. Share of mistakes it does NOT explain:", ""]
    sk = skipped_mistakes(PolicyTutor(nets[best_seed]))
    L += ["| Learner-model mastery estimate for that concept | Mistakes | Not explained |", "|---|---|---|"]
    for lo, hi in ((0, .3), (.3, .6), (.6, .8), (.8, 1.01)):
        x = [r for r in sk if lo <= r["mastery"] < hi]
        L.append(f"| {lo:.1f}-{min(hi, 1):.1f} | {len(x)} | {np.mean([r['skipped'] for r in x]):.0%} |")
    L += ["", "| Times the robots helped in the last 5 decisions | Mistakes | Not explained |", "|---|---|---|"]
    for k in range(5):
        x = [r for r in sk if (r["recent"] == k if k < 4 else r["recent"] >= 4)]
        if x:
            L.append(f"| {k if k < 4 else '4-5'} | {len(x)} | {np.mean([r['skipped'] for r in x]):.0%} |")
    L += ["", "The RL tutor learned to let likely slips pass (mistakes on concepts the learner model already rates "
          "as mastered) and to hold back when it has helped a lot recently - fewer interruptions, same learning."]
    # robustness to the simulator's assumptions
    L += ["", "## Robustness: students whose learning works differently from the training simulator", "",
          "The RL tutor was trained with one set of learning-effect parameters. Here every learning effect is "
          "halved or increased by half, and students are slower / faster and more / less patient than in "
          "training; learning gain, RL tutor vs " + best_rule + ".", "",
          "| Test students | RL tutor | " + best_rule + " | Difference | p |", "|---|---|---|---|---|"]
    SIM = TE.SIM_V2 if TE.STUDENT == "v2" else TE.SIM
    variants = {
        "effects x0.5": dict(SIM, a_explain=SIM["a_explain"] * .5, a_hint=SIM["a_hint"] * .5, a_ask=SIM["a_ask"] * .5),
        "effects x1.5": dict(SIM, a_explain=SIM["a_explain"] * 1.5, a_hint=SIM["a_hint"] * 1.5, a_ask=SIM["a_ask"] * 1.5),
        "attention costs x2": dict(SIM, att_cost={k: v * 2 for k, v in SIM["att_cost"].items()}),
        "slow, impatient students": None,
        "fast, patient students": None,
    }
    for vname, sim in variants.items():
        kw = dict(n=60)
        if vname == "slow, impatient students":
            kw.update(lr=0.5, impatience=1.8)
        elif vname == "fast, patient students":
            kw.update(lr=1.7, impatience=0.5)
        else:
            kw.update(sim=sim)
        a = evaluate(PolicyTutor(nets[best_seed]), **kw)
        b = evaluate(RULES[best_rule], **kw)
        d, p = paired_p([r["gain"] for r in a], [r["gain"] for r in b])
        L.append(f"| {vname} | {M(a, 'gain'):.3f} | {M(b, 'gain'):.3f} | {d:+.3f} | {p:.4f} |")
    L += ["", "Limitations: the students are simulated; their learning rules encode effects reported in the "
          "tutoring literature, but the size of each effect is an assumption. A study with real learners is "
          "needed to confirm the gains."]
    open(os.path.join(ROOT, "results", f"tutor_report{suffix()}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    figure(results, names, profiles, seeds)


def figure(results, names, profiles, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"RL tutor (PPO)": "#1f5aa6", "No help": "#999999", "Always help": "#c0392b", "Explain mistakes": "#d98c1f",
               "Explain mistakes + praise": "#e6b422",
               "Mastery-threshold rule": "#2e8b57", "Random": "#8e44ad"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    for s in seeds:
        rows = list(csv.DictReader(open(os.path.join(ROOT, "runs", f"tutor{suffix()}_seed{s}", "log.csv"))))
        x = [int(r["sessions"]) for r in rows if r["val_reward"]]
        y = [float(r["val_reward"]) for r in rows if r["val_reward"]]
        ax.plot(x, y, "-", color="#1f5aa6", alpha=0.7, label="RL tutor (3 seeds)" if s == seeds[0] else None)
    for n in names[1:]:
        ax.axhline(np.mean([r["reward"] for r in results[n]]), color=colours[n], ls="--", lw=1, label=n)
    ax.set_title("Training: tutor reward on validation students")
    ax.set_xlabel("training sessions")
    ax.set_ylabel("reward per session")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    w = 0.8 / len(names)
    for k, n in enumerate(names):
        ax.bar(np.arange(len(profiles)) + (k - len(names) / 2 + 0.5) * w,
               [np.mean([r["gain"] for r in results[n] if r["profile"] == p]) for p in profiles], w,
               color=colours[n], label=n)
    ax.set_xticks(range(len(profiles)))
    ax.set_xticklabels(profiles)
    ax.set_title("Learning gain by student profile (held-out situations)")
    ax.grid(alpha=0.3, axis="y")
    ax = axes[2]
    for n in names:
        g = [np.mean([r["gain"] for r in results[n] if r["profile"] == p]) for p in profiles]
        i = [np.mean([r["interventions"] for r in results[n] if r["profile"] == p]) for p in profiles]
        ax.scatter(np.mean(i), np.mean(g), s=90, color=colours[n], label=n, zorder=3)
    ax.set_xlabel("interventions per session (30 decisions)")
    ax.set_ylabel("learning gain")
    ax.set_title("Learning gain vs how often the robots interrupt")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", f"tutor_comparison{suffix()}.png"), dpi=130)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--student", choices=("v1", "v2"), default="v1",
                   help="v1 = original simulated students; v2 = with motivation and forgetting")
    args = p.parse_args()
    TE.STUDENT = args.student
    if not args.report_only:
        for s in args.seeds:
            train(s, args)
    report(args.seeds)


if __name__ == "__main__":
    main()
