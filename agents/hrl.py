"""Hierarchical control for the Disaster Response environment.

High level (learned, or a rule for baselines): whenever a robot is free, choose
which victim to go for next. "Wait 1 s" is only available when no victim is
reachable (a free rescue robot should always be heading for someone). Re-decide when the target is taken or
lost, when a new victim is reported, or when a wait ends. While carrying, a
robot always heads to triage.

Low level (fixed): the path planner and collision avoidance from
heuristic_policy.py drive the robot to the chosen goal.

This module is shared by training (train_mappo.py), evaluation (evaluate.py)
and the viewer, so every policy is compared with identical navigation.
"""
import numpy as np

from agents.heuristic_policy import GreedyPolicy, GridPlanner
from envs.disaster_env import (DIRS, DT, MAX_STEPS, N_ROBOTS, N_VICTIMS, SAFE_CENTER,
                               SPEED, TIME_LIMIT, UNREPORTED, WAITING)

WAIT = N_VICTIMS                # action index for "wait 1 s"
N_ACTIONS = N_VICTIMS + 1
WAIT_STEPS = 20                 # 1 s
RL_ROBOTS = (1, 2)              # robot_0 is the learner's slot (human or simulated teammate)
DIST_SCALE = 40.0               # metres, for normalising path lengths
TIME_SCALE = TIME_LIMIT[2] * DT     # 45 s
NEW_STEPS = 100                 # a report counts as "new" for 5 s

MODEL_VERSION = 2              # v2: deadline masking, duplicate penalty, human-intent feature
ACTOR_VICTIM_DIM = 13
ACTOR_GLOBAL_DIM = 7
CRITIC_VICTIM_DIM = 14
INTENT_WINDOW = 30              # steps (1.5 s) of movement used to infer where the human is heading
INTENT_MIN_DROP = 1.0           # metres of route distance gained towards a victim in that window
CRITIC_GLOBAL_DIM = 11
TEAMMATE_TYPES = ("greedy", "noisy", "idle", "wander")


# --------------------------------------------------------------------------- navigation
class Navigator:
    """Path-following with collision avoidance (reuses the greedy policy's motion code)."""

    def __init__(self, env, planner):
        self.env = env
        self.mover = GreedyPolicy(env, planner=planner)
        self.planner = planner
        self.reset()

    def reset(self):
        self.mover.reset()
        self.anchor, self.counter, self.escape = {}, {}, {}
        self.escape_count = 0                         # unjam manoeuvres this episode

    def move(self, i, goal, pos):
        if self.escape.get(i, (0, 0))[0] > 0:
            left, a = self.escape[i]
            self.escape[i] = (left - 1, a)
            return a
        if goal is None:
            self.counter[i] = 0
            return 0
        a = self.mover.best_move(i, pos, goal)
        self.mover.prev[i] = a
        # unjam if two robots block each other
        self.counter[i] = self.counter.get(i, 0) + 1
        if i not in self.anchor:
            self.anchor[i] = pos[i].copy()
        if self.counter[i] >= 30:
            if np.linalg.norm(pos[i] - self.anchor[i]) < 0.3 and a != 0:
                side = (a - 1 + int(self.mover.rng.choice((2, -2)))) % 8 + 1
                self.escape[i] = (8, side)
                self.escape_count += 1
            self.anchor[i] = pos[i].copy()
            self.counter[i] = 0
        return a


# --------------------------------------------------------------------------- features
def path_dist(planner, frm, to):
    """Route length (m) from frm to to; points inside an obstacle's safety margin snap to the nearest free cell."""
    c = planner.cell(frm)
    if not planner.free[c]:
        c = planner.nearest_free(c)
    return float(planner.field(to)[c])


def actor_features(env, ctrl, i, mask_infeasible=True):
    """What robot i knows when deciding: per-victim features, global features and the action mask.

    mask_infeasible: when at least one victim can still be reached in time, victims whose deadline can no
    longer be met are not offered (fix 1). The learner model and the rule baselines pass False."""
    plan = ctrl.planner
    pos = env.robot_positions()
    V = np.zeros((N_VICTIMS, ACTOR_VICTIM_DIM), np.float32)
    mask = np.zeros(N_ACTIONS, bool)
    mask[WAIT] = True
    # teammates whose plans robot i can see: the other blue robot, or both blue robots for the learner (i = 0)
    mates = [k for k in RL_ROBOTS if k != i]
    for j in range(N_VICTIMS):
        if env.status[j] != WAITING:
            continue
        sev = env.severity[j]
        d_rv = path_dist(plan, pos[i], env.victim_pos[j])
        if not np.isfinite(d_rv):
            continue
        mask[j] = True
        d_vt = ctrl.dist_to_triage(env.victim_pos[j])
        tl = env.time_left[j] * DT
        if sev == 1:
            slack = 1.0
        else:
            slack = (tl - (d_rv / SPEED + 1.0)) / TIME_SCALE   # deadline = being picked up
        others = [path_dist(plan, pos[k], env.victim_pos[j]) for k in range(N_ROBOTS)
                  if k != i and env.carrying[k] < 0]
        nearest_other = min(others) if others else DIST_SCALE * 2
        V[j] = [1.0, sev == 3, sev == 2, sev == 1, min(tl / TIME_SCALE, 1.0), d_rv / DIST_SCALE,
                d_vt / DIST_SCALE, np.clip(slack, -1, 1), float(any(ctrl.target[k] == j for k in mates)),
                min(nearest_other, 2 * DIST_SCALE) / DIST_SCALE, float(ctrl.target[i] == j),
                float(env.t - ctrl.reported_at.get(j, -10 ** 6) < NEW_STEPS),
                float(i != 0 and ctrl.human_intent == j)]          # fix 3: the human seems to be heading here
    V = np.nan_to_num(V, posinf=2.0, neginf=-2.0)
    if mask_infeasible:
        feasible = mask[:WAIT] & (V[:, 7] >= 0)
        if feasible.any():
            mask[:WAIT] = feasible
    mask[WAIT] = not mask[:WAIT].any()
    G = np.array([1 - env.t / MAX_STEPS, np.sum(env.status == UNREPORTED) / 4, np.sum(env.status == WAITING) / 10,
                  ctrl.dist_to_triage(pos[i]) / DIST_SCALE, float(env.carrying[0] >= 0),
                  float(any(env.carrying[k] >= 0 for k in mates)),
                  float(any(ctrl.target[k] is not None for k in mates))], np.float32)
    return V, G, mask


def critic_features(env, ctrl, i, teammate):
    """Centralised (privileged) view for the critic: every robot's distances and both RL robots' targets."""
    plan = ctrl.planner
    pos = env.robot_positions()
    V = np.zeros((N_VICTIMS, CRITIC_VICTIM_DIM), np.float32)
    for j in range(N_VICTIMS):
        if env.status[j] != WAITING:
            continue
        sev = env.severity[j]
        d = [min(path_dist(plan, pos[k], env.victim_pos[j]), 2 * DIST_SCALE) / DIST_SCALE for k in range(N_ROBOTS)]
        V[j] = [1.0, sev == 3, sev == 2, sev == 1, min(env.time_left[j] * DT / TIME_SCALE, 1.0), *d,
                ctrl.dist_to_triage(env.victim_pos[j]) / DIST_SCALE, float(ctrl.target[1] == j),
                float(ctrl.target[2] == j), float(env.t - ctrl.reported_at.get(j, -10 ** 6) < NEW_STEPS),
                float(i == 1), float(ctrl.human_intent == j)]
    V = np.nan_to_num(V, posinf=2.0, neginf=-2.0)
    tm = [float(teammate == t) for t in TEAMMATE_TYPES]
    G = np.array([1 - env.t / MAX_STEPS, np.sum(env.status == UNREPORTED) / 4,
                  *[float(env.carrying[k] >= 0) for k in range(N_ROBOTS)], float(i == 1), float(i == 2), *tm],
                 np.float32)
    return V, G


# --------------------------------------------------------------------------- rule-based choosers
def greedy_choice(env, ctrl, i, feasible_only=False):
    """severity / (path + 2), skipping the other RL robot's target. With feasible_only, also skip
    victims who cannot reach triage before their deadline (a stronger, deadline-aware rule)."""
    V, G, mask = actor_features(env, ctrl, i, mask_infeasible=False)
    best, best_v = WAIT, 0.0
    for j in range(N_VICTIMS):
        if not mask[j] or V[j, 8] > 0:
            continue
        if feasible_only and V[j, 7] < 0:
            continue
        sev = env.severity[j]
        v = sev / (V[j, 5] * DIST_SCALE + 2.0)
        if ctrl.target[i] == j:
            v *= 1.15
        if v > best_v:
            best, best_v = j, v
    return best


def random_choice(env, ctrl, i, rng):
    _, _, mask = actor_features(env, ctrl, i, mask_infeasible=False)
    valid = np.flatnonzero(mask[:WAIT])
    return int(rng.choice(valid)) if len(valid) else WAIT


# --------------------------------------------------------------------------- simulated teammates (robot_0)
class Teammate:
    """Stand-in for the human learner during training and evaluation."""

    def __init__(self, kind, rng, noise=None):
        self.kind, self.rng = kind, rng
        self.noise = noise if noise is not None else (0.35 if kind == "noisy" else 0.0)
        self.wander_dir, self.next_turn = 0, 0

    def choose(self, env, ctrl):
        """Target for robot_0 (doesn't know the RL robots' plans)."""
        plan = ctrl.planner
        pos = env.robot_positions()[0]
        valid = [j for j in range(N_VICTIMS) if env.status[j] == WAITING]
        if not valid:
            return WAIT
        if self.noise > 0 and self.rng.random() < self.noise:
            return int(self.rng.choice(valid))
        best, best_v = WAIT, 0.0
        for j in valid:
            d = path_dist(plan, pos, env.victim_pos[j])
            if np.isfinite(d):
                v = env.severity[j] / (d + 2.0) * (1.15 if ctrl.target[0] == j else 1.0)
                if v > best_v:
                    best, best_v = j, v
        return best

    def raw_action(self, env):
        """For teammates that don't use the planner at all."""
        if self.kind == "idle":
            return 0
        if env.t >= self.next_turn:
            self.wander_dir = int(self.rng.integers(1, len(DIRS)))
            self.next_turn = env.t + int(self.rng.integers(20, 60))
        return self.wander_dir


# --------------------------------------------------------------------------- the controller
class HierarchicalController:
    """Turns high-level choices into low-level actions for all three robots.

    chooser(i) -> action index, called only at decision points of robots in RL_ROBOTS.
    robot_0 is driven by `teammate` (a Teammate) or, if teammate is None, by the caller (human input).
    """

    def __init__(self, env, planner=None):
        self.env = env
        self.planner = planner or GridPlanner(env)
        self.nav = Navigator(env, self.planner)
        self._triage = self.planner.field(SAFE_CENTER)
        self.reset()

    def dist_to_triage(self, p):
        c = self.planner.cell(p)
        if not self.planner.free[c]:
            c = self.planner.nearest_free(c)
        return float(min(self._triage[c], 2 * DIST_SCALE))

    def reset(self):
        self.nav.reset()
        seed = getattr(self.env, "episode_seed", None)       # reproducible unjam moves per scenario
        self.nav.mover.rng = np.random.default_rng(None if seed is None else seed + 7919)
        self.target = [None] * N_ROBOTS
        self.wait_until = [0] * N_ROBOTS
        self.replan = [True] * N_ROBOTS
        self.reported_at = {}
        self.decisions = [0] * N_ROBOTS
        # coordination metrics
        self.duplicate_choices = [0] * N_ROBOTS     # chose the victim the other blue robot was already heading for
        self.wasted_trips = [0] * N_ROBOTS          # target was picked up by someone else or lost before arrival
        self.idle_steps = [0] * N_ROBOTS            # free, with no target (nothing reachable to do)
        self.dup_steps = 0                          # steps where both blue robots chase the same victim
        self.human_intent = None                    # victim the human robot appears to be heading for
        self._intent_hist = []                      # recent route distances from the human robot to each victim

    def notify(self, events):
        for kind, robot, victim in events:
            if kind == "report":
                self.reported_at[victim] = self.env.t
                self.replan = [True] * N_ROBOTS

    def update_human_intent(self):
        """Infer which victim the human (robot_0) is heading for, from movement alone (its plan is not
        visible): the waiting victim whose route distance shrank most over the last 1.5 s, if by > 1 m.
        Uses the same idea as the learner model's commit detection. Measured with a simulated human:
        a guess exists 86% of the time and is right 74% of the time (victims in the same direction
        cannot be told apart from movement alone)."""
        env = self.env
        if env.t % 5:
            return
        if env.carrying[0] >= 0:
            self.human_intent, self._intent_hist = None, []
            return
        p0 = env.robot_positions()[0]
        now = {j: path_dist(self.planner, p0, env.victim_pos[j]) for j in range(N_VICTIMS) if env.status[j] == WAITING}
        self._intent_hist.append(now)
        self._intent_hist = self._intent_hist[-(INTENT_WINDOW // 5 + 1):]
        old = self._intent_hist[0]
        drops = {j: old[j] - d for j, d in now.items() if j in old and np.isfinite(d) and np.isfinite(old[j])}
        if not drops or max(drops.values()) <= INTENT_MIN_DROP:
            self.human_intent = None
            return
        # victims in the same direction shrink together: among those within 0.5 m of the best, assume the nearest
        top = max(drops.values())
        self.human_intent = min((j for j, v in drops.items() if v >= top - 0.5), key=lambda j: now[j])

    def needs_decision(self, i):
        env = self.env
        if env.carrying[i] >= 0:
            return False
        if self.replan[i]:
            return True
        tgt = self.target[i]
        if tgt is None:
            return env.t >= self.wait_until[i]
        if env.status[tgt] != WAITING:
            self.wasted_trips[i] += 1
            return True
        return False

    def set_choice(self, i, a):
        self.replan[i] = False
        self.decisions[i] += 1
        if a == WAIT:
            self.target[i] = None
            self.wait_until[i] = self.env.t + WAIT_STEPS
        else:
            if i in RL_ROBOTS and any(self.target[k] == a for k in RL_ROBOTS if k != i):
                self.duplicate_choices[i] += 1
            self.target[i] = int(a)

    def goal(self, i):
        env = self.env
        if env.carrying[i] >= 0:
            self.target[i] = None
            return SAFE_CENTER
        tgt = self.target[i]
        if tgt is None or env.status[tgt] != WAITING:
            return None
        return env.victim_pos[tgt]

    def actions(self, chooser, teammate=None, learner_action=None):
        """Low-level actions for this step. Calls chooser(i) for RL robots that must decide."""
        env = self.env
        pos = env.robot_positions()
        self.update_human_intent()
        out = {}
        for i in RL_ROBOTS:
            if self.needs_decision(i):
                self.set_choice(i, chooser(i))
            g = self.goal(i)
            if g is None and env.carrying[i] < 0:
                self.idle_steps[i] += 1
            out[f"robot_{i}"] = self.nav.move(i, g, pos)
        if self.target[1] is not None and self.target[1] == self.target[2]:
            self.dup_steps += 1
        if teammate is None:
            out["robot_0"] = learner_action if learner_action is not None else 0
        elif teammate.kind in ("idle", "wander"):
            out["robot_0"] = teammate.raw_action(env)
        else:
            if self.needs_decision(0):
                self.set_choice(0, teammate.choose(env, self))
            out["robot_0"] = self.nav.move(0, self.goal(0), pos)
        return out


# --------------------------------------------------------------------------- networks
def build_networks():
    import torch
    import torch.nn as nn

    class DeepSet(nn.Module):
        """Shared per-victim encoder + mean-pooled context (same idea as the mini project's VictimScorer)."""

        def __init__(self, vdim, gdim, h=64):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(vdim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
            self.ctx = nn.Sequential(nn.Linear(h + gdim, h), nn.ReLU())

        def forward(self, V, G):
            valid = V[..., :1]                                   # first feature = waiting flag
            e = self.enc(V) * valid
            pooled = e.sum(-2) / valid.sum(-2).clamp(min=1.0)
            return e, self.ctx(torch.cat([pooled, G], -1))

    class Actor(nn.Module):
        def __init__(self, h=64):
            super().__init__()
            self.body = DeepSet(ACTOR_VICTIM_DIM, ACTOR_GLOBAL_DIM, h)
            self.score = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
            self.wait = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

        def forward(self, V, G, mask):
            e, c = self.body(V, G)
            cc = c.unsqueeze(-2).expand(*e.shape[:-1], c.shape[-1])
            logits = torch.cat([self.score(torch.cat([e, cc], -1)).squeeze(-1), self.wait(c)], -1)
            return logits.masked_fill(~mask, -1e9)

    class Critic(nn.Module):
        def __init__(self, h=64):
            super().__init__()
            self.body = DeepSet(CRITIC_VICTIM_DIM, CRITIC_GLOBAL_DIM, h)
            self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

        def forward(self, V, G):
            return self.head(self.body(V, G)[1]).squeeze(-1)

    return Actor, Critic


class MappoChooser:
    """Uses a trained actor to choose targets (greedy argmax by default)."""

    def __init__(self, env, ctrl, path, deterministic=True):
        import torch
        Actor, _ = build_networks()
        self.torch = torch
        self.actor = Actor()
        ckpt = torch.load(path, map_location="cpu")
        if ckpt.get("version", 1) != MODEL_VERSION:
            raise ValueError(f"{path} was trained with model version {ckpt.get('version', 1)}, but this code is "
                             f"version {MODEL_VERSION}. Retrain, or use the matching version of the code.")
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        self.env, self.ctrl, self.det = env, ctrl, deterministic
        self.last_probs = {}

    def probs(self, i):
        """The trained actor's probability for each action from robot i's point of view (works for i = 0 too)."""
        torch = self.torch
        V, G, mask = actor_features(self.env, self.ctrl, i)
        with torch.no_grad():
            logits = self.actor(torch.from_numpy(V), torch.from_numpy(G), torch.from_numpy(mask))
        return torch.softmax(logits, -1).numpy()

    def __call__(self, i):
        torch = self.torch
        V, G, mask = actor_features(self.env, self.ctrl, i)
        with torch.no_grad():
            logits = self.actor(torch.from_numpy(V), torch.from_numpy(G), torch.from_numpy(mask))
            probs = torch.softmax(logits, -1).numpy()
        self.last_probs[i] = probs
        if self.det:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))
