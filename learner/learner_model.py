"""Learner model: estimates what the person playing robot_0 understands, from
what they do inside the Disaster Response scene (no quizzes).

How it works
1. While the learner's robot is free, the situation is recorded once per
   second: which victims are waiting, their severity, deadline and route
   distance, which victims the blue robots are heading to, and what the
   trained MAPPO agent would choose from the learner's position.
2. When the learner picks someone up, the model finds the moment they
   committed to that victim (when their route distance to it started
   shrinking steadily) and judges the choice against the situation then.
3. Each choice gives evidence about the RL concepts that situation tests:
       priority      reward design: severity matters more than convenience
       deadline      acting under time limits (episode termination, feasibility)
       coordination  multi-agent credit assignment: don't duplicate a teammate
       trade_off     discounting: a far high-value goal vs a near low-value one, judged by
                     discounted value = severity x 0.95 ^ (seconds to reach)
       adaptation    non-stationary environments: re-planning when new information arrives
   plus an overall decision-quality score: how much probability the trained
   agent gives the learner's choice, relative to its own best choice.
4. Mastery of each concept is tracked with Bayesian Knowledge Tracing (BKT).
"""
import numpy as np

from agents.hrl import WAIT, actor_features, path_dist
from envs.disaster_env import DT, N_VICTIMS, SEVERITY_NAMES, SPEED, WAITING

CONCEPTS = {
    "priority": "Prioritising by severity (reward design)",
    "deadline": "Acting before deadlines (time limits)",
    "coordination": "Not duplicating a teammate (coordination)",
    "trade_off": "Near vs far: long-term value (discounting)",
    "adaptation": "Re-planning when new victims appear",
}
BKT = dict(p_init=0.2, p_learn=0.12, p_slip=0.1, p_guess=0.25)
SNAPSHOT_EVERY = 20            # steps (1 s)
SNAPSHOT_KEEP = 90             # 90 s of history
COMPARABLE_EXTRA_M = 8.0       # a higher-severity victim within this extra route distance was a fair alternative
NEW_REPORT_S = 8.0
GAMMA_PER_S = 0.95             # discount used to explain near-vs-far choices: value = severity * 0.95 ** seconds


class LearnerModel:
    def __init__(self, env, ctrl, expert=None, learner=0):
        self.env, self.ctrl, self.expert, self.me = env, ctrl, expert, learner
        self.reset()

    def new_episode(self):
        """Start of a new scenario: forget the old situation history, keep what the learner has shown so far."""
        self.snapshots = []
        self.last_report_t = -10 ** 6

    def reset(self):
        """A new learner: everything starts again."""
        self.mastery = {c: BKT["p_init"] for c in CONCEPTS}
        self.evidence = {c: [] for c in CONCEPTS}       # list of True/False per concept
        self.quality = []                                 # 0..1 per assessed decision
        self.decisions = []                               # full log (dicts)
        self.snapshots = []
        self.last_report_t = -10 ** 6

    # ------------------------------------------------------------------ observation
    def observe(self, events):
        """Call once per env step, after env.step(), with that step's events."""
        env = self.env
        for kind, robot, victim in events:
            if kind == "report":
                self.last_report_t = env.t
                self._snapshot(reason="report")
            elif kind == "pickup" and robot == self.me:
                self._judge(victim)
                self.snapshots = []
            elif kind == "rescue" and robot == self.me:
                self.snapshots = []
        if env.carrying[self.me] < 0 and env.t % SNAPSHOT_EVERY == 0:
            self._snapshot(reason="tick")

    def _snapshot(self, reason):
        env, ctrl = self.env, self.ctrl
        if env.carrying[self.me] >= 0:
            return
        pos = env.robot_positions()[self.me]
        V, _, mask = actor_features(env, ctrl, self.me, mask_infeasible=False)
        options = {}
        for j in range(N_VICTIMS):
            if env.status[j] != WAITING or not mask[j]:
                continue
            claimed_by = [k for k in (1, 2) if ctrl.target[k] == j]
            mate_d = min([path_dist(ctrl.planner, env.robot_positions()[k], env.victim_pos[j]) for k in claimed_by],
                         default=np.inf)
            options[j] = dict(sev=int(env.severity[j]), dist=float(V[j, 5] * 40.0),
                              time_left=float(env.time_left[j] * DT) if env.severity[j] > 1 else None,
                              slack=float(V[j, 7]), claimed=bool(claimed_by), mate_dist=float(mate_d),
                              new=bool(V[j, 11] > 0))
        probs = self.expert.probs(self.me) if self.expert is not None else None
        self.snapshots.append(dict(t=env.t, reason=reason, pos=pos.copy(), options=options, probs=probs,
                                   since_report=(env.t - self.last_report_t) * DT))
        self.snapshots = self.snapshots[-int(SNAPSHOT_KEEP / (SNAPSHOT_EVERY * DT)):]

    # ------------------------------------------------------------------ assessment
    def _commit_snapshot(self, j):
        """The situation when the learner started heading specifically for victim j.

        Between consecutive snapshots, the learner is heading for j if j's route distance
        shrank (by > 0.3 m) and by about as much as the fastest-shrinking option (others in the
        same direction shrink too). Walk back through that run; the snapshot just before it is
        when the choice was made. No such run means j was picked up in passing: not a decision."""
        snaps = [s for s in self.snapshots if j in s["options"]]
        k = len(snaps) - 1
        while k > 0:
            prev, cur = snaps[k - 1]["options"], snaps[k]["options"]
            drops = [prev[o]["dist"] - cur[o]["dist"] for o in cur if o in prev]
            drop_j = prev[j]["dist"] - cur[j]["dist"]
            if drop_j > 0.3 and drop_j >= max(drops) - 0.4:
                k -= 1
            else:
                break
        if k == len(snaps) - 1:
            return None          # never steadily approached j: picked up in passing
        return snaps[k]

    def _judge(self, j):
        snap = self._commit_snapshot(j)
        if snap is None or len(snap["options"]) < 2:
            return                                         # no real choice was made
        opts, me = snap["options"], snap["options"][j]
        others = {k: o for k, o in opts.items() if k != j}
        results, notes = {}, []

        # priority: was a clearly more severe victim available for similar effort?
        fair = {k: o for k, o in others.items() if o["dist"] <= me["dist"] + COMPARABLE_EXTRA_M and o["slack"] >= 0}
        if any(o["sev"] != me["sev"] for o in fair.values()):
            better = [k for k, o in fair.items() if o["sev"] > me["sev"] and not o["claimed"]]
            results["priority"] = not better
            if better:
                b = opts[better[0]]
                notes.append(f"a {SEVERITY_NAMES[b['sev']]} victim {b['dist']:.0f} m away was about the same effort "
                             f"as the {SEVERITY_NAMES[me['sev']]} one you chose ({me['dist']:.0f} m), and nobody "
                             f"else was going for them")

        # deadline: did they go for someone they could not reach in time, while others were reachable?
        if me["slack"] < 0 or any(o["slack"] < 0 for o in others.values()):
            feasible_alt = [k for k, o in others.items() if o["slack"] >= 0 and o["sev"] >= 2]
            ok = me["slack"] >= 0 or not feasible_alt
            results["deadline"] = ok
            if not ok:
                notes.append(f"that victim had {me['time_left']:.0f} s left but was {me['dist']:.0f} m away - "
                             f"too far to reach in time")

        # coordination: was a blue robot already heading there, and closer?
        if any(o["claimed"] for o in opts.values()) and any(not o["claimed"] for o in others.values()):
            dup = me["claimed"] and me["mate_dist"] < me["dist"]
            results["coordination"] = not dup
            if dup:
                notes.append(f"a blue robot was already on its way there and was closer ({me['mate_dist']:.0f} m)")

        # trade-off (discounting): near low-severity vs far high-severity, by discounted value
        # (victims a blue robot is already heading for don't count: they are being taken care of)
        free_opts = {k: o for k, o in opts.items() if k == j or not o["claimed"]}
        nearest = min(free_opts, key=lambda k: free_opts[k]["dist"])
        far_better = [k for k, o in free_opts.items()
                      if o["sev"] > free_opts[nearest]["sev"] and o["dist"] > free_opts[nearest]["dist"] + 8]
        if far_better:
            value = {k: o["sev"] * GAMMA_PER_S ** (o["dist"] / SPEED) for k, o in free_opts.items() if o["slack"] >= 0}
            if value:
                best_v = max(value, key=value.get)
                ok = value.get(j, 0.0) >= 0.85 * value[best_v]
                results["trade_off"] = ok
                if not ok:
                    b = opts[best_v]
                    notes.append(f"counting travel time, the {SEVERITY_NAMES[b['sev']]} victim {b['dist']:.0f} m away was "
                                 f"worth more ({b['sev']} x 0.95^{b['dist'] / SPEED:.0f}s = {value[best_v]:.2f}) than "
                                 f"yours ({me['sev']} x 0.95^{me['dist'] / SPEED:.0f}s = {value.get(j, 0.0):.2f})")

        # adaptation and overall quality are judged against the trained agent's choice
        if snap["probs"] is not None:
            p = snap["probs"]
            best = int(np.argmax(p[:WAIT]))
            q = float(p[j] / max(p[best], 1e-9))
            self.quality.append(min(q, 1.0))
            if snap["since_report"] <= NEW_REPORT_S and any(o["new"] and o["sev"] >= 2 for o in opts.values()):
                results["adaptation"] = q >= 0.5

        for c, ok in results.items():
            self._bkt(c, ok)
        self.decisions.append(dict(t=self.env.t, victim=j, commit_t=snap["t"], results=results, notes=notes,
                                   quality=self.quality[-1] if snap["probs"] is not None else None))

    def _bkt(self, concept, correct):
        pL = self.mastery[concept]
        s, g = BKT["p_slip"], BKT["p_guess"]
        if correct:
            post = pL * (1 - s) / (pL * (1 - s) + (1 - pL) * g)
        else:
            post = pL * s / (pL * s + (1 - pL) * (1 - g))
        self.mastery[concept] = post + (1 - post) * BKT["p_learn"]
        self.evidence[concept].append(bool(correct))

    # ------------------------------------------------------------------ summaries
    def summary(self):
        return dict(mastery=dict(self.mastery), quality=float(np.mean(self.quality)) if self.quality else None,
                    decisions=len(self.decisions),
                    evidence={c: (sum(v), len(v)) for c, v in self.evidence.items()})

    def last_feedback(self):
        if not self.decisions:
            return None
        d = self.decisions[-1]
        if d["notes"]:
            return "Could be better: " + "; ".join(d["notes"]) + "."
        if d["results"]:
            return "Good call: " + ", ".join(CONCEPTS[c].split(" (")[0].lower() for c, ok in d["results"].items() if ok) + "."
        return None
