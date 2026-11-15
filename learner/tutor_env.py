"""Tutoring environment for the pedagogical agent: simulated students working through real game situations.

The pedagogical agent decides, at every decision the learner makes in the Disaster Response game, whether the
blue robots should help, and how:
    before the learner chooses:  stay quiet | HINT about a concept | ASK a reflective question about a concept
    after the learner chose:     stay quiet | EXPLAIN a concept (why the choice was good / not)
over the five RL concepts of the learner model (priority, deadline, coordination, trade-off, adaptation).

Situations: real decision situations recorded from the game (learner/collect_situations.py), with every possible
choice judged in advance by the learner model's own rules.

Simulated student (hidden from the tutor): knowledge K_c in [0, 1] per concept and attention in [0, 1].
  Choosing: for each concept the situation tests, the student applies it with probability K_c (a hint or a
  question raises that probability for this decision); applying a concept rules out choices that violate it;
  among the remaining choices the student takes the nearest victim (a typical novice heuristic), with 10% noise.
  Learning (effects documented in the intelligent-tutoring literature, encoded as simple rules):
    practice      a correct choice strengthens the concept a little (less if it was scaffolded by a hint)
    explain       large gain when it follows a mistake on that concept (feedback on errors), small otherwise
    hint          moderate gain for low knowledge; none for high knowledge, and it annoys (expertise reversal)
    ask           largest gain when knowledge is medium (zone of proximal development); costs more time
  Attention: every intervention costs attention (more for unneeded ones, more for impatient students); quiet
  decisions restore it; all intervention effects scale with attention (too much help -> ignored help).
  Profiles: prior knowledge (novice / intermediate / expert / mixed), learning speed, patience.

The tutor only observes what the platform can observe: the learner model's BKT mastery estimates, the evidence
so far, which concepts the current situation tests, the judged outcome of the last choice, and its own recent
interventions. The reward uses the hidden knowledge (available in simulation only):
    reward = 100 * (gain in mean knowledge) - 0.15 * seconds of interruption - 0.5 * (1 - attention)
"""
import os
import pickle

import numpy as np

CONCEPTS = ("priority", "deadline", "coordination", "trade_off", "adaptation")
NC = len(CONCEPTS)
QUIET, HINT0, ASK0, EXPLAIN0 = 0, 1, 1 + NC, 1 + 2 * NC
N_ACTIONS = 1 + 3 * NC
ACTION_NAMES = ["quiet"] + [f"hint {c}" for c in CONCEPTS] + [f"ask {c}" for c in CONCEPTS] + \
               [f"explain {c}" for c in CONCEPTS]
SECONDS = {"hint": 1.5, "ask": 4.0, "explain": 4.0}
SESSION = 30                    # learner decisions per session (about 5 game episodes)
OBS_DIM = 5 * NC + 5
BKT = dict(p_init=0.2, p_learn=0.12, p_slip=0.1, p_guess=0.25)       # same as the learner model

# learning-effect parameters of the simulated students (assumptions; see robustness test in train_tutor.py)
SIM = dict(a_practice=0.03, a_explain=0.25, a_hint=0.10, a_ask=0.18, noise=0.10,
           att_cost=dict(hint=0.06, ask=0.10, explain=0.08), att_unneeded=0.08, att_recover=0.04,
           hint_follow=0.9, ask_boost=0.5, expert_k=0.75)
PROFILES = {
    "novice": dict(prior=(0.05, 0.25)),
    "intermediate": dict(prior=(0.3, 0.6)),
    "expert": dict(prior=(0.65, 0.9)),
    "mixed": dict(prior="mixed"),
}

_DATA = None


def situations():
    """Real game situations, split by game episode: 80% for training, 20% held out for testing."""
    global _DATA
    if _DATA is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "situations.pkl")
        data = [s for s in pickle.load(open(path, "rb")) if tested(s)]
        eps = sorted({s["episode"] for s in data})
        test_eps = set(eps[::5])
        _DATA = dict(train=[s for s in data if s["episode"] not in test_eps],
                     test=[s for s in data if s["episode"] in test_eps])
    return _DATA


def tested(s):
    """Concepts this situation really tests: some choice shows it understood and some choice violates it."""
    out = []
    for c in CONCEPTS:
        vals = [r[c] for r in s["results"].values() if c in r]
        if True in vals and False in vals:
            out.append(c)
    return out


class Student:
    def __init__(self, rng, profile, lr=1.0, impatience=1.0, sim=SIM):
        self.rng, self.sim, self.lr, self.impatience = rng, sim, lr, impatience
        pr = PROFILES[profile]["prior"]
        if pr == "mixed":
            self.K = np.array([rng.uniform(0.05, 0.25) if rng.random() < 0.5 else rng.uniform(0.65, 0.9)
                               for _ in CONCEPTS])
        else:
            self.K = rng.uniform(*pr, size=NC)
        self.att = 1.0

    def choose(self, s, boost):
        """boost: {concept index: probability of applying it this time} from a hint / question."""
        opts = list(s["options"])
        cand = opts
        for c in self.rng.permutation(tested(s)):
            ci = CONCEPTS.index(c)
            if self.rng.random() < boost.get(ci, self.K[ci]):
                keep = [j for j in cand if s["results"][j].get(c, True)]
                if keep:
                    cand = keep
        if self.rng.random() < self.sim["noise"]:
            return cand[self.rng.integers(len(cand))]
        return min(cand, key=lambda j: s["options"][j]["dist"])

    def _gain(self, ci, rate):
        self.K[ci] += min(1.0, rate * self.lr) * (1 - self.K[ci])

    def _spend(self, kind, unneeded):
        self.att = max(0.0, self.att - self.impatience * (self.sim["att_cost"][kind] +
                                                         (self.sim["att_unneeded"] if unneeded else 0.0)))


class TutorEnv:
    """Batched-friendly single environment. Steps alternate between the 'before' and 'after' phase of each
    decision. obs() -> observation, mask() -> valid actions, step(a) -> reward, done."""

    def __init__(self, rng, profile=None, split="train", lr=None, impatience=None, sim=SIM):
        self.rng, self.split, self.sim = rng, split, sim
        self.profile = profile or rng.choice(list(PROFILES))
        lr = lr if lr is not None else float(rng.choice([0.6, 1.0, 1.5]))
        imp = impatience if impatience is not None else float(rng.choice([0.6, 1.0, 1.6]))
        self.student = Student(rng, self.profile, lr, imp, sim)
        self.K0 = self.student.K.copy()
        self.pool = situations()[split]
        self.mastery = np.full(NC, BKT["p_init"])
        self.evidence = np.zeros(NC)
        self.last = np.zeros(NC)                      # +1 / -1 outcome of the last judgement per concept
        self.k, self.phase = 0, 0                     # decision index, 0 = before, 1 = after
        self.since_help, self.recent = 10, []
        self.log = dict(interventions=0, seconds=0.0, correct=[], quality=[], actions=[])
        self._next_situation()

    # ------------------------------------------------------------------ observation
    def _next_situation(self):
        self.s = self.pool[self.rng.integers(len(self.pool))]
        self.tested = [CONCEPTS.index(c) for c in tested(self.s)]
        self.boost, self.hinted, self.result = {}, set(), {}

    def obs(self):
        now = np.zeros(NC)
        now[self.tested] = 1
        res = np.zeros(NC)
        for c, ok in self.result.items():
            res[CONCEPTS.index(c)] = 1 if ok else -1
        return np.concatenate([self.mastery, np.minimum(self.evidence, 10) / 10, self.last, now, res,
                               [self.phase, min(self.since_help, 10) / 10, sum(self.recent[-5:]) / 5,
                                self.k / SESSION, self.log["quality"][-1] if self.log["quality"] else 0.5]]
                              ).astype(np.float32)

    def mask(self):
        m = np.zeros(N_ACTIONS, bool)
        m[QUIET] = True
        if self.phase == 0:
            for ci in self.tested:
                m[HINT0 + ci] = m[ASK0 + ci] = True
        else:
            for c in self.result:
                m[EXPLAIN0 + CONCEPTS.index(c)] = True
        return m

    # ------------------------------------------------------------------ dynamics
    def step(self, a):
        st, sim = self.student, self.sim
        before = st.K.mean()
        seconds = 0.0
        helped = a != QUIET
        if self.phase == 0:
            if HINT0 <= a < ASK0:
                ci = a - HINT0
                unneeded = st.K[ci] > sim["expert_k"]
                self.boost[ci] = max(st.K[ci], sim["hint_follow"] * st.att)
                if not unneeded:
                    st._gain(ci, sim["a_hint"] * st.att)
                st._spend("hint", unneeded)
                self.hinted.add(ci)
                seconds = SECONDS["hint"]
            elif ASK0 <= a < EXPLAIN0:
                ci = a - ASK0
                k = st.K[ci]
                zpd = 1.0 if 0.25 < k < 0.75 else 0.4
                self.boost[ci] = max(k, k + sim["ask_boost"] * (1 - k) * st.att)
                st._gain(ci, sim["a_ask"] * st.att * zpd)
                st._spend("ask", k > sim["expert_k"])
                seconds = SECONDS["ask"]
            # the learner now chooses
            j = st.choose(self.s, self.boost)
            self.result = self.s["results"][j]
            self.chosen = j
            q = self.s["quality"][j]
            self.log["quality"].append(q if q is not None else 0.5)
            ok_all = all(self.result.values()) if self.result else True
            self.log["correct"].append(float(ok_all))
            for c, ok in self.result.items():                       # practice + learner model (BKT) update
                ci = CONCEPTS.index(c)
                if ok:
                    st._gain(ci, sim["a_practice"] * (0.3 if ci in self.hinted else 1.0))
                self._bkt(ci, ok)
            self.phase = 1
        else:
            if a >= EXPLAIN0:
                ci = a - EXPLAIN0
                wrong = self.result.get(CONCEPTS[ci]) is False
                st._gain(ci, sim["a_explain"] * st.att * (1.0 if wrong else 0.2))
                st._spend("explain", not wrong)
                seconds = SECONDS["explain"]
            self.phase = 0
            self.k += 1
            helped_decision = helped or self.helped_before
            self.recent.append(int(helped_decision))
            self.since_help = 0 if helped_decision else self.since_help + 1
            if not helped_decision:
                st.att = min(1.0, st.att + sim["att_recover"])
            self._next_situation()
        if self.phase == 1:
            self.helped_before = helped           # whether the 'before' step of this decision helped
        if helped:
            self.log["interventions"] += 1
            self.log["seconds"] += seconds
            self.log["actions"].append(a)
        r = 100 * (st.K.mean() - before) - 0.15 * seconds - 0.5 * (1 - st.att) * (self.phase == 0)
        return float(r), self.k >= SESSION

    def _bkt(self, ci, ok):
        pL, s, g = self.mastery[ci], BKT["p_slip"], BKT["p_guess"]
        post = pL * (1 - s) / (pL * (1 - s) + (1 - pL) * g) if ok else pL * s / (pL * s + (1 - pL) * (1 - g))
        self.mastery[ci] = post + (1 - post) * BKT["p_learn"]
        self.evidence[ci] += 1
        self.last[ci] = 1 if ok else -1

    def summary(self):
        st = self.student
        return dict(profile=self.profile, final_k=float(st.K.mean()), gain=float(st.K.mean() - self.K0.mean()),
                    interventions=self.log["interventions"], seconds=self.log["seconds"],
                    attention=float(st.att), correct_last10=float(np.mean(self.log["correct"][-10:])),
                    quality_last10=float(np.mean(self.log["quality"][-10:])))


# ---------------------------------------------------------------------- rule-based tutors (baselines)
def rule_never(env):
    return QUIET


def rule_always(env):
    """Always help: hint the weakest tested concept before every decision; explain every mistake."""
    if env.phase == 0:
        return HINT0 + min(env.tested, key=lambda ci: env.mastery[ci]) if env.tested else QUIET
    wrong = [CONCEPTS.index(c) for c, ok in env.result.items() if not ok]
    return EXPLAIN0 + wrong[0] if wrong else QUIET


def rule_explain_errors(env):
    """Feedback on errors only: explain every mistake, never hint."""
    if env.phase == 0:
        return QUIET
    wrong = [CONCEPTS.index(c) for c, ok in env.result.items() if not ok]
    return EXPLAIN0 + wrong[0] if wrong else QUIET


def rule_mastery(env):
    """Standard intelligent-tutoring rule using the learner model: hint a tested concept whose estimated mastery
    is below 0.4; explain a mistake on a concept whose estimated mastery is below 0.7; otherwise stay quiet."""
    if env.phase == 0:
        low = [ci for ci in env.tested if env.mastery[ci] < 0.4]
        return HINT0 + min(low, key=lambda ci: env.mastery[ci]) if low else QUIET
    wrong = [CONCEPTS.index(c) for c, ok in env.result.items() if not ok and env.mastery[CONCEPTS.index(c)] < 0.7]
    return EXPLAIN0 + wrong[0] if wrong else QUIET


def rule_random(env):
    m = np.flatnonzero(env.mask())
    return int(env.rng.choice(m))


RULES = {"No help": rule_never, "Always help": rule_always, "Explain mistakes": rule_explain_errors,
         "Mastery-threshold rule": rule_mastery, "Random": rule_random}
