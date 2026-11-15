"""The trained pedagogical agent running live in the game.

It sees exactly what it saw in training (learner/tutor_env.py), built from the real learner model:
  - a decision moment starts when the learner's robot is free and has a real choice (start of an episode,
    after a delivery, or when a new victim is reported): the tutor may HINT or ASK before the learner picks;
  - when the learner model has judged the learner's pickup, the tutor may EXPLAIN.
on_step(events) -> (speaker, kind, text) or None. Call it once per env step, after LearnerModel.observe().
"""
import os

import numpy as np

from learner.learner_model import judge_choice, situation
from learner.tutor_env import ASK0, CONCEPTS, EXPLAIN0, HINT0, N_ACTIONS, QUIET, SESSION, tested
from learner.tutor_messages import message

TUTOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "tutor.pt")


class LiveTutor:
    def __init__(self, env, ctrl, model, expert, path=TUTOR, learner=0):
        import torch
        from learner.train_tutor import load
        self.torch, self.net = torch, load(path)
        self.env, self.ctrl, self.model, self.expert, self.me = env, ctrl, model, expert, learner
        self.reset()

    def reset(self):
        """New learner (same as LearnerModel.reset)."""
        self.k, self.since_help, self.recent, self.last_quality = 0, 10, [], 0.5
        self.new_episode()

    def new_episode(self):
        self.prompted, self.helped_before, self.was_free = False, False, False
        self.n_decisions = len(self.model.decisions)

    # ------------------------------------------------------------------ observation, as in TutorEnv.obs()
    def _obs(self, phase, now, result):
        m = self.model
        mastery = np.array([m.mastery[c] for c in CONCEPTS])
        evidence = np.array([len(m.evidence[c]) for c in CONCEPTS], float)
        last = np.array([(1 if m.evidence[c][-1] else -1) if m.evidence[c] else 0 for c in CONCEPTS], float)
        nowv = np.zeros(len(CONCEPTS))
        nowv[now] = 1
        res = np.array([(1 if result[c] else -1) if c in result else 0 for c in CONCEPTS], float)
        return np.concatenate([mastery, np.minimum(evidence, 10) / 10, last, nowv, res,
                               [phase, min(self.since_help, 10) / 10, sum(self.recent[-5:]) / 5,
                                min(self.k, SESSION) / SESSION, self.last_quality]]).astype(np.float32)

    def _act(self, obs, mask):
        t = self.torch
        with t.no_grad():
            lg = self.net.logits(t.from_numpy(obs)[None], t.from_numpy(mask)[None])
        return int(lg.argmax())

    # ------------------------------------------------------------------ live loop
    def on_step(self, events):
        env, m = self.env, self.model
        free = env.carrying[self.me] < 0
        reported = any(kind == "report" for kind, _, _ in events)
        if free and (not self.was_free or reported):
            self.prompted = False                      # a new decision moment
        self.was_free = free
        out = None

        # after the learner's choice has been judged: explain or stay quiet
        if len(m.decisions) > self.n_decisions:
            self.n_decisions = len(m.decisions)
            d = m.decisions[-1]
            if d["quality"] is not None:
                self.last_quality = d["quality"]
            mask = np.zeros(N_ACTIONS, bool)
            mask[QUIET] = True
            for c in d["results"]:
                mask[EXPLAIN0 + CONCEPTS.index(c)] = True
            a = self._act(self._obs(1, [], d["results"]), mask)
            helped = self.helped_before or a != QUIET
            self.k += 1
            self.recent.append(int(helped))
            self.since_help = 0 if helped else self.since_help + 1
            self.helped_before = False
            if a != QUIET:
                speaker, text = message(a, None, d["results"], d["notes"])
                out = (speaker, "explain", text)

        # before the learner chooses: hint, ask or stay quiet (once per decision moment)
        elif free and not self.prompted:
            s = situation(env, self.ctrl, self.me, self.expert, m.last_report_t)
            if len(s["options"]) >= 2:
                self.prompted = True
                judged = {j: judge_choice(s["options"], j, s["probs"], s["since_report"]) for j in s["options"]}
                s["results"] = {j: r for j, (r, _, _) in judged.items()}
                now = [CONCEPTS.index(c) for c in tested(s)]
                mask = np.zeros(N_ACTIONS, bool)
                mask[QUIET] = True
                for ci in now:
                    mask[HINT0 + ci] = mask[ASK0 + ci] = True
                a = self._act(self._obs(0, now, {}), mask)
                self.helped_before = a != QUIET
                if a != QUIET:
                    speaker, text = message(a, s)
                    out = (speaker, "ask" if a >= ASK0 else "hint", text)
        return out
