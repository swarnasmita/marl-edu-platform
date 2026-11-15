# Pedagogical agent: deciding when the robots should help the learner

## What it does
At every decision the learner makes in the Disaster Response game, the pedagogical agent decides whether the
blue robots should
- **before the choice:** stay quiet, give a **hint** about one RL concept, or **ask** a reflective question
  (the game pauses);
- **after the choice:** stay quiet, or **explain** the choice in terms of an RL concept.

The five concepts are those of the learner model: priority (reward design), deadlines (time limits),
coordination (multi-agent credit assignment), near vs far (discounting) and re-planning (non-stationary
environments). Messages are filled in from the live situation (distances, time left, severities) and link it
to the RL idea (learner/tutor_messages.py).

## How it was built
- **Observation** (only what the platform can observe): the learner model's BKT mastery estimates, evidence so
  far, the last outcome per concept, which concepts the current situation tests, the outcome of the current
  choice, and how often the robots helped recently.
- **Reward:** 100 x gain in the student's true knowledge - 0.15 x seconds of interruption - 0.5 x (1 - attention).
- **Training:** PPO with action masking (3 seeds, 19,200 simulated sessions each, about 2 min per seed), on
  simulated students working through 1,423 **real decision situations recorded from the game** (80% training,
  20% held out for testing).
- **Simulated students:** hidden knowledge per concept and attention; learning effects from the tutoring
  literature (feedback on errors, expertise reversal for hints, zone of proximal development for questions,
  assistance dilemma via attention); profiles novice / intermediate / expert / mixed, slow / fast, patient /
  impatient.
- **Baselines:** no help, always help, explain every mistake, the standard mastery-threshold rule, random.

## Results
Held-out situations, 600 students, the same students for every tutor (tutor_report.md):

| Tutor | Learning gain | Interventions per session | Attention at end |
|---|---|---|---|
| **RL tutor** | **0.259** | **6.0** | **0.97** |
| Explain mistakes | 0.261 | 6.8 | 0.93 |
| Mastery-threshold rule | 0.221 | 7.7 | 0.94 |
| Always help | 0.180 | 37.2 | 0.00 |
| No help | 0.143 | 0.0 | 1.00 |

Inside the real MuJoCo game, 80 students x 5 episodes per tutor (tutor_in_game.md):

| Tutor | Learning gain | Interventions | Attention at end |
|---|---|---|---|
| **RL tutor** | **0.228** | **6.3** | **0.94** |
| Explain mistakes | 0.228 (difference 0.000, p = 0.97) | 7.4 (p < 0.0001) | 0.86 (p < 0.0001) |
| Mastery-threshold rule | 0.191 (RL +0.037, p < 0.0001) | 8.2 (p < 0.0001) | 0.86 (p = 0.002) |
| No help | 0.110 | 0 | 1.00 |

- The RL tutor **matches the best rule's learning with 12-15% fewer interruptions** and keeps learners more
  attentive, and **clearly beats the standard mastery-threshold rule** (+0.04 learning gain). The same holds
  in the fast simulator and in the real game, and across learning-speed / patience / effect-size variations.
- **What it learned by itself:** to let likely **slips** pass (it leaves 21% of mistakes unexplained when the
  learner model already rates the concept as mastered, vs 5% for weak concepts) and to **hold back after
  helping several times recently** (35% unexplained after 3+ recent interventions, vs 4% after none).
- It almost never hints or asks. In this simulator, explaining right after a mistake gives most of the learning;
  questions add a little (learning gain 0.265 vs 0.260 in a hand-written test) but cost more interruption
  time than they are worth at the chosen cost of 0.15 per second.

## Limitations
The students are simulated: the direction of each learning effect comes from the tutoring literature, but its
size is an assumption, and the balance between learning and interruption is set by the reward weights. A study
with real learners is needed to confirm the gains - including whether questions help real learners more than
the simulator assumes.

## Files
learner/tutor_env.py (students, tutoring environment, rules), learner/train_tutor.py (PPO, evaluation, report),
learner/live_tutor.py (the tutor running in the game), learner/tutor_messages.py (what the robots say),
learner/collect_situations.py, learner/validate_tutor_in_game.py, learner/checkpoints/tutor.pt (final model),
envs/view_disaster.py (speech bubbles; T = tutor on/off, Space = continue after a question).
