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

## Version 2: students with motivation and forgetting
The first tutor only ever explained mistakes, so the robots spoke only when the learner was wrong. To check
whether that was an artefact of an over-simple student model, a second student model (v2) adds two well-known
effects (learner/tutor_env.py, SIM_V2):
- **motivation:** successes raise it, mistakes and criticism lower it, encouragement after a good choice raises
  it (less if given often); low motivation slows learning and makes choices more careless;
- **forgetting:** knowledge of a concept fades when it is not practised; a hint or question about a concept not
  seen for a while works better (spacing / testing effects).
The tutor also observes how long ago each concept was last tested and the learner's current run of mistakes.
Same reward, same training (3 seeds); a new baseline "explain mistakes + praise every good choice" was added.

Results with v2 students, held-out situations (tutor_report_v2.md):

| Tutor | Learning gain | Interventions | Attention | Motivation | Reward |
|---|---|---|---|---|---|
| **RL tutor v2** | 0.201 | **6.8** | **0.95** | 0.74 | **15.3** |
| Explain mistakes | **0.204** | 7.7 | 0.89 | 0.70 | 14.2 |
| Explain mistakes + praise | 0.152 | 29.4 | 0.00 | **0.93** | -15.7 |
| Mastery-threshold rule | 0.178 | 8.4 | 0.92 | 0.86 | 11.7 |
| No help | 0.109 | 0 | 1.00 | 0.77 | 10.9 |

Inside the real game with v2 students (tutor_in_game_v2.md): RL tutor vs explain mistakes - 25% fewer
interruptions (5.6 vs 7.4), attention +0.12, motivation +0.03, learning gain -0.008 (0.166 vs 0.174; all
p < 0.0001); vs the mastery-threshold rule: +0.014 learning gain and fewer interruptions, but lower
motivation (0.66 vs 0.82: that rule's hints give the learner more scaffolded successes).

**The tutor still almost only explains mistakes.** Hand-written alternatives show why: adding encouragement
after a quiet stretch raises motivation (0.75-0.82) and slightly raises learning, but each encouragement is a
full interruption, and the reward drops (8.5-12.7 vs 15.1); adding spaced questions changes little
(reward 13.9-14.0). With both student models, the best use of an interruption is targeted feedback on a
mistake. **Design consequence:** positive feedback should not interrupt - the platform shows a quiet
"✓ Good call" in the side panel after every good choice, outside the tutor's decisions.

The viewer uses the v2 tutor. For demonstrations, T cycles: RL tutor -> 'always help' rule (a hint before every
decision, so all message types can be shown) -> off.

## Limitations
The students are simulated: the direction of each learning effect comes from the tutoring literature, but its
size is an assumption, and the balance between learning and interruption is set by the reward weights. A study
with real learners is needed to confirm the gains - including whether questions and encouragement help real
learners more than the simulators assume, and whether real learners find error-only feedback discouraging.

## Files
learner/tutor_env.py (students, tutoring environment, rules), learner/train_tutor.py (PPO, evaluation, report),
learner/live_tutor.py (the tutor running in the game), learner/tutor_messages.py (what the robots say),
learner/collect_situations.py, learner/validate_tutor_in_game.py (--student v1 / v2),
learner/checkpoints/tutor.pt (v1 students) and tutor_v2.pt (v2 students; used in the viewer),
envs/view_disaster.py (speech bubbles; T = RL tutor / always help (demo) / off; Space = continue after a question).
Reproduce: python learner/train_tutor.py --student v2 ; python learner/validate_tutor_in_game.py --student v2
