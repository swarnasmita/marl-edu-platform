# Pedagogical agent: when should the robots help?

Simulated students (hidden knowledge per concept, attention) work through real decision situations from the game; 30 decisions per session (about 5 game episodes). The tutor sees only the learner model's estimates. Student model: v2 (with motivation and forgetting). Test: held-out game situations, 150 students per profile, the same students for every tutor. RL tutor: mean over 3 training seeds.

Learning gain = increase in the student's true mean knowledge over the session (0-1 scale).

## All students

| Tutor | Learning gain | Final knowledge | Interventions per session | Interruption (s) | Attention at end | Motivation at end | Correct choices (last 10) | Reward |
|---|---|---|---|---|---|---|---|---|
| **RL tutor (PPO)** | 0.201 | 0.658 | 6.8 | 25 | 0.95 | 0.74 | 78% | 15.3 |
| No help | 0.109 | 0.566 | 0.0 | 0 | 1.00 | 0.77 | 72% | 10.9 |
| Always help | 0.148 | 0.605 | 37.6 | 75 | 0.00 | 0.95 | 76% | -9.1 |
| Explain mistakes | 0.204 | 0.661 | 7.7 | 31 | 0.89 | 0.70 | 79% | 14.2 |
| Explain mistakes + praise | 0.152 | 0.609 | 29.4 | 118 | 0.00 | 0.93 | 75% | -15.7 |
| Mastery-threshold rule | 0.178 | 0.635 | 8.4 | 21 | 0.92 | 0.86 | 77% | 11.7 |
| Random | 0.145 | 0.602 | 43.5 | 145 | 0.00 | 0.98 | 75% | -20.8 |

## Learning gain by student profile

| Tutor | novice | intermediate | expert | mixed |
|---|---|---|---|---|
| RL tutor (PPO) | 0.310 | 0.206 | 0.078 | 0.208 |
| No help | 0.146 | 0.121 | 0.058 | 0.110 |
| Always help | 0.233 | 0.162 | 0.054 | 0.142 |
| Explain mistakes | 0.314 | 0.210 | 0.080 | 0.213 |
| Explain mistakes + praise | 0.227 | 0.159 | 0.068 | 0.154 |
| Mastery-threshold rule | 0.282 | 0.181 | 0.068 | 0.181 |
| Random | 0.216 | 0.160 | 0.064 | 0.141 |

## Interventions per session by student profile

| Tutor | novice | intermediate | expert | mixed |
|---|---|---|---|---|
| RL tutor (PPO) | 9.3 | 7.1 | 4.2 | 6.7 |
| No help | 0.0 | 0.0 | 0.0 | 0.0 |
| Always help | 41.0 | 37.6 | 34.2 | 37.5 |
| Explain mistakes | 11.3 | 8.0 | 4.1 | 7.4 |
| Explain mistakes + praise | 29.5 | 29.4 | 29.4 | 29.4 |
| Mastery-threshold rule | 11.8 | 8.1 | 5.0 | 8.5 |
| Random | 43.2 | 43.3 | 43.6 | 43.8 |

## RL tutor vs the strongest rule (Explain mistakes), paired over the same students

| Measure | RL tutor | Explain mistakes | Difference | p |
|---|---|---|---|---|
| Learning gain | 0.201 | 0.204 | -0.004 | 0.0000 |
| Interventions | 6.847 | 7.690 | -0.843 | 0.0000 |
| Attention at end | 0.953 | 0.888 | +0.064 | 0.0000 |
| Motivation at end | 0.744 | 0.704 | +0.040 | 0.0000 |
| Reward | 15.309 | 14.249 | +1.060 | 0.0000 |

## What the RL tutor says (final model; per session of 30 decisions)

| Profile | Hints | Questions | Explanations of mistakes | Encouragement (explaining a good choice) | Total |
|---|---|---|---|---|---|
| novice | 0.1 | 0.0 | 8.9 | 0.0 | 9.1 |
| intermediate | 0.2 | 0.0 | 6.9 | 0.0 | 7.1 |
| expert | 0.2 | 0.0 | 3.6 | 0.0 | 3.8 |
| mixed | 0.2 | 0.0 | 6.2 | 0.0 | 6.3 |

## What the RL tutor learned differently from 'Explain mistakes'

Both explain mistakes; the RL tutor leaves some unexplained. Share of mistakes it does NOT explain:

| Learner-model mastery estimate for that concept | Mistakes | Not explained |
|---|---|---|
| 0.0-0.3 | 640 | 12% |
| 0.3-0.6 | 260 | 14% |
| 0.6-0.8 | 189 | 18% |
| 0.8-1.0 | 1217 | 24% |

| Times the robots helped in the last 5 decisions | Mistakes | Not explained |
|---|---|---|
| 0 | 653 | 3% |
| 1 | 914 | 17% |
| 2 | 578 | 32% |
| 3 | 147 | 46% |
| 4-5 | 14 | 79% |

The RL tutor learned to let likely slips pass (mistakes on concepts the learner model already rates as mastered) and to hold back when it has helped a lot recently - fewer interruptions, same learning.

## Robustness: students whose learning works differently from the training simulator

The RL tutor was trained with one set of learning-effect parameters. Here every learning effect is halved or increased by half, and students are slower / faster and more / less patient than in training; learning gain, RL tutor vs Explain mistakes.

| Test students | RL tutor | Explain mistakes | Difference | p |
|---|---|---|---|---|
| effects x0.5 | 0.157 | 0.159 | -0.001 | 0.0903 |
| effects x1.5 | 0.232 | 0.239 | -0.007 | 0.0000 |
| attention costs x2 | 0.181 | 0.174 | +0.007 | 0.0000 |
| slow, impatient students | 0.103 | 0.100 | +0.003 | 0.0000 |
| fast, patient students | 0.283 | 0.290 | -0.008 | 0.0000 |

Limitations: the students are simulated; their learning rules encode effects reported in the tutoring literature, but the size of each effect is an assumption. A study with real learners is needed to confirm the gains.
