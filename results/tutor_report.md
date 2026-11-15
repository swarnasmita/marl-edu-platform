# Pedagogical agent: when should the robots help?

Simulated students (hidden knowledge per concept, attention) work through real decision situations from the game; 30 decisions per session (about 5 game episodes). The tutor sees only the learner model's estimates. Test: held-out game situations, 150 students per profile, the same students for every tutor. RL tutor: mean over 3 training seeds.

Learning gain = increase in the student's true mean knowledge over the session (0-1 scale).

## All students

| Tutor | Learning gain | Final knowledge | Interventions per session | Interruption (s) | Attention at end | Correct choices (last 10) | Reward |
|---|---|---|---|---|---|---|---|
| **RL tutor (PPO)** | 0.259 | 0.716 | 6.0 | 24 | 0.97 | 83% | 21.4 |
| No help | 0.143 | 0.600 | 0.0 | 0 | 1.00 | 74% | 14.3 |
| Always help | 0.180 | 0.637 | 37.2 | 74 | 0.00 | 78% | -5.7 |
| Explain mistakes | 0.261 | 0.718 | 6.8 | 27 | 0.93 | 83% | 20.8 |
| Mastery-threshold rule | 0.221 | 0.678 | 7.7 | 19 | 0.94 | 80% | 16.6 |
| Random | 0.170 | 0.627 | 43.4 | 144 | 0.00 | 76% | -18.3 |

## Learning gain by student profile

| Tutor | novice | intermediate | expert | mixed |
|---|---|---|---|---|
| RL tutor (PPO) | 0.422 | 0.258 | 0.092 | 0.264 |
| No help | 0.205 | 0.153 | 0.069 | 0.145 |
| Always help | 0.292 | 0.192 | 0.062 | 0.172 |
| Explain mistakes | 0.423 | 0.262 | 0.094 | 0.267 |
| Mastery-threshold rule | 0.359 | 0.220 | 0.079 | 0.225 |
| Random | 0.259 | 0.185 | 0.071 | 0.164 |

## Interventions per session by student profile

| Tutor | novice | intermediate | expert | mixed |
|---|---|---|---|---|
| RL tutor (PPO) | 8.8 | 6.4 | 3.2 | 5.8 |
| No help | 0.0 | 0.0 | 0.0 | 0.0 |
| Always help | 40.2 | 37.4 | 33.9 | 37.1 |
| Explain mistakes | 9.9 | 7.2 | 3.6 | 6.5 |
| Mastery-threshold rule | 11.4 | 7.3 | 4.6 | 7.7 |
| Random | 43.1 | 43.0 | 43.7 | 43.8 |

## RL tutor vs the strongest rule (Explain mistakes), paired over the same students

| Measure | RL tutor | Explain mistakes | Difference | p |
|---|---|---|---|---|
| Learning gain | 0.259 | 0.261 | -0.002 | 0.0001 |
| Interventions | 6.029 | 6.817 | -0.788 | 0.0000 |
| Attention at end | 0.969 | 0.935 | +0.034 | 0.0000 |
| Reward | 21.395 | 20.784 | +0.611 | 0.0000 |

## What the RL tutor does (share of its interventions, final model)

| Profile | Hints | Questions | Explanations | Interventions per session |
|---|---|---|---|---|
| novice | 1% | 0% | 99% | 8.8 |
| intermediate | 1% | 0% | 99% | 6.5 |
| expert | 2% | 0% | 98% | 3.4 |
| mixed | 2% | 0% | 98% | 6.0 |

## What the RL tutor learned differently from 'Explain mistakes'

Both explain mistakes; the RL tutor leaves some unexplained. Share of mistakes it does NOT explain:

| Learner-model mastery estimate for that concept | Mistakes | Not explained |
|---|---|---|
| 0.0-0.3 | 616 | 5% |
| 0.3-0.6 | 271 | 8% |
| 0.6-0.8 | 201 | 13% |
| 0.8-1.0 | 1017 | 21% |

| Times the robots helped in the last 5 decisions | Mistakes | Not explained |
|---|---|---|
| 0 | 667 | 4% |
| 1 | 791 | 12% |
| 2 | 484 | 23% |
| 3 | 137 | 35% |
| 4-5 | 26 | 35% |

The RL tutor learned to let likely slips pass (mistakes on concepts the learner model already rates as mastered) and to hold back when it has helped a lot recently - fewer interruptions, same learning.

## Robustness: students whose learning works differently from the training simulator

The RL tutor was trained with one set of learning-effect parameters. Here every learning effect is halved or increased by half, and students are slower / faster and more / less patient than in training; learning gain, RL tutor vs Explain mistakes.

| Test students | RL tutor | Explain mistakes | Difference | p |
|---|---|---|---|---|
| effects x0.5 | 0.212 | 0.213 | -0.001 | 0.1164 |
| effects x1.5 | 0.297 | 0.297 | -0.000 | 0.8499 |
| attention costs x2 | 0.238 | 0.234 | +0.004 | 0.0005 |
| slow, impatient students | 0.142 | 0.141 | +0.001 | 0.0656 |
| fast, patient students | 0.347 | 0.349 | -0.002 | 0.2364 |

Limitations: the students are simulated; their learning rules encode effects reported in the tutoring literature, but the size of each effect is an assumption. A study with real learners is needed to confirm the gains.
