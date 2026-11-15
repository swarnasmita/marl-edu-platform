# Pedagogical agent inside the real game

Simulated students played the learner's robot in real MuJoCo episodes (5 episodes each, 20 students per profile, the same students for every tutor), with the trained blue robots. The tutor acted at every real decision of the student.

| Tutor | Learning gain | Decisions | Interventions | Interruption (s) | Attention at end | Team score |
|---|---|---|---|---|---|---|
| RL tutor (PPO) | 0.228 | 25 | 6.3 | 25 | 0.94 | 168.3 |
| Explain mistakes | 0.228 | 25 | 7.4 | 29 | 0.86 | 168.6 |
| Mastery-threshold rule | 0.191 | 24 | 8.2 | 21 | 0.86 | 168.7 |
| No help | 0.110 | 25 | 0.0 | 0 | 1.00 | 167.7 |

## Learning gain by profile

| Tutor | novice | intermediate | expert | mixed |
|---|---|---|---|---|
| RL tutor (PPO) | 0.374 | 0.217 | 0.088 | 0.234 |
| Explain mistakes | 0.373 | 0.213 | 0.090 | 0.237 |
| Mastery-threshold rule | 0.316 | 0.181 | 0.070 | 0.198 |
| No help | 0.164 | 0.114 | 0.054 | 0.107 |

## RL tutor vs each rule (paired over the same students)

| Rule | Gain difference (RL - rule) | p | Interventions difference | Team score difference | p |
|---|---|---|---|---|---|
| Explain mistakes | +0.000 | 0.9700 | -1.0 | -0.2 | 0.206 |
| Mastery-threshold rule | +0.037 | 0.0000 | -1.9 | -0.4 | 0.497 |
| No help | +0.118 | 0.0000 | +6.3 | +0.6 | 0.327 |

## Interruptions and attention, RL tutor vs each rule (paired)

| Rule | Interventions difference | p | Attention difference | p |
|---|---|---|---|---|
| Explain mistakes | -1.0 | 0.0000 | +0.080 | 0.0000 |
| Mastery-threshold rule | -1.9 | 0.0000 | +0.072 | 0.0021 |
