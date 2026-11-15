# Pedagogical agent inside the real game

Student model v2. Simulated students played the learner's robot in real MuJoCo episodes (5 episodes each, 20 students per profile, the same students for every tutor), with the trained blue robots. The tutor acted at every real decision of the student.

| Tutor | Learning gain | Decisions | Interventions | Interruption (s) | Attention at end | Motivation at end | Team score |
|---|---|---|---|---|---|---|---|
| RL tutor (PPO) | 0.166 | 25 | 5.6 | 22 | 0.97 | 0.66 | 168.6 |
| Explain mistakes | 0.174 | 25 | 7.4 | 30 | 0.85 | 0.63 | 168.9 |
| Mastery-threshold rule | 0.152 | 25 | 8.0 | 20 | 0.89 | 0.82 | 168.9 |
| No help | 0.085 | 25 | 0.0 | 0 | 1.00 | 0.73 | 168.2 |

## Learning gain by profile

| Tutor | novice | intermediate | expert | mixed |
|---|---|---|---|---|
| RL tutor (PPO) | 0.264 | 0.164 | 0.065 | 0.171 |
| Explain mistakes | 0.274 | 0.175 | 0.068 | 0.178 |
| Mastery-threshold rule | 0.244 | 0.157 | 0.057 | 0.151 |
| No help | 0.119 | 0.093 | 0.042 | 0.085 |

## RL tutor vs each rule (paired over the same students)

| Rule | Gain difference (RL - rule) | p | Interventions difference | Team score difference | p |
|---|---|---|---|---|---|
| Explain mistakes | -0.008 | 0.0000 | -1.9 | -0.3 | 0.395 |
| Mastery-threshold rule | +0.014 | 0.0000 | -2.4 | -0.3 | 0.622 |
| No help | +0.081 | 0.0000 | +5.6 | +0.4 | 0.514 |

## Interruptions, attention and motivation, RL tutor vs each rule (paired)

| Rule | Interventions difference | p | Attention difference | p | Motivation difference | p |
|---|---|---|---|---|---|---|
| Explain mistakes | -1.85 | 0.0000 | +0.12 | 0.0000 | +0.03 | 0.0000 |
| Mastery-threshold rule | -2.40 | 0.0000 | +0.08 | 0.0008 | -0.16 | 0.0000 |
