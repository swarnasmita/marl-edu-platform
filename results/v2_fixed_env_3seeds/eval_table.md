# Disaster Response: policy comparison (100 unseen scenarios per teammate)

± = standard error over scenarios for the rules; for MAPPO, sd across 3 training seeds.

Success = at least 80% of total severity saved. Wait = seconds from a victim being reported to being picked up. Duplicate choices = share of blue-robot decisions that picked the victim the other blue robot was already heading for. Wasted trips = targets taken by someone else or lost before arrival. Balance = rescues of the less busy blue robot / the busier one (1 = equal).

## Teammate in the learner's slot: greedy

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 136.7 ± 2.9 | 82.6% | 63% | 8.61 | 1.39 | 0.71 | 76% |
| Greedy rule (severity / distance) | 164.8 ± 2.1 | 92.2% | 96% | 9.33 | 0.67 | 0.22 | 93% |
| Deadline-aware rule | 163.5 ± 2.2 | 91.8% | 95% | 9.29 | 0.71 | 0.23 | 92% |
| MAPPO (hierarchical) | 171.1 ± 1.7 | 94.5% | 97% | 9.49 | 0.51 | 0.10 | 97% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 12.0 | 18.5 | 32.1 | 31.3 | 6.2% | 2.41 | 9.9 | 37% | 0.83 |
| Greedy rule (severity / distance) | 10.8 | 18.0 | 45.0 | 37.3 | 0.0% | 1.73 | 12.7 | 31% | 0.86 |
| Deadline-aware rule | 10.9 | 17.7 | 44.2 | 36.7 | 0.0% | 1.28 | 12.7 | 31% | 0.85 |
| MAPPO (hierarchical) | 10.5 | 20.4 | 51.1 | 35.3 | 5.8% | 2.71 | 11.3 | 36% | 0.86 |

Paired comparison (team score), MAPPO minus greedy rule (severity / distance): +6.25 (t = 3.44, p ≈ 0.001, n = 100).  
Paired comparison (critical victims lost), MAPPO minus greedy rule (severity / distance): -0.12 (t = -3.21, p ≈ 0.001, n = 100).  

## Teammate in the learner's slot: noisy

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 126.4 ± 3.4 | 79.1% | 49% | 8.35 | 1.65 | 0.88 | 71% |
| Greedy rule (severity / distance) | 161.7 ± 2.5 | 91.1% | 91% | 9.25 | 0.75 | 0.29 | 90% |
| Deadline-aware rule | 161.0 ± 2.5 | 90.8% | 91% | 9.23 | 0.77 | 0.30 | 90% |
| MAPPO (hierarchical) | 167.8 ± 1.8 | 93.3% | 96% | 9.39 | 0.61 | 0.13 | 96% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 11.9 | 18.3 | 29.6 | 32.5 | 5.4% | 2.43 | 10.5 | 36% | 0.82 |
| Greedy rule (severity / distance) | 11.2 | 18.1 | 42.6 | 35.8 | 0.0% | 1.51 | 12.1 | 31% | 0.83 |
| Deadline-aware rule | 11.1 | 18.0 | 41.8 | 36.4 | 0.0% | 1.07 | 12.6 | 31% | 0.84 |
| MAPPO (hierarchical) | 10.9 | 21.1 | 48.1 | 35.5 | 6.7% | 2.62 | 11.4 | 35% | 0.85 |

Paired comparison (team score), MAPPO minus greedy rule (severity / distance): +6.16 (t = 3.31, p ≈ 0.001, n = 100).  
Paired comparison (critical victims lost), MAPPO minus greedy rule (severity / distance): -0.16 (t = -3.97, p ≈ 0.000, n = 100).  

## Teammate in the learner's slot: idle

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 57.2 ± 3.0 | 57.2% | 1% | 6.65 | 3.35 | 1.86 | 38% |
| Greedy rule (severity / distance) | 134.3 ± 2.7 | 83.5% | 74% | 8.49 | 1.40 | 0.39 | 87% |
| Deadline-aware rule | 133.2 ± 2.7 | 83.1% | 71% | 8.48 | 1.44 | 0.42 | 86% |
| MAPPO (hierarchical) | 139.1 ± 2.3 | 85.2% | 82% | 8.56 | 1.28 | 0.23 | 92% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 12.8 | 20.4 | 38.2 | 24.8 | 7.0% | 1.94 | 6.3 | 0% | 0.83 |
| Greedy rule (severity / distance) | 11.9 | 21.5 | 67.9 | 31.8 | 0.0% | 0.56 | 9.6 | 0% | 0.87 |
| Deadline-aware rule | 11.8 | 21.3 | 65.9 | 30.9 | 0.0% | 0.02 | 9.5 | 0% | 0.88 |
| MAPPO (hierarchical) | 11.6 | 27.1 | 77.8 | 28.5 | 6.3% | 1.44 | 7.5 | 0% | 0.86 |

Paired comparison (team score), MAPPO minus greedy rule (severity / distance): +4.79 (t = 2.72, p ≈ 0.007, n = 100).  
Paired comparison (critical victims lost), MAPPO minus greedy rule (severity / distance): -0.16 (t = -2.79, p ≈ 0.005, n = 100).  
