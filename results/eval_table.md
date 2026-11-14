# Disaster Response: policy comparison (100 unseen scenarios per teammate)

± = standard error over scenarios for the rules; for MAPPO, sd across 3 training seeds.

Success = at least 80% of total severity saved. Wait = seconds from a victim being reported to being picked up. Duplicate choices = share of blue-robot decisions that picked the victim the other blue robot was already heading for. Wasted trips = targets taken by someone else or lost before arrival. Balance = rescues of the less busy blue robot / the busier one (1 = equal).

## Teammate in the learner's slot: greedy

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 136.7 ± 2.9 | 82.6% | 63% | 8.61 | 1.39 | 0.71 | 76% |
| Greedy rule (severity / distance) | 165.2 ± 2.1 | 92.3% | 96% | 9.34 | 0.66 | 0.22 | 93% |
| Deadline-aware rule | 163.8 ± 2.2 | 91.9% | 95% | 9.30 | 0.70 | 0.23 | 92% |
| MAPPO (hierarchical) | 172.0 ± 0.2 | 94.7% | 96% | 9.53 | 0.47 | 0.12 | 96% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 12.0 | 18.6 | 32.1 | 31.4 | 6.4% | 2.39 | 10.0 | 37% | 0.82 |
| Greedy rule (severity / distance) | 10.9 | 18.1 | 45.1 | 37.2 | 0.0% | 1.76 | 12.6 | 31% | 0.85 |
| Deadline-aware rule | 10.9 | 17.7 | 44.4 | 36.9 | 0.0% | 1.32 | 12.7 | 31% | 0.84 |
| MAPPO (hierarchical) | 10.5 | 20.4 | 51.6 | 35.0 | 4.7% | 2.35 | 11.4 | 36% | 0.86 |

Paired comparison (team score), MAPPO minus greedy rule (severity / distance): +6.80 (t = 3.49, p ≈ 0.000, n = 100).  
Paired comparison (critical victims lost), MAPPO minus greedy rule (severity / distance): -0.10 (t = -2.43, p ≈ 0.015, n = 100).  

## Teammate in the learner's slot: noisy

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 126.0 ± 3.4 | 79.0% | 49% | 8.34 | 1.66 | 0.89 | 70% |
| Greedy rule (severity / distance) | 161.9 ± 2.5 | 91.2% | 91% | 9.26 | 0.74 | 0.29 | 90% |
| Deadline-aware rule | 161.3 ± 2.5 | 90.9% | 91% | 9.24 | 0.76 | 0.30 | 90% |
| MAPPO (hierarchical) | 169.4 ± 0.5 | 93.8% | 95% | 9.45 | 0.55 | 0.13 | 96% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 11.9 | 18.2 | 29.6 | 32.1 | 5.3% | 2.41 | 10.4 | 36% | 0.81 |
| Greedy rule (severity / distance) | 11.2 | 18.3 | 42.7 | 35.9 | 0.0% | 1.49 | 12.1 | 31% | 0.83 |
| Deadline-aware rule | 11.1 | 18.2 | 41.8 | 36.7 | 0.0% | 1.04 | 12.7 | 31% | 0.84 |
| MAPPO (hierarchical) | 10.9 | 21.4 | 48.5 | 33.7 | 5.3% | 2.31 | 10.7 | 35% | 0.86 |

Paired comparison (team score), MAPPO minus greedy rule (severity / distance): +7.45 (t = 3.76, p ≈ 0.000, n = 100).  
Paired comparison (critical victims lost), MAPPO minus greedy rule (severity / distance): -0.16 (t = -3.55, p ≈ 0.000, n = 100).  

## Teammate in the learner's slot: idle

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 56.8 ± 3.0 | 57.0% | 1% | 6.64 | 3.36 | 1.87 | 38% |
| Greedy rule (severity / distance) | 134.3 ± 2.7 | 83.5% | 74% | 8.49 | 1.40 | 0.39 | 87% |
| Deadline-aware rule | 133.2 ± 2.7 | 83.1% | 71% | 8.48 | 1.44 | 0.42 | 86% |
| MAPPO (hierarchical) | 140.5 ± 1.8 | 85.8% | 82% | 8.61 | 1.24 | 0.22 | 93% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 12.9 | 20.3 | 38.2 | 24.5 | 7.3% | 1.91 | 6.2 | 0% | 0.83 |
| Greedy rule (severity / distance) | 11.9 | 21.5 | 67.9 | 31.7 | 0.0% | 0.56 | 9.6 | 0% | 0.87 |
| Deadline-aware rule | 11.8 | 21.3 | 66.0 | 30.8 | 0.0% | 0.02 | 9.5 | 0% | 0.88 |
| MAPPO (hierarchical) | 11.5 | 27.2 | 78.6 | 27.9 | 4.7% | 1.04 | 7.4 | 0% | 0.87 |

Paired comparison (team score), MAPPO minus greedy rule (severity / distance): +6.16 (t = 3.47, p ≈ 0.001, n = 100).  
Paired comparison (critical victims lost), MAPPO minus greedy rule (severity / distance): -0.17 (t = -2.79, p ≈ 0.005, n = 100).  
