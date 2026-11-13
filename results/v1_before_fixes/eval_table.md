# Disaster Response: policy comparison (100 unseen scenarios per teammate)

± = standard error over scenarios.

Success = at least 80% of total severity saved. Wait = seconds from a victim being reported to being picked up. Duplicate choices = share of blue-robot decisions that picked the victim the other blue robot was already heading for. Wasted trips = targets taken by someone else or lost before arrival. Balance = rescues of the less busy blue robot / the busier one (1 = equal).

## Teammate in the learner's slot: greedy

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 119.2 ± 3.6 | 77.2% | 48% | 8.07 | 1.73 | 0.83 | 72% |
| Greedy rule (severity / distance) | 150.3 ± 2.8 | 87.9% | 86% | 8.87 | 0.98 | 0.32 | 89% |
| Deadline-aware rule | 148.3 ± 2.9 | 87.2% | 84% | 8.81 | 1.02 | 0.34 | 89% |
| MAPPO (hierarchical) | 155.9 ± 3.1 | 89.9% | 89% | 9.02 | 0.78 | 0.23 | 92% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 12.3 | 17.4 | 36.9 | 41.5 | 5.7% | 2.53 | 14.7 | 35% | 0.75 |
| Greedy rule (severity / distance) | 11.3 | 17.8 | 46.0 | 48.5 | 0.0% | 2.02 | 18.2 | 30% | 0.85 |
| Deadline-aware rule | 11.5 | 17.4 | 45.5 | 47.7 | 0.0% | 1.46 | 18.0 | 29% | 0.84 |
| MAPPO (hierarchical) | 10.9 | 20.3 | 53.9 | 41.7 | 5.0% | 3.33 | 14.0 | 34% | 0.84 |

Paired comparison (team score), MAPPO minus greedy rule (severity / distance): +5.58 (t = 2.53, p ≈ 0.011, n = 100).  
Paired comparison (critical victims lost), MAPPO minus greedy rule (severity / distance): -0.09 (t = -2.38, p ≈ 0.017, n = 100).  

## Teammate in the learner's slot: noisy

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 107.6 ± 3.5 | 73.6% | 35% | 7.79 | 2.00 | 1.04 | 65% |
| Greedy rule (severity / distance) | 146.0 ± 3.0 | 86.4% | 76% | 8.76 | 1.05 | 0.43 | 86% |
| Deadline-aware rule | 147.3 ± 3.0 | 86.7% | 77% | 8.82 | 1.03 | 0.44 | 85% |
| MAPPO (hierarchical) | 151.2 ± 3.4 | 88.1% | 82% | 8.89 | 0.93 | 0.30 | 90% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 12.4 | 18.0 | 32.9 | 48.3 | 4.7% | 2.66 | 18.2 | 35% | 0.76 |
| Greedy rule (severity / distance) | 11.4 | 18.6 | 44.1 | 47.4 | 0.0% | 1.74 | 17.7 | 30% | 0.84 |
| Deadline-aware rule | 11.4 | 18.7 | 44.0 | 44.1 | 0.0% | 1.24 | 16.4 | 30% | 0.84 |
| MAPPO (hierarchical) | 11.2 | 20.5 | 50.9 | 41.3 | 5.9% | 3.12 | 14.0 | 33% | 0.83 |

Paired comparison (team score), MAPPO minus deadline-aware rule: +3.90 (t = 1.49, p ≈ 0.136, n = 100).  
Paired comparison (critical victims lost), MAPPO minus deadline-aware rule: -0.14 (t = -2.97, p ≈ 0.003, n = 100).  

## Teammate in the learner's slot: idle

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 54.3 ± 3.1 | 56.5% | 0% | 6.47 | 3.36 | 1.81 | 40% |
| Greedy rule (severity / distance) | 122.5 ± 3.1 | 79.7% | 55% | 8.17 | 1.69 | 0.54 | 82% |
| Deadline-aware rule | 123.1 ± 2.9 | 79.9% | 55% | 8.19 | 1.67 | 0.55 | 82% |
| MAPPO (hierarchical) | 125.2 ± 3.6 | 80.6% | 66% | 8.14 | 1.56 | 0.42 | 86% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 12.9 | 19.4 | 40.4 | 28.1 | 7.1% | 2.11 | 7.8 | 0% | 0.79 |
| Greedy rule (severity / distance) | 12.1 | 21.0 | 64.6 | 33.5 | 0.0% | 0.87 | 10.4 | 0% | 0.85 |
| Deadline-aware rule | 12.3 | 20.7 | 63.6 | 32.4 | 0.0% | 0.20 | 10.2 | 0% | 0.86 |
| MAPPO (hierarchical) | 11.6 | 25.8 | 76.4 | 28.9 | 6.2% | 2.13 | 7.2 | 0% | 0.84 |

Paired comparison (team score), MAPPO minus deadline-aware rule: +2.11 (t = 0.86, p ≈ 0.392, n = 100).  
Paired comparison (critical victims lost), MAPPO minus deadline-aware rule: -0.13 (t = -2.31, p ≈ 0.021, n = 100).  
