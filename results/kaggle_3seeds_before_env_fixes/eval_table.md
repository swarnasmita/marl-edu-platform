# Disaster Response: policy comparison (100 unseen scenarios per teammate)

± = standard error over scenarios for the rules; for MAPPO, sd across 3 training seeds.

Success = at least 80% of total severity saved. Wait = seconds from a victim being reported to being picked up. Duplicate choices = share of blue-robot decisions that picked the victim the other blue robot was already heading for. Wasted trips = targets taken by someone else or lost before arrival. Balance = rescues of the less busy blue robot / the busier one (1 = equal).

## Teammate in the learner's slot: greedy

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 120.2 ± 3.6 | 77.6% | 48% | 8.11 | 1.71 | 0.82 | 72% |
| Greedy rule (severity / distance) | 149.8 ± 2.9 | 87.6% | 85% | 8.86 | 0.98 | 0.33 | 89% |
| Deadline-aware rule | 149.5 ± 2.9 | 87.5% | 85% | 8.84 | 0.97 | 0.33 | 89% |
| MAPPO (hierarchical) | 151.3 ± 3.2 | 88.3% | 85% | 8.88 | 0.89 | 0.27 | 90% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 12.2 | 17.6 | 36.7 | 39.4 | 6.0% | 2.45 | 13.7 | 35% | 0.75 |
| Greedy rule (severity / distance) | 11.3 | 17.9 | 46.5 | 46.2 | 0.0% | 2.03 | 17.0 | 30% | 0.84 |
| Deadline-aware rule | 11.5 | 17.8 | 45.4 | 46.7 | 0.0% | 1.44 | 17.5 | 29% | 0.84 |
| MAPPO (hierarchical) | 10.8 | 20.1 | 53.1 | 45.4 | 5.7% | 3.01 | 16.1 | 33% | 0.85 |

Paired comparison (team score), MAPPO minus greedy rule (severity / distance): +1.53 (t = 0.65, p ≈ 0.517, n = 100).  
Paired comparison (critical victims lost), MAPPO minus greedy rule (severity / distance): -0.06 (t = -1.37, p ≈ 0.170, n = 100).  

## Teammate in the learner's slot: noisy

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 107.6 ± 3.5 | 73.5% | 34% | 7.77 | 1.99 | 1.04 | 65% |
| Greedy rule (severity / distance) | 146.0 ± 3.0 | 86.3% | 75% | 8.74 | 1.02 | 0.41 | 86% |
| Deadline-aware rule | 146.1 ± 3.1 | 86.2% | 75% | 8.76 | 1.03 | 0.44 | 85% |
| MAPPO (hierarchical) | 149.6 ± 1.8 | 87.6% | 80% | 8.86 | 0.95 | 0.33 | 89% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 12.4 | 17.9 | 33.1 | 48.9 | 4.3% | 2.67 | 18.5 | 35% | 0.75 |
| Greedy rule (severity / distance) | 11.5 | 18.7 | 44.7 | 50.2 | 0.0% | 1.77 | 19.1 | 30% | 0.84 |
| Deadline-aware rule | 11.3 | 18.7 | 44.5 | 48.0 | 0.0% | 1.31 | 18.3 | 30% | 0.84 |
| MAPPO (hierarchical) | 11.0 | 20.9 | 50.9 | 42.9 | 5.7% | 2.69 | 15.0 | 33% | 0.81 |

Paired comparison (team score), MAPPO minus deadline-aware rule: +3.49 (t = 1.24, p ≈ 0.215, n = 100).  
Paired comparison (critical victims lost), MAPPO minus deadline-aware rule: -0.11 (t = -2.44, p ≈ 0.015, n = 100).  

## Teammate in the learner's slot: idle

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 54.4 ± 3.1 | 56.5% | 0% | 6.47 | 3.35 | 1.81 | 40% |
| Greedy rule (severity / distance) | 123.0 ± 3.1 | 79.9% | 55% | 8.18 | 1.68 | 0.53 | 82% |
| Deadline-aware rule | 123.1 ± 2.9 | 79.8% | 55% | 8.18 | 1.67 | 0.55 | 82% |
| MAPPO (hierarchical) | 123.4 ± 2.1 | 80.0% | 61% | 8.11 | 1.62 | 0.47 | 84% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 12.8 | 19.5 | 41.8 | 28.4 | 7.2% | 2.16 | 7.9 | 0% | 0.79 |
| Greedy rule (severity / distance) | 12.3 | 21.0 | 65.2 | 33.0 | 0.0% | 0.86 | 10.1 | 0% | 0.85 |
| Deadline-aware rule | 12.3 | 20.7 | 64.1 | 32.4 | 0.0% | 0.21 | 10.2 | 0% | 0.86 |
| MAPPO (hierarchical) | 11.7 | 26.1 | 74.7 | 28.2 | 7.2% | 1.51 | 7.2 | 0% | 0.82 |

Paired comparison (team score), MAPPO minus deadline-aware rule: +0.34 (t = 0.15, p ≈ 0.878, n = 100).  
Paired comparison (critical victims lost), MAPPO minus deadline-aware rule: -0.08 (t = -1.42, p ≈ 0.155, n = 100).  
