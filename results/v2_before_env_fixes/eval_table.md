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
| MAPPO (hierarchical) | 154.5 ± 3.1 | 89.4% | 88% | 8.98 | 0.82 | 0.25 | 91% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 12.3 | 17.4 | 36.9 | 41.5 | 5.7% | 2.53 | 14.7 | 35% | 0.75 |
| Greedy rule (severity / distance) | 11.3 | 17.8 | 46.0 | 48.5 | 0.0% | 2.02 | 18.2 | 30% | 0.85 |
| Deadline-aware rule | 11.5 | 17.4 | 45.5 | 47.7 | 0.0% | 1.46 | 18.0 | 29% | 0.84 |
| MAPPO (hierarchical) | 10.7 | 19.8 | 53.1 | 43.9 | 3.3% | 2.26 | 15.7 | 33% | 0.84 |

Paired comparison (team score), MAPPO minus greedy rule (severity / distance): +4.18 (t = 1.99, p ≈ 0.047, n = 100).  
Paired comparison (critical victims lost), MAPPO minus greedy rule (severity / distance): -0.07 (t = -1.83, p ≈ 0.067, n = 100).  

## Teammate in the learner's slot: noisy

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 107.6 ± 3.5 | 73.6% | 35% | 7.79 | 2.00 | 1.04 | 65% |
| Greedy rule (severity / distance) | 146.0 ± 3.0 | 86.4% | 76% | 8.76 | 1.05 | 0.43 | 86% |
| Deadline-aware rule | 147.3 ± 3.0 | 86.7% | 77% | 8.82 | 1.03 | 0.44 | 85% |
| MAPPO (hierarchical) | 150.3 ± 3.2 | 87.9% | 81% | 8.87 | 0.96 | 0.31 | 89% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 12.4 | 18.0 | 32.9 | 48.3 | 4.7% | 2.66 | 18.2 | 35% | 0.76 |
| Greedy rule (severity / distance) | 11.4 | 18.6 | 44.1 | 47.4 | 0.0% | 1.74 | 17.7 | 30% | 0.84 |
| Deadline-aware rule | 11.4 | 18.7 | 44.0 | 44.1 | 0.0% | 1.24 | 16.4 | 30% | 0.84 |
| MAPPO (hierarchical) | 11.1 | 20.0 | 50.8 | 42.0 | 3.2% | 1.96 | 14.9 | 33% | 0.84 |

Paired comparison (team score), MAPPO minus deadline-aware rule: +2.98 (t = 1.16, p ≈ 0.246, n = 100).  
Paired comparison (critical victims lost), MAPPO minus deadline-aware rule: -0.13 (t = -2.80, p ≈ 0.005, n = 100).  

## Teammate in the learner's slot: idle

**Task performance**

| Policy for blue robots | Team score | Severity saved | Success rate | Rescued /10 | Lost | Critical lost | Critical survival |
|---|---|---|---|---|---|---|---|
| Random target | 54.3 ± 3.1 | 56.5% | 0% | 6.47 | 3.36 | 1.81 | 40% |
| Greedy rule (severity / distance) | 122.5 ± 3.1 | 79.7% | 55% | 8.17 | 1.69 | 0.54 | 82% |
| Deadline-aware rule | 123.1 ± 2.9 | 79.9% | 55% | 8.19 | 1.67 | 0.55 | 82% |
| MAPPO (hierarchical) | 126.1 ± 3.2 | 80.9% | 65% | 8.17 | 1.55 | 0.42 | 86% |

**Response time and coordination**

| Policy for blue robots | Wait: critical (s) | Wait: moderate (s) | Wait: minor (s) | Decisions | Duplicate choices | Wasted trips | Idle (s per robot) | Rescues by learner | Balance R1/R2 |
|---|---|---|---|---|---|---|---|---|---|
| Random target | 12.9 | 19.4 | 40.4 | 28.1 | 7.1% | 2.11 | 7.8 | 0% | 0.79 |
| Greedy rule (severity / distance) | 12.1 | 21.0 | 64.6 | 33.5 | 0.0% | 0.87 | 10.4 | 0% | 0.85 |
| Deadline-aware rule | 12.3 | 20.7 | 63.6 | 32.4 | 0.0% | 0.20 | 10.2 | 0% | 0.86 |
| MAPPO (hierarchical) | 11.5 | 25.0 | 73.9 | 28.6 | 4.0% | 1.12 | 7.7 | 0% | 0.84 |

Paired comparison (team score), MAPPO minus deadline-aware rule: +3.01 (t = 1.48, p ≈ 0.139, n = 100).  
Paired comparison (critical victims lost), MAPPO minus deadline-aware rule: -0.13 (t = -2.57, p ≈ 0.010, n = 100).  
