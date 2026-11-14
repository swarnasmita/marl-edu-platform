# MAPPO version 1 vs version 2 (fixes from the behaviour tests)

Version 2 changes: (1) victims that can no longer be reached in time are not offered while a feasible one exists; (2) a small reward penalty (-1) for choosing the victim the other blue robot is already heading for; (3) a feature telling each blue robot which victim the human appears to be heading for (inferred from movement; right about 74% of the time). Both versions: seed 0, same training settings, same test scenarios.

## Behaviour tests: controlled situations (pass rate)

| Situation | Version 1 | Version 2 | Deadline-aware rule |
|---|---|---|---|
| Severity: critical vs minor, same effort | 100% (40/40) | **100% (40/40)** | 100% (40/40) |
| Distance: same severity, one 10+ m closer | 90% (36/40) | **90% (36/40)** | 100% (40/40) |
| Trade-off: minor nearby vs critical 10+ m further (reachable) | 100% (40/40) | **100% (40/40)** | 75% (30/40) |
| Impossible deadline: critical that can't be reached in time vs moderate | 0% (0/40) | **100% (40/40)** | 100% (40/40) |
| Urgency: two critical, one about to run out | 95% (38/40) | **100% (40/40)** | 48% (19/40) |
| R2 heading for a victim and 2-4 m from it | 100% (40/40) | **100% (40/40)** | 100% (40/40) |
| R1 and R2 free together, two critical victims | 100% (40/40) | **100% (40/40)** | 100% (40/40) |
| Human robot 2-4 m from a victim | 48% (19/40) | **25% (10/40)** | 62% (25/40) |
| Human robot 2-4 m from a victim and moving towards it | - (new test) | **98% (39/40)** | 62% (25/40) |
| Heading for a minor victim when a critical one is reported (reachable) | 100% (40/40) | **100% (40/40)** | 100% (40/40) |
| Heading for a critical victim when a minor one is reported | 100% (40/40) | **100% (40/40)** | 100% (40/40) |

## Behaviour tests: full episodes (greedy simulated human)

| Measure | Version 1 | Version 2 | Greedy rule |
|---|---|---|---|
| 4 Re-planning: Survival: critical victims reported late | 97% | **96%** | 96% |
| 4 Re-planning: Survival: critical victims known from the start | 94% | **94%** | 88% |
| 5 Waste: Path efficiency: 90th percentile | 1.11 | **1.11** | 1.13 |
| -: Team score | 158.5 | **157.3** | 153.0 |

## Evaluation: 100 unseen scenarios per teammate

| Teammate | Metric | Version 1 | Version 2 | Best rule |
|---|---|---|---|---|
| greedy | Team score | 155.9 | **154.5** | 150.3 |
| greedy | Severity saved | 89.9% | **89.4%** | 87.9% |
| greedy | Success rate | 89% | **88%** | 86% |
| greedy | Critical lost | 0.23 | **0.25** | 0.32 |
| greedy | Duplicate choices | 5.0% | **3.3%** | 0.0% |
| greedy | Wasted trips | 3.33 | **2.26** | 2.02 |
| greedy | Wait: moderate (s) | 20.3 | **19.8** | 17.8 |
| noisy | Team score | 151.2 | **150.3** | 147.3 |
| noisy | Severity saved | 88.1% | **87.9%** | 86.7% |
| noisy | Success rate | 82% | **81%** | 77% |
| noisy | Critical lost | 0.30 | **0.31** | 0.44 |
| noisy | Duplicate choices | 5.9% | **3.2%** | 0.0% |
| noisy | Wasted trips | 3.12 | **1.96** | 1.24 |
| noisy | Wait: moderate (s) | 20.5 | **20.0** | 18.7 |
| idle | Team score | 125.2 | **126.1** | 123.1 |
| idle | Severity saved | 80.6% | **80.9%** | 79.9% |
| idle | Success rate | 66% | **65%** | 55% |
| idle | Critical lost | 0.42 | **0.42** | 0.55 |
| idle | Duplicate choices | 6.2% | **4.0%** | 0.0% |
| idle | Wasted trips | 2.13 | **1.12** | 0.20 |
| idle | Wait: moderate (s) | 25.8 | **25.0** | 20.7 |
