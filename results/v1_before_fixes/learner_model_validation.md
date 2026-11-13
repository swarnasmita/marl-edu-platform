# Learner model validation (simulated learners, 30 scenarios each)

Blue robots: trained MAPPO. Correct / assessed choices per concept, and mean BKT mastery at the end of an episode (only episodes where the concept was tested).

| Learner | Team score | Decision quality | Prioritising by severity | Acting before deadlines | Not duplicating a teammate | Near vs far: long-term value | Re-planning when new victims appear |
|---|---|---|---|---|---|---|---|
| expert | 158.1 | 0.77 | 89% of 56 (mastery 0.70) | 80% of 5 (mastery 0.46) | 91% of 46 (mastery 0.66) | 77% of 13 (mastery 0.46) | 83% of 23 (mastery 0.52) |
| greedy | 161.0 | 0.62 | 98% of 53 (mastery 0.74) | 100% of 10 (mastery 0.57) | 96% of 51 (mastery 0.75) | 59% of 17 (mastery 0.40) | 79% of 19 (mastery 0.49) |
| noisy 30% | 157.9 | 0.58 | 85% of 55 (mastery 0.68) | 100% of 11 (mastery 0.57) | 100% of 46 (mastery 0.72) | 50% of 18 (mastery 0.38) | 63% of 19 (mastery 0.41) |
| noisy 60% | 160.6 | 0.48 | 71% of 55 (mastery 0.54) | 86% of 7 (mastery 0.48) | 100% of 41 (mastery 0.71) | 31% of 16 (mastery 0.29) | 56% of 18 (mastery 0.38) |
| random | 153.4 | 0.38 | 60% of 57 (mastery 0.47) | 100% of 12 (mastery 0.54) | 95% of 44 (mastery 0.66) | 11% of 19 (mastery 0.17) | 21% of 19 (mastery 0.22) |
