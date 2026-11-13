# Learner model validation (simulated learners, 30 scenarios each)

Blue robots: trained MAPPO. Correct / assessed choices per concept, and mean BKT mastery at the end of an episode (only episodes where the concept was tested).

| Learner | Team score | Decision quality | Prioritising by severity | Acting before deadlines | Not duplicating a teammate | Near vs far: long-term value | Re-planning when new victims appear |
|---|---|---|---|---|---|---|---|
| expert | 157.2 | 0.86 | 95% of 57 (mastery 0.73) | 100% of 7 (mastery 0.59) | 96% of 56 (mastery 0.74) | 88% of 16 (mastery 0.54) | 86% of 22 (mastery 0.53) |
| greedy | 159.8 | 0.82 | 98% of 49 (mastery 0.74) | 100% of 7 (mastery 0.59) | 93% of 42 (mastery 0.67) | 67% of 18 (mastery 0.43) | 84% of 19 (mastery 0.50) |
| noisy 30% | 155.1 | 0.71 | 85% of 54 (mastery 0.68) | 100% of 11 (mastery 0.57) | 100% of 46 (mastery 0.74) | 53% of 17 (mastery 0.40) | 71% of 17 (mastery 0.45) |
| noisy 60% | 157.9 | 0.54 | 68% of 53 (mastery 0.51) | 83% of 6 (mastery 0.47) | 96% of 45 (mastery 0.68) | 43% of 14 (mastery 0.33) | 50% of 16 (mastery 0.36) |
| random | 156.2 | 0.49 | 65% of 52 (mastery 0.51) | 100% of 12 (mastery 0.54) | 98% of 44 (mastery 0.69) | 29% of 17 (mastery 0.26) | 32% of 19 (mastery 0.27) |
