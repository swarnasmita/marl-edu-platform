# Learner model validation (simulated learners, 30 scenarios each)

Blue robots: trained MAPPO. Correct / assessed choices per concept, and mean BKT mastery at the end of an episode (only episodes where the concept was tested).

| Learner | Team score | Decision quality | Prioritising by severity | Acting before deadlines | Not duplicating a teammate | Near vs far: long-term value | Re-planning when new victims appear |
|---|---|---|---|---|---|---|---|
| expert | 179.6 | 0.80 | 99% of 76 (mastery 0.86) | 100% of 3 (mastery 0.54) | 95% of 58 (mastery 0.71) | 95% of 20 (mastery 0.60) | 81% of 21 (mastery 0.49) |
| greedy | 178.6 | 0.70 | 100% of 67 (mastery 0.85) | 100% of 9 (mastery 0.57) | 100% of 52 (mastery 0.75) | 60% of 20 (mastery 0.43) | 84% of 19 (mastery 0.52) |
| noisy 30% | 174.0 | 0.58 | 86% of 70 (mastery 0.72) | 100% of 11 (mastery 0.57) | 95% of 58 (mastery 0.75) | 50% of 18 (mastery 0.36) | 64% of 22 (mastery 0.40) |
| noisy 60% | 171.5 | 0.55 | 71% of 70 (mastery 0.58) | 100% of 11 (mastery 0.60) | 97% of 58 (mastery 0.73) | 57% of 23 (mastery 0.37) | 50% of 24 (mastery 0.35) |
| random | 171.2 | 0.46 | 66% of 73 (mastery 0.55) | 100% of 9 (mastery 0.54) | 98% of 55 (mastery 0.75) | 48% of 23 (mastery 0.34) | 15% of 20 (mastery 0.21) |
