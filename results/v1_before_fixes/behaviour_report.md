# Behaviour tests: blue robots (MAPPO vs two hand-written rules)

## Part A: controlled situations

Each situation was built by hand on 40 random placements of the deciding robot and the victims (same placements for both policies). Pass = the robot made the expected choice.

| Area | Situation | Expected | MAPPO | Greedy rule | Deadline-aware rule |
|---|---|---|---|---|---|
| 1 Victim selection | Severity: critical vs minor, same effort | pick critical | 100% (40/40) | 100% (40/40) | 100% (40/40) |
| 1 Victim selection | Distance: same severity, one 10+ m closer | pick nearer | 90% (36/40) | 100% (40/40) | 100% (40/40) |
| 1 Victim selection | Trade-off: minor nearby vs critical 10+ m further (reachable) | pick critical | 100% (40/40) | 75% (30/40) | 75% (30/40) |
| 1 Victim selection | Impossible deadline: critical that can't be reached in time vs moderate | pick moderate | 0% (0/40) | 0% (0/40) | 100% (40/40) |
| 1 Victim selection | Urgency: two critical, one about to run out | pick urgent one | 95% (38/40) | 48% (19/40) | 48% (19/40) |
| 2 R1-R2 coordination | R2 heading for a victim and 2-4 m from it | pick the other victim | 100% (40/40) | 100% (40/40) | 100% (40/40) |
| 2 R1-R2 coordination | R1 and R2 free together, two critical victims | different victims | 100% (40/40) | 100% (40/40) | 100% (40/40) |
| 3 Human coordination | Human robot 2-4 m from a victim | pick the other victim | 48% (19/40) | 62% (25/40) | 62% (25/40) |
| 4 Re-planning | Heading for a minor victim when a critical one is reported (reachable) | switch to critical | 100% (40/40) | 100% (40/40) | 100% (40/40) |
| 4 Re-planning | Heading for a critical victim when a minor one is reported | keep critical | 100% (40/40) | 100% (40/40) | 100% (40/40) |

## Part B: full episodes (50 unseen scenarios per simulated human)

| Measure | MAPPO, greedy human | MAPPO, noisy human | Greedy rule, greedy human | Greedy rule, noisy human | Deadline-aware rule, greedy human | Deadline-aware rule, noisy human |
|---|---|---|---|---|---|---|
| 2 R1-R2: Duplicate choices (% of blue decisions) | 5.6% | 5.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 R1-R2: Wasted trips: target taken by the other blue robot (per episode) | 1.20 | 1.04 | 0.10 | 0.06 | 0.08 | 0.06 |
| 3 Human: Clashes: blue chose the human's current target (% of decisions) | 6.7% | 6.1% | 4.8% | 4.0% | 4.2% | 3.9% |
| 3 Human: Wasted trips: target taken by the human (per episode) | 1.64 | 1.38 | 1.36 | 1.06 | 1.20 | 1.06 |
| 4 Re-planning: Reaction time to a new report: first robot heading there (s) | 9.2 | 8.9 | 8.4 | 7.8 | 7.7 | 7.5 |
| 4 Re-planning: Late-reported victims nobody ever went for (%) | 0% | 0% | 0% | 0% | 0% | 0% |
| 4 Re-planning: Survival: critical victims reported late | 97% | 96% | 96% | 94% | 94% | 94% |
| 4 Re-planning: Survival: critical victims known from the start | 94% | 93% | 88% | 83% | 87% | 82% |
| 5 Waste: Wasted trips: target lost before arrival (per episode) | 0.64 | 0.64 | 0.46 | 0.54 | 0.14 | 0.10 |
| 5 Stuck: Stuck events: has a goal but no progress for 2+ s (per episode) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 Stuck: Time stuck (s per episode) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 5 Stuck: Time two robots are touching (s per episode) | 5.7 | 4.8 | 4.7 | 4.8 | 4.7 | 4.8 |
| 5 Waste: Idle while victims are waiting (s per blue robot per episode) | 0.1 | 0.1 | 2.8 | 2.2 | 3.0 | 2.4 |
| 5 Waste: Path efficiency: distance driven / planned route (median) | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 |
| 5 Waste: Path efficiency: 90th percentile | 1.11 | 1.12 | 1.13 | 1.10 | 1.12 | 1.12 |
| -: Team score | 158.5 | 156.5 | 153.0 | 152.0 | 150.7 | 152.7 |
