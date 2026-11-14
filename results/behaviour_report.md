# Behaviour tests: blue robots (MAPPO vs two hand-written rules)

## Part A: controlled situations

Each situation was built by hand on 40 random placements of the deciding robot and the victims (same placements for both policies). Pass = the robot made the expected choice.

| Area | Situation | Expected | MAPPO | Greedy rule | Deadline-aware rule |
|---|---|---|---|---|---|
| 1 Victim selection | Severity: critical vs minor, same effort | pick critical | 100% (40/40) | 100% (40/40) | 100% (40/40) |
| 1 Victim selection | Distance: same severity, one 10+ m closer | pick nearer | 80% (32/40) | 100% (40/40) | 100% (40/40) |
| 1 Victim selection | Trade-off: minor nearby vs critical 10+ m further (reachable) | pick critical | 100% (40/40) | 78% (31/40) | 78% (31/40) |
| 1 Victim selection | Impossible deadline: critical that can't be reached in time vs moderate | pick moderate | 100% (40/40) | 0% (0/40) | 100% (40/40) |
| 1 Victim selection | Urgency: two critical, one about to run out | pick urgent one | 100% (40/40) | 60% (24/40) | 60% (24/40) |
| 2 R1-R2 coordination | R2 heading for a victim and 2-4 m from it | pick the other victim | 100% (40/40) | 100% (40/40) | 100% (40/40) |
| 2 R1-R2 coordination | R1 and R2 free together, two critical victims | different victims | 100% (40/40) | 100% (40/40) | 100% (40/40) |
| 3 Human coordination | Human robot 2-4 m from a victim | pick the other victim | 45% (18/40) | 55% (22/40) | 55% (22/40) |
| 3 Human coordination | Human robot 2-4 m from a victim and moving towards it | pick the other victim | 20% (8/40) | 55% (22/40) | 55% (22/40) |
| 4 Re-planning | Heading for a minor victim when a critical one is reported (reachable) | switch to critical | 100% (40/40) | 100% (40/40) | 100% (40/40) |
| 4 Re-planning | Heading for a critical victim when a minor one is reported | keep critical | 100% (40/40) | 100% (40/40) | 100% (40/40) |

## Part B: full episodes (50 unseen scenarios per simulated human)

| Measure | MAPPO, greedy human | MAPPO, noisy human | Greedy rule, greedy human | Greedy rule, noisy human | Deadline-aware rule, greedy human | Deadline-aware rule, noisy human |
|---|---|---|---|---|---|---|
| 2 R1-R2: Duplicate choices (% of blue decisions) | 5.0% | 4.9% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 R1-R2: Wasted trips: target taken by the other blue robot (per episode) | 0.98 | 0.98 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 Human: Clashes: blue chose the human's current target (% of decisions) | 5.1% | 5.6% | 5.3% | 5.5% | 4.9% | 4.9% |
| 3 Human: Wasted trips: target taken by the human (per episode) | 1.40 | 1.32 | 1.38 | 1.14 | 1.28 | 1.14 |
| 4 Re-planning: Reaction time to a new report: first robot heading there (s) | 8.8 | 8.5 | 7.6 | 7.1 | 7.4 | 7.0 |
| 4 Re-planning: Late-reported victims nobody ever went for (%) | 0% | 0% | 0% | 0% | 0% | 0% |
| 4 Re-planning: Survival: critical victims reported late | 99% | 99% | 100% | 96% | 100% | 97% |
| 4 Re-planning: Survival: critical victims known from the start | 99% | 98% | 89% | 88% | 89% | 88% |
| 5 Waste: Wasted trips: target lost before arrival (per episode) | 0.04 | 0.04 | 0.34 | 0.42 | 0.02 | 0.02 |
| 5 Stuck: Stuck events: has a goal but no route progress for 5+ s (per episode) | 0.08 | 0.14 | 0.08 | 0.04 | 0.08 | 0.12 |
| 5 Stuck: Time stuck (s per episode) | 0.2 | 0.3 | 0.2 | 0.1 | 0.2 | 0.2 |
| 5 Stuck: Time two robots are touching (s per episode) | 7.3 | 7.5 | 6.7 | 6.1 | 6.9 | 5.9 |
| 5 Waste: Idle while victims are waiting (s per blue robot per episode) | 0.1 | 0.1 | 1.4 | 1.6 | 1.6 | 1.8 |
| 5 Waste: Path efficiency: distance driven / planned route (median) | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 |
| 5 Waste: Path efficiency: 90th percentile | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 1.00 |
| -: Team score | 173.6 | 171.1 | 167.4 | 165.1 | 166.2 | 165.4 |
