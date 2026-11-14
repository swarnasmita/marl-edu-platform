# Environment fixes found by watching the trained agents (version 2.1)

Watching the trained blue robots in the viewer showed two problems that the metrics had missed.
Both were in the environment and the low-level movement code, not in the learned policy.

## What was wrong

| Problem | Cause | Fix |
|---|---|---|
| Robots "turning round and round" outside a house / in a back alley | The movement rule looks 0.8 m ahead in 8 directions. Before a sharp turn in a narrow gap, every forward direction looked blocked, so the robot rocked back and forth forever | In tight spots, step to the neighbouring route-map cell closest to the goal (`GreedyPolicy._descend`) |
| Robots blocking each other at the triage tarp | A paramedic figure (a solid obstacle) stood in the only entrance, making it one lane wide | Moved him beside the ambulance; the entrance is now two lanes wide |
| Two robots meeting head-on, both waiting | No right-of-way rule | Right of way: human's robot first, then a robot carrying a victim, then lower index (`GreedyPolicy._gives_way`) |
| Robot blocked by a parked robot | Route planner ignored robots | After 2 s without progress next to a robot, plan a detour that treats it as an obstacle (`Navigator._check_blocked`) |
| Robot with nothing to do parked in a narrow alley | Idle robots just stopped | Idle robots move aside when a busy robot is close (`Navigator.make_way`) |
| Dead-end back alley only one robot wide | Scene geometry | Widened the alley (2.4 m) and its entrance (2.2 m) |
| A blue robot heading for a critical victim picked up a moderate one on the way and had to deliver them first | Any empty robot within 1 m of any victim picked them up automatically | Blue robots only pick up the victim they chose (`env.pickup_only`); the human still picks up anyone they touch |

The "stuck" metric in the behaviour tests only counted robots that were not moving, so a robot rocking back and
forth was never counted. It now counts robots whose route distance to their goal has not shrunk for 5 s.
`envs/check_scene.py` now checks that every robot start and victim location is reachable after any scene change.

## Effect (same trained agents, no retraining; 100 unseen scenarios x 3 simulated humans)

Stuck events (5 s without progress) in 30 episodes with an idle human: **17 before, 0 after**.

Trained agent (3 Kaggle seeds) vs the best hand-written rule in each setting (p = paired t-test over the 100 scenarios):

| Simulated human | | Before fixes | After fixes |
|---|---|---|---|
| Greedy | Team score: MAPPO / rule | 151.3 / 149.8 (p = 0.52) | **171.1 / 164.8 (p = 0.001)** |
| | Critical victims lost: MAPPO / rule | 0.27 / 0.33 (p = 0.17) | **0.10 / 0.22 (p = 0.001)** |
| Noisy | Team score | 149.6 / 146.1 (p = 0.22) | **167.8 / 161.7 (p = 0.001)** |
| | Critical victims lost | 0.33 / 0.44 (p = 0.015) | **0.13 / 0.29 (p < 0.001)** |
| Idle | Team score | 123.4 / 123.1 (p = 0.88) | **139.1 / 134.3 (p = 0.007)** |
| | Critical victims lost | 0.47 / 0.55 (p = 0.16) | **0.23 / 0.39 (p = 0.005)** |

Each of the three trained seeds, on its own, scored higher than the greedy rule and lost fewer critical victims
with every simulated human; the difference is significant for most seed/human pairs but not all (for example
seed 1 with the idle human: +2.0, p = 0.33). Before the fixes, getting stuck and automatic pickups
interrupted the agent's plans; the rules, which re-decide greedily, suffered less from this.

Still open: R1 and R2 choose the same victim in about 4-9% of decisions (the rules never do).
The agents were trained in the old environment; retraining in the fixed one (version 3) may improve further.
