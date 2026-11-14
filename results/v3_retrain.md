# Version 3: retrained in the fixed environment

Same algorithm, reward and settings as version 2; the only difference is that training ran in the fixed
environment (see env_fixes.md). 3 seeds on Kaggle, 2,400 episodes each (~73 min per seed).
Both versions tested in the fixed environment on the same 100 unseen scenarios per simulated human.

## Training

| | v2 (old environment) | v3 (fixed environment) |
|---|---|---|
| Convergence episode (3 seeds) | 750 ± 160 | **603 ± 113** |
| Plateau training return | 140.5 ± 1.6 | 159.3 ± 0.6 |

(The plateau returns are not directly comparable - the environment changed - but v3 learns faster and the
three seeds agree more closely.)

## Evaluation (mean of 3 seeds; p = paired t-test v3 vs v2 over the 100 scenarios)

| Simulated human | Metric | v2 | v3 | p |
|---|---|---|---|---|
| Greedy | Team score | 171.1 | 172.0 | 0.24 |
| | Victims lost | 0.51 | 0.47 | 0.06 |
| | Critical victims lost | 0.10 | 0.12 | 0.09 |
| | R1-R2 duplicate choices | 5.8% | **4.7%** | <0.001 |
| | Wasted trips per episode | 2.71 | **2.35** | <0.001 |
| Noisy | Team score | 167.8 | 169.4 | 0.09 |
| | Victims lost | 0.61 | **0.55** | 0.045 |
| | Critical victims lost | 0.13 | 0.13 | 0.83 |
| | R1-R2 duplicate choices | 6.7% | **5.3%** | <0.001 |
| | Wasted trips per episode | 2.62 | **2.31** | <0.001 |
| Idle | Team score | 139.1 | **140.5** | 0.040 |
| | Victims lost | 1.28 | **1.24** | 0.019 |
| | Critical victims lost | 0.23 | 0.22 | 0.85 |
| | R1-R2 duplicate choices | 6.3% | **4.7%** | <0.001 |
| | Wasted trips per episode | 1.44 | **1.04** | <0.001 |

v3 vs the best hand-written rule (greedy rule): team score +6.8 / +7.5 / +6.2 (greedy / noisy / idle human,
all p <= 0.001); critical victims lost -0.10 / -0.16 / -0.17 (p = 0.015 / <0.001 / 0.005).
v3 seeds are also more consistent: team score with the greedy human 171.8 / 171.8 / 172.3.

**Summary:** retraining in the fixed environment gave fewer duplicate choices and wasted trips (consistently
significant), slightly fewer victims lost, and faster, more consistent training. Critical victims lost did
not change (already low).

## Final model
v3 seed 1 (`agents/checkpoints/mappo_disaster.pt`), chosen by **validation** score (mean of greedy / idle
teammate: seed 0 154.4, seed 1 155.0, seed 2 154.3) - not by the test results.

## Behaviour tests with the final model (results/behaviour_report.md)
Mostly unchanged from v2. Two controlled tests moved:
- Same severity, one victim 10+ m closer: picks the nearer one 80% (v2: 92%).
- Human robot moving towards a victim: leaves that victim to the human only 20% of the time (v2 seed 0 in the
  fixed environment: 48%). In full episodes, however, the rate at which blue robots choose the human's current
  target is the same as the rules' (5.6% vs 5.1-5.5%). How much the agent relies on the inferred-intent
  signal varies between training seeds; this remains a limitation.

## Learner model with the final model (results/learner_model_validation.md)
Decision quality still ranks the simulated learners correctly:
expert 0.80 > greedy 0.70 > noisy-30% 0.58 > noisy-60% 0.55 > random 0.46.
