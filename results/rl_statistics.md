# RL results with robust statistics

Following Agarwal et al. (2021): interval estimates instead of single numbers. 95% CIs from a stratified bootstrap (2000 resamples of test scenarios and training seeds). IQM = interquartile mean (mean of the middle 50% of episodes). Paired comparisons: per test scenario, averaged over each algorithm's 3 seeds; d_z = mean difference / sd of differences (0.2 small, 0.5 medium, 0.8 large); p-values Holm-Bonferroni corrected within each table; P(MAPPO better) = probability of improvement on a random test scenario.

## Team score - standard test (100 unseen scenarios per simulated human)

| Policy | Human | Mean [95% CI] | IQM [95% CI] |
|---|---|---|---|
| **MAPPO** | greedy | 172.0 [167.5, 176.1] | 179.8 [174.5, 185.1] |
| Greedy rule | greedy | 165.2 [161.1, 169.3] | 167.1 [160.6, 173.1] |
| Deadline-aware rule | greedy | 163.8 [159.4, 168.4] | 165.6 [159.1, 171.9] |
| Random target | greedy | 136.7 [130.8, 142.6] | 135.7 [129.3, 142.9] |
| IPPO | greedy | 173.6 [169.1, 177.8] | 181.4 [175.5, 186.5] |
| MAA2C | greedy | 171.8 [167.0, 176.1] | 178.0 [171.6, 183.9] |
| QMIX | greedy | 171.1 [166.2, 175.7] | 178.4 [172.0, 183.9] |
| Full-information planner | greedy | 170.3 [165.5, 174.9] | 177.7 [171.6, 183.0] |
| **MAPPO** | noisy | 169.4 [164.9, 173.7] | 176.2 [170.2, 181.9] |
| Greedy rule | noisy | 161.9 [157.0, 166.9] | 164.8 [157.5, 171.0] |
| Deadline-aware rule | noisy | 161.3 [156.4, 166.1] | 163.7 [156.4, 170.2] |
| Random target | noisy | 126.0 [119.2, 132.8] | 125.4 [117.8, 132.9] |
| IPPO | noisy | 170.6 [166.2, 174.8] | 178.2 [172.3, 183.4] |
| MAA2C | noisy | 167.5 [163.3, 171.6] | 171.3 [165.8, 176.7] |
| QMIX | noisy | 169.1 [164.5, 173.6] | 175.5 [169.0, 181.3] |
| Full-information planner | noisy | 166.1 [161.4, 170.9] | 171.3 [164.0, 177.7] |
| **MAPPO** | idle | 140.5 [134.6, 145.8] | 141.7 [135.6, 147.1] |
| Greedy rule | idle | 134.3 [128.9, 139.4] | 138.1 [131.2, 143.5] |
| Deadline-aware rule | idle | 133.2 [128.0, 138.2] | 135.2 [128.4, 141.4] |
| Random target | idle | 56.8 [50.7, 62.7] | 53.0 [46.7, 60.2] |
| IPPO | idle | 138.9 [131.9, 144.9] | 140.2 [132.0, 146.8] |
| MAA2C | idle | 134.2 [127.8, 140.6] | 135.2 [127.4, 141.9] |
| QMIX | idle | 140.6 [135.5, 145.5] | 141.7 [136.7, 146.8] |
| Full-information planner | idle | 144.9 [140.1, 149.7] | 145.5 [140.3, 150.2] |

Paired comparison with MAPPO (Team score; difference = other minus MAPPO):

| Other | Human | Difference [95% CI] | d_z | p (Holm) | P(MAPPO better) |
|---|---|---|---|---|---|
| Greedy rule | greedy | -6.8 [-10.8, -2.9] | -0.35 | 0.00868 | 60% |
| Deadline-aware rule | greedy | -8.1 [-11.9, -4.2] | -0.43 | 0.00053 | 62% |
| Random target | greedy | -35.3 [-41.1, -29.3] | -1.20 | 8.11e-20 | 84% |
| IPPO | greedy | +1.7 [-0.6, +4.6] | +0.22 | 0.297 | 40% |
| MAA2C | greedy | -0.1 [-2.9, +3.4] | -0.01 | 1 | 51% |
| QMIX | greedy | -0.8 [-3.8, +2.2] | -0.10 | 1 | 51% |
| Full-information planner | greedy | -1.7 [-4.9, +1.8] | -0.10 | 1 | 51% |
| Greedy rule | noisy | -7.4 [-11.6, -3.3] | -0.38 | 0.00398 | 60% |
| Deadline-aware rule | noisy | -8.1 [-11.6, -4.7] | -0.45 | 0.000282 | 57% |
| Random target | noisy | -43.4 [-50.3, -36.4] | -1.22 | 4.03e-20 | 87% |
| IPPO | noisy | +1.2 [-0.7, +3.1] | +0.17 | 0.585 | 42% |
| MAA2C | noisy | -1.9 [-4.9, +0.8] | -0.16 | 0.634 | 62% |
| QMIX | noisy | -0.2 [-3.1, +2.2] | -0.03 | 1 | 43% |
| Full-information planner | noisy | -3.3 [-6.7, +0.5] | -0.18 | 0.576 | 51% |
| Greedy rule | idle | -6.2 [-10.1, -1.9] | -0.35 | 0.00868 | 53% |
| Deadline-aware rule | idle | -7.3 [-11.4, -3.3] | -0.41 | 0.00145 | 54% |
| Random target | idle | -83.7 [-90.4, -76.8] | -2.43 | 4.02e-42 | 99% |
| IPPO | idle | -1.6 [-7.2, +3.2] | -0.25 | 0.147 | 58% |
| MAA2C | idle | -6.3 [-11.3, -1.8] | -0.60 | 6.14e-07 | 79% |
| QMIX | idle | +0.1 [-2.6, +3.3] | +0.02 | 1 | 41% |
| Full-information planner | idle | +4.4 [+1.3, +7.7] | +0.35 | 0.00807 | 30% |

## Critical victims rescued - standard test (100 unseen scenarios per simulated human)

| Policy | Human | Mean [95% CI] | IQM [95% CI] |
|---|---|---|---|
| **MAPPO** | greedy | 96.0% [93.7%, 98.0%] | 100.0% [100.0%, 100.0%] |
| Greedy rule | greedy | 92.7% [90.0%, 95.3%] | 100.0% [96.7%, 100.0%] |
| Deadline-aware rule | greedy | 92.3% [89.3%, 95.0%] | 100.0% [96.0%, 100.0%] |
| Random target | greedy | 76.3% [72.3%, 80.3%] | 75.3% [68.7%, 81.3%] |
| IPPO | greedy | 96.1% [94.2%, 97.9%] | 100.0% [100.0%, 100.0%] |
| MAA2C | greedy | 97.6% [95.8%, 99.0%] | 100.0% [100.0%, 100.0%] |
| QMIX | greedy | 96.4% [94.3%, 98.1%] | 100.0% [100.0%, 100.0%] |
| Full-information planner | greedy | 95.3% [92.7%, 97.7%] | 100.0% [100.0%, 100.0%] |
| **MAPPO** | noisy | 95.7% [93.6%, 97.6%] | 100.0% [100.0%, 100.0%] |
| Greedy rule | noisy | 90.3% [87.0%, 93.3%] | 98.7% [92.7%, 100.0%] |
| Deadline-aware rule | noisy | 90.0% [86.7%, 93.0%] | 98.0% [92.0%, 100.0%] |
| Random target | noisy | 70.3% [65.7%, 75.0%] | 71.3% [66.0%, 77.3%] |
| IPPO | noisy | 95.8% [93.4%, 97.9%] | 100.0% [100.0%, 100.0%] |
| MAA2C | noisy | 96.1% [93.2%, 98.3%] | 100.0% [100.0%, 100.0%] |
| QMIX | noisy | 96.2% [94.1%, 98.1%] | 100.0% [100.0%, 100.0%] |
| Full-information planner | noisy | 93.3% [90.3%, 96.0%] | 100.0% [98.7%, 100.0%] |
| **MAPPO** | idle | 92.6% [89.7%, 95.2%] | 100.0% [96.0%, 100.0%] |
| Greedy rule | idle | 87.0% [83.0%, 91.0%] | 94.7% [88.7%, 100.0%] |
| Deadline-aware rule | idle | 86.0% [82.3%, 90.0%] | 92.7% [86.7%, 99.3%] |
| Random target | idle | 37.7% [33.0%, 42.3%] | 36.0% [33.3%, 42.0%] |
| IPPO | idle | 90.8% [86.4%, 94.6%] | 98.4% [90.4%, 100.0%] |
| MAA2C | idle | 93.2% [89.2%, 96.7%] | 100.0% [95.3%, 100.0%] |
| QMIX | idle | 93.7% [90.3%, 97.0%] | 100.0% [97.5%, 100.0%] |
| Full-information planner | idle | 92.7% [90.0%, 95.3%] | 100.0% [96.0%, 100.0%] |

Paired comparison with MAPPO (Critical victims rescued; difference = other minus MAPPO):

| Other | Human | Difference [95% CI] | d_z | p (Holm) | P(MAPPO better) |
|---|---|---|---|---|---|
| Greedy rule | greedy | -3.3 pts [-6.3 pts, -0.6 pts] | -0.24 | 0.203 | 54% |
| Deadline-aware rule | greedy | -3.7 pts [-6.4 pts, -1.0 pts] | -0.28 | 0.0897 | 54% |
| Random target | greedy | -19.7 pts [-24.0 pts, -15.3 pts] | -0.89 | 4.67e-13 | 76% |
| IPPO | greedy | +0.1 pts [-1.8 pts, +1.9 pts] | +0.02 | 1 | 50% |
| MAA2C | greedy | +1.6 pts [-0.1 pts, +3.8 pts] | +0.22 | 0.33 | 46% |
| QMIX | greedy | +0.4 pts [-1.0 pts, +2.2 pts] | +0.09 | 1 | 50% |
| Full-information planner | greedy | -0.7 pts [-2.8 pts, +1.6 pts] | -0.07 | 1 | 50% |
| Greedy rule | noisy | -5.3 pts [-8.2 pts, -2.4 pts] | -0.36 | 0.01 | 56% |
| Deadline-aware rule | noisy | -5.7 pts [-8.6 pts, -3.0 pts] | -0.40 | 0.00241 | 57% |
| Random target | noisy | -25.3 pts [-30.1 pts, -20.4 pts] | -1.05 | 2.15e-16 | 80% |
| IPPO | noisy | +0.1 pts [-1.7 pts, +1.7 pts] | +0.02 | 1 | 50% |
| MAA2C | noisy | +0.4 pts [-2.2 pts, +2.8 pts] | +0.07 | 1 | 49% |
| QMIX | noisy | +0.6 pts [-0.8 pts, +1.9 pts] | +0.14 | 1 | 48% |
| Full-information planner | noisy | -2.3 pts [-5.0 pts, +0.3 pts] | -0.17 | 0.749 | 52% |
| Greedy rule | idle | -5.6 pts [-9.9 pts, -1.7 pts] | -0.28 | 0.0897 | 55% |
| Deadline-aware rule | idle | -6.6 pts [-10.7 pts, -2.6 pts] | -0.32 | 0.0252 | 57% |
| Random target | idle | -54.9 pts [-60.2 pts, -49.4 pts] | -2.04 | 6.12e-36 | 97% |
| IPPO | idle | -1.8 pts [-5.4 pts, +1.2 pts] | -0.33 | 0.0223 | 57% |
| MAA2C | idle | +0.7 pts [-3.0 pts, +4.3 pts] | +0.07 | 1 | 50% |
| QMIX | idle | +1.1 pts [-1.7 pts, +4.2 pts] | +0.20 | 0.494 | 46% |
| Full-information planner | idle | +0.1 pts [-3.3 pts, +3.6 pts] | +0.01 | 1 | 47% |

## Severity saved - standard test (100 unseen scenarios per simulated human)

| Policy | Human | Mean [95% CI] | IQM [95% CI] |
|---|---|---|---|
| **MAPPO** | greedy | 94.7% [93.2%, 96.2%] | 97.6% [95.6%, 99.5%] |
| Greedy rule | greedy | 92.3% [90.9%, 93.8%] | 93.2% [90.8%, 95.2%] |
| Deadline-aware rule | greedy | 91.9% [90.4%, 93.2%] | 92.6% [90.2%, 94.6%] |
| Random target | greedy | 82.6% [80.7%, 84.4%] | 82.2% [80.0%, 84.7%] |
| IPPO | greedy | 95.3% [93.8%, 96.6%] | 98.1% [96.2%, 100.0%] |
| MAA2C | greedy | 94.7% [93.1%, 96.2%] | 96.9% [94.7%, 99.2%] |
| QMIX | greedy | 94.4% [92.7%, 96.0%] | 97.0% [94.6%, 99.1%] |
| Full-information planner | greedy | 94.1% [92.5%, 95.7%] | 96.8% [94.8%, 98.6%] |
| **MAPPO** | noisy | 93.8% [92.3%, 95.2%] | 96.3% [94.2%, 98.3%] |
| Greedy rule | noisy | 91.2% [89.5%, 92.8%] | 92.2% [89.8%, 94.6%] |
| Deadline-aware rule | noisy | 90.9% [89.2%, 92.5%] | 91.8% [89.4%, 94.1%] |
| Random target | noisy | 79.0% [76.6%, 81.2%] | 78.6% [76.0%, 81.4%] |
| IPPO | noisy | 94.2% [92.7%, 95.7%] | 96.9% [95.2%, 98.8%] |
| MAA2C | noisy | 93.2% [91.9%, 94.6%] | 94.6% [92.7%, 96.4%] |
| QMIX | noisy | 93.7% [92.1%, 95.2%] | 95.9% [93.7%, 98.0%] |
| Full-information planner | noisy | 92.6% [91.0%, 94.2%] | 94.6% [92.1%, 96.8%] |
| **MAPPO** | idle | 85.8% [84.0%, 87.5%] | 86.2% [84.3%, 88.1%] |
| Greedy rule | idle | 83.5% [81.6%, 85.4%] | 85.0% [82.4%, 86.9%] |
| Deadline-aware rule | idle | 83.1% [81.4%, 85.0%] | 84.0% [81.4%, 86.2%] |
| Random target | idle | 57.0% [55.1%, 59.1%] | 55.6% [53.4%, 58.2%] |
| IPPO | idle | 85.2% [82.9%, 87.3%] | 85.8% [83.1%, 87.9%] |
| MAA2C | idle | 83.7% [81.6%, 85.9%] | 84.1% [81.5%, 86.3%] |
| QMIX | idle | 85.7% [84.0%, 87.4%] | 86.2% [84.4%, 87.9%] |
| Full-information planner | idle | 87.2% [85.6%, 88.8%] | 87.5% [85.8%, 89.0%] |

Paired comparison with MAPPO (Severity saved; difference = other minus MAPPO):

| Other | Human | Difference [95% CI] | d_z | p (Holm) | P(MAPPO better) |
|---|---|---|---|---|---|
| Greedy rule | greedy | -2.4 pts [-3.8 pts, -1.1 pts] | -0.37 | 0.00503 | 65% |
| Deadline-aware rule | greedy | -2.9 pts [-4.1 pts, -1.5 pts] | -0.45 | 0.000257 | 67% |
| Random target | greedy | -12.2 pts [-14.2 pts, -10.3 pts] | -1.22 | 3.11e-20 | 88% |
| IPPO | greedy | +0.6 pts [-0.2 pts, +1.5 pts] | +0.21 | 0.343 | 46% |
| MAA2C | greedy | -0.0 pts [-1.0 pts, +1.1 pts] | -0.01 | 1 | 50% |
| QMIX | greedy | -0.3 pts [-1.4 pts, +0.6 pts] | -0.12 | 1 | 56% |
| Full-information planner | greedy | -0.6 pts [-1.7 pts, +0.5 pts] | -0.11 | 1 | 54% |
| Greedy rule | noisy | -2.7 pts [-4.0 pts, -1.3 pts] | -0.40 | 0.00187 | 64% |
| Deadline-aware rule | noisy | -2.9 pts [-4.3 pts, -1.7 pts] | -0.48 | 0.00011 | 64% |
| Random target | noisy | -14.9 pts [-17.2 pts, -12.6 pts] | -1.24 | 1.4e-20 | 88% |
| IPPO | noisy | +0.4 pts [-0.2 pts, +1.0 pts] | +0.17 | 0.698 | 46% |
| MAA2C | noisy | -0.6 pts [-1.6 pts, +0.3 pts] | -0.15 | 0.818 | 55% |
| QMIX | noisy | -0.2 pts [-1.1 pts, +0.7 pts] | -0.06 | 1 | 54% |
| Full-information planner | noisy | -1.2 pts [-2.5 pts, +0.0 pts] | -0.20 | 0.401 | 57% |
| Greedy rule | idle | -2.2 pts [-3.5 pts, -0.9 pts] | -0.37 | 0.00499 | 60% |
| Deadline-aware rule | idle | -2.6 pts [-4.0 pts, -1.3 pts] | -0.43 | 0.000625 | 64% |
| Random target | idle | -28.7 pts [-31.1 pts, -26.3 pts] | -2.43 | 3.72e-42 | 100% |
| IPPO | idle | -0.5 pts [-2.2 pts, +1.0 pts] | -0.24 | 0.17 | 56% |
| MAA2C | idle | -2.0 pts [-3.5 pts, -0.6 pts] | -0.57 | 1.9e-06 | 68% |
| QMIX | idle | -0.0 pts [-0.9 pts, +1.1 pts] | -0.01 | 1 | 49% |
| Full-information planner | idle | +1.5 pts [+0.4 pts, +2.6 pts] | +0.33 | 0.0166 | 34% |

## Team score with learners never seen in training (4 types x 50 scenarios, pooled)

| Policy | Mean [95% CI] | IQM [95% CI] | Difference from MAPPO [95% CI] | d_z | p (Holm) | P(MAPPO better) |
|---|---|---|---|---|---|---|
| Greedy rule | 160.7 [157.2, 164.0] | 163.3 [158.6, 167.8] | -5.4 [-8.1, -2.9] | -0.30 | 0.000175 | 59% |
| **MAPPO** | 166.1 [162.9, 169.1] | 170.5 [166.4, 174.5] | - | - | - | - |
| IPPO | 166.3 [163.1, 169.4] | 171.7 [167.0, 176.0] | +0.2 [-1.1, +1.5] | +0.04 | 0.549 | 49% |
| MAA2C | 164.0 [160.3, 167.4] | 169.5 [164.2, 174.0] | -2.1 [-4.2, +0.1] | -0.19 | 0.0158 | 62% |
| QMIX | 164.4 [160.6, 168.1] | 168.4 [163.8, 172.5] | -1.6 [-3.9, +0.3] | -0.22 | 0.00573 | 55% |

## How close to the full-information planner? (standard test)

Normalised score = (score - random target) / (full-information planner - random target): 0 = random, 100% = the planner that knows every victim in advance. Mean over the 3 simulated humans. The planner is a strong reference, not the optimum: it knows the future but does not model how the human will act, so learned policies can exceed 100% with an active human; with the idle human (the robots do everything) it is ahead of every learned policy.

| Policy | Normalised score [95% CI] |
|---|---|
| MAPPO | 103% [98%, 108%] |
| Greedy rule | 87% [82%, 92%] |
| Deadline-aware rule | 85% [81%, 90%] |
| IPPO | 105% [100%, 110%] |
| MAA2C | 99% [93%, 104%] |
| QMIX | 102% [97%, 107%] |
| Full-information planner | 100% [100%, 100%] |

