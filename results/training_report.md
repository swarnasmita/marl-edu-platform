# MAPPO training report

Convergence episode = first episode where the 10-iteration moving average of the training return reaches 95% of its plateau (mean of the last 15 iterations).

| Run | Episodes | Training time (min) | Plateau return | Convergence episode | Best validation (greedy / idle teammate) | Final entropy |
|---|---|---|---|---|---|---|
| mappo_v3_seed0 | 2400 | 73.6 | 158.7 | 480 | 174.8 / 133.9 (episode 640) | 0.126 |
| mappo_v3_seed1 | 2400 | 73.1 | 158.9 | 752 | 174.6 / 135.4 (episode 1120) | 0.133 |
| mappo_v3_seed2 | 2400 | 73.1 | 160.1 | 576 | 173.2 / 135.4 (episode 800) | 0.111 |

**Across 3 seeds:** convergence at episode 603 ± 113, plateau return 159.3 ± 0.6.
