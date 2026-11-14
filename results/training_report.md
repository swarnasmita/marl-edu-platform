# MAPPO training report

Convergence episode = first episode where the 10-iteration moving average of the training return reaches 95% of its plateau (mean of the last 15 iterations).

| Run | Episodes | Training time (min) | Plateau return | Convergence episode | Best validation (greedy / idle teammate) | Final entropy |
|---|---|---|---|---|---|---|
| mappo_v2_seed0 | 2400 | 76.8 | 138.4 | 576 | 153.5 / 118.8 (episode 1280) | 0.087 |
| mappo_v2_seed1 | 2400 | 76.5 | 140.8 | 960 | 153.5 / 118.7 (episode 1280) | 0.115 |
| mappo_v2_seed2 | 2400 | 75.6 | 142.2 | 704 | 154.3 / 113.5 (episode 640) | 0.057 |

**Across 3 seeds:** convergence at episode 747 ± 160, plateau return 140.5 ± 1.6.
