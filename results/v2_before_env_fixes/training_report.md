# MAPPO training report

Convergence episode = first episode where the 10-iteration moving average of the training return reaches 95% of its plateau (mean of the last 15 iterations).

| Run | Episodes | Training time (min) | Plateau return | Convergence episode | Best validation (greedy / idle teammate) | Final entropy |
|---|---|---|---|---|---|---|
| mappo_v2_seed0 | 2400 | 41.8 | 140.2 | 576 | 154.3 / 117.3 (episode 1280) | 0.096 |
