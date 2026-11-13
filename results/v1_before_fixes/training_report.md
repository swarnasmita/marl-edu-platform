# MAPPO training report

Convergence episode = first episode where the 10-iteration moving average of the training return reaches 95% of its plateau (mean of the last 15 iterations).

| Run | Episodes | Training time (min) | Plateau return | Convergence episode | Best validation (greedy / idle teammate) | Final entropy |
|---|---|---|---|---|---|---|
| mappo_seed0 | 2400 | 35.3 | 142.8 | 976 | 153.8 / 118.5 (episode 1280) | 0.075 |
