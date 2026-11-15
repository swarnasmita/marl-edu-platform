# Why MAPPO is the right algorithm for this platform

Four MARL algorithms were trained under identical conditions (same hierarchical set-up, features, network size,
reward, 2,400 episodes, simulated teammates; 3 seeds each): MAPPO, IPPO, MAA2C and QMIX.
Details: algorithm_comparison.md, robustness_unseen_learners.md, judge_comparison.md.

The platform's requirements decide which properties matter:
1. the robots must work well with **real beginners**, who behave unlike the simulated teammates used in training;
2. **every training run** must give good robots (the platform is retrained for new environments);
3. the policy must be a **good judge** of the learner's decisions (the learner model depends on it);
4. it must rescue victims well, especially critical ones.

## 1. Learners never seen in training - MAPPO is the only algorithm no other one beats

Four beginner-like learner types none of the algorithms trained with (hesitant, copycat, severity-blind,
very noisy); 50 scenarios each, so 200 paired comparisons per algorithm (paired t-test, other minus MAPPO):

| Algorithm | Team score | vs MAPPO | Critical victims lost per episode | vs MAPPO |
|---|---|---|---|---|
| **MAPPO** | **166.1** | - | **0.10** | - |
| IPPO | 166.3 | +0.2 (p = 0.55, tie) | 0.13 | **+0.035 worse (p < 0.001)** |
| MAA2C | 164.0 | **−2.1 worse (p = 0.008)** | 0.10 | 0.00 (tie) |
| QMIX | 164.4 | **−1.6 worse (p = 0.002)** | 0.09 | −0.003 (p = 0.62, tie) |
| Greedy rule | 160.7 | −5.4 worse (p < 0.001) | 0.28 | +0.18 worse (p < 0.001) |

Every other algorithm is significantly worse than MAPPO on at least one of the two measures; MAPPO is
significantly worse than none of them. (IPPO matches MAPPO's score but loses about a third more critical victims;
MAA2C and QMIX match its critical losses but score lower.)

Strength of this evidence (rl_statistics.md): the paired tests stay significant after Holm-Bonferroni correction
(MAA2C p = 0.016, QMIX p = 0.006), but the stratified-bootstrap 95% intervals, which also account for differences
between training seeds, just touch zero (MAA2C -2.1 [-4.1, +0.1], QMIX -1.6 [-3.8, +0.4]). So the score
advantage over MAA2C and QMIX is moderate evidence, not conclusive; IPPO's extra critical losses are the clearest
difference.

## 2. Reliability across training runs - MAPPO about twice as consistent

Standard deviation of the mean team score across the 3 training seeds (lower = every run gives similar robots):

| Algorithm | Standard test (100 scenarios x 3 teammates) | Unseen learners (independent test) |
|---|---|---|
| **MAPPO** | **0.84** | **0.69** |
| IPPO | 1.83 | 1.41 |
| MAA2C | 1.74 | 1.53 |
| QMIX | 1.44 | 1.57 |

The same ordering appears in two independent tests. (With 3 seeds per algorithm this is a consistent
observation rather than a formal significance test.)

## 3. Judging the learner's decisions - MAPPO gives the clearest, most stable signal

Every policy judged the same decisions of simulated learners of known skill (judge_comparison.md):

| Judge | Quality gap, skilled minus random learner | AUC skilled vs random |
|---|---|---|
| **MAPPO** | **0.255 ± 0.024** | 0.763 ± 0.019 |
| IPPO | 0.191 ± 0.120 | 0.741 ± 0.048 |
| MAA2C | 0.247 ± 0.048 | 0.786 ± 0.040 |
| QMIX | 0.041 ± 0.014 | 0.695 ± 0.096 |

MAPPO has the largest gap and the smallest variation across seeds; MAA2C is close on average (slightly higher
AUC) but less stable; QMIX's values are nearly flat, so it can hardly tell learners apart.

## 4. Learning speed

Episodes to converge: IPPO 549, **MAPPO 603**, QMIX 1,035, MAA2C 1,595.

## 5. Standard test with the trained teammate types - all four are close

On the 100 standard test scenarios all four algorithms clearly beat the hand-written rules (critical victim
rescue rate about 95% vs 90%) and are within about one percentage point of each other. There MAPPO does not
lead: MAA2C (95.6%) and QMIX (95.4%) rescue slightly more critical victims than MAPPO (94.7%), and IPPO scores
slightly higher with the greedy teammate. MAPPO's advantage appears where the platform needs it: with unseen
learners, across training runs and as a judge.

## Conclusion

On the familiar test all four algorithms perform similarly. **MAPPO outperforms the alternatives on the
properties this platform depends on**: it is the only algorithm that no other algorithm beats with learners it
never trained with, it is about twice as consistent across training runs, and it gives the clearest and most
stable signal for judging learners - while learning faster than MAA2C and QMIX.

Limitations: 3 seeds per algorithm; the same hyperparameters for all algorithms (not tuned per algorithm);
QMIX adapted for asynchronous decisions; simulated learners only (no human study yet).
