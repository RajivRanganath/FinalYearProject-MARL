# QMIX Deployment Ensemble Evaluation

Split: `final`. This experiment changes deployment aggregation only; trained weights, environment, and reward are unchanged.

| Policy | Reward | Recall | Energy | AoI | Redundancy | Sample fraction |
|---|---:|---:|---:|---:|---:|---:|
| Entropy Threshold | 159.70 | 0.611 | 10.82 | 5.64 | 48.30 | 0.214 |
| Battery + Entropy | 150.45 | 0.600 | 10.62 | 5.82 | 47.13 | 0.210 |
| Greedy | -191.35 | 0.331 | 19.36 | 8.82 | 184.23 | 0.640 |
| Extended QMIX Replica Mean | 153.38 | 0.608 | 11.78 | 5.10 | 53.84 | 0.234 |
| QMIX Unanimous Ensemble | 157.43 | 0.612 | 11.55 | 5.21 | 52.70 | 0.230 |

## Paired candidate comparisons

Positive values favor the candidate in the engineering direction. Holm adjustment covers the displayed comparison family.

| Candidate | Comparator | Metric | Advantage | 95% CI | Holm p |
|---|---|---|---:|---:|---:|
| QMIX Unanimous Ensemble | Extended QMIX Replica Mean | raw_episode_reward | 4.0516 | [2.9228, 5.1804] | 3.928e-07 |
| QMIX Unanimous Ensemble | Extended QMIX Replica Mean | event_recall | 0.0037 | [0.0024, 0.0050] | 2.486e-05 |
| QMIX Unanimous Ensemble | Extended QMIX Replica Mean | total_energy_consumption | 0.2294 | [0.1997, 0.2592] | 1.016e-14 |
| QMIX Unanimous Ensemble | Extended QMIX Replica Mean | mean_aoi | -0.1121 | [-0.1599, -0.0642] | 2.272e-04 |
| QMIX Unanimous Ensemble | Extended QMIX Replica Mean | redundant_sampling | 1.1444 | [0.8068, 1.4821] | 1.023e-06 |
| QMIX Unanimous Ensemble | Extended QMIX Replica Mean | network_coverage | 0.0010 | [0.0003, 0.0017] | 1.925e-02 |
| QMIX Unanimous Ensemble | Entropy Threshold | raw_episode_reward | -2.2685 | [-4.9232, 0.3862] | 1.822e-01 |
| QMIX Unanimous Ensemble | Entropy Threshold | event_recall | 0.0014 | [-0.0023, 0.0051] | 4.444e-01 |
| QMIX Unanimous Ensemble | Entropy Threshold | total_energy_consumption | -0.7250 | [-0.8080, -0.6420] | 4.136e-16 |
| QMIX Unanimous Ensemble | Entropy Threshold | mean_aoi | 0.4217 | [0.2530, 0.5905] | 1.117e-04 |
| QMIX Unanimous Ensemble | Entropy Threshold | redundant_sampling | -4.4000 | [-5.4029, -3.3971] | 7.268e-09 |
| QMIX Unanimous Ensemble | Entropy Threshold | network_coverage | 0.0018 | [0.0003, 0.0033] | 6.895e-02 |

## Decision

Promote the highest-reward candidate only when its paired reward 95% CI versus the three-replica Extended mean is above zero, recall advantage is at least -0.005, energy advantage at least -0.25, and redundancy advantage at least -2.0.

Promoted candidate: `QMIX Unanimous Ensemble`.
