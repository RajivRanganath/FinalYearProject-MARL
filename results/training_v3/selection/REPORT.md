# QMIX Deployment Ensemble Evaluation

Split: `selection`. This experiment changes deployment aggregation only; trained weights, environment, and reward are unchanged.

| Policy | Reward | Recall | Energy | AoI | Redundancy | Sample fraction |
|---|---:|---:|---:|---:|---:|---:|
| Entropy Threshold | 160.94 | 0.613 | 10.81 | 5.63 | 49.50 | 0.215 |
| Battery + Entropy | 152.39 | 0.602 | 10.64 | 5.81 | 48.10 | 0.212 |
| Greedy | -180.61 | 0.340 | 19.31 | 8.95 | 184.60 | 0.639 |
| Extended QMIX Replica Mean | 151.76 | 0.607 | 11.76 | 5.14 | 54.47 | 0.236 |
| QMIX Majority Ensemble | 152.03 | 0.607 | 11.73 | 5.14 | 54.05 | 0.236 |
| QMIX Unanimous Ensemble | 155.95 | 0.610 | 11.53 | 5.27 | 53.05 | 0.231 |

## Paired candidate comparisons

Positive values favor the candidate in the engineering direction. Holm adjustment covers the displayed comparison family.

| Candidate | Comparator | Metric | Advantage | 95% CI | Holm p |
|---|---|---|---:|---:|---:|
| QMIX Majority Ensemble | Extended QMIX Replica Mean | raw_episode_reward | 0.2762 | [-0.7022, 1.2547] | 1.000e+00 |
| QMIX Majority Ensemble | Extended QMIX Replica Mean | event_recall | 0.0002 | [-0.0009, 0.0012] | 1.000e+00 |
| QMIX Majority Ensemble | Extended QMIX Replica Mean | total_energy_consumption | 0.0333 | [0.0024, 0.0642] | 3.228e-01 |
| QMIX Majority Ensemble | Extended QMIX Replica Mean | mean_aoi | -0.0009 | [-0.0453, 0.0434] | 1.000e+00 |
| QMIX Majority Ensemble | Extended QMIX Replica Mean | redundant_sampling | 0.4167 | [0.0931, 0.7403] | 1.434e-01 |
| QMIX Majority Ensemble | Extended QMIX Replica Mean | network_coverage | 0.0000 | [-0.0004, 0.0004] | 1.000e+00 |
| QMIX Majority Ensemble | Entropy Threshold | raw_episode_reward | -8.9082 | [-13.0509, -4.7655] | 3.672e-03 |
| QMIX Majority Ensemble | Entropy Threshold | event_recall | -0.0065 | [-0.0108, -0.0021] | 6.243e-02 |
| QMIX Majority Ensemble | Entropy Threshold | total_energy_consumption | -0.9175 | [-1.0344, -0.8006] | 2.662e-11 |
| QMIX Majority Ensemble | Entropy Threshold | mean_aoi | 0.4897 | [0.2895, 0.6899] | 1.035e-03 |
| QMIX Majority Ensemble | Entropy Threshold | redundant_sampling | -4.5500 | [-6.0259, -3.0741] | 6.963e-05 |
| QMIX Majority Ensemble | Entropy Threshold | network_coverage | 0.0002 | [-0.0030, 0.0034] | 1.000e+00 |
| QMIX Unanimous Ensemble | Extended QMIX Replica Mean | raw_episode_reward | 4.1930 | [2.5756, 5.8104] | 5.577e-04 |
| QMIX Unanimous Ensemble | Extended QMIX Replica Mean | event_recall | 0.0035 | [0.0019, 0.0050] | 2.109e-03 |
| QMIX Unanimous Ensemble | Extended QMIX Replica Mean | total_energy_consumption | 0.2283 | [0.1757, 0.2810] | 5.407e-07 |
| QMIX Unanimous Ensemble | Extended QMIX Replica Mean | mean_aoi | -0.1265 | [-0.1915, -0.0616] | 9.006e-03 |
| QMIX Unanimous Ensemble | Extended QMIX Replica Mean | redundant_sampling | 1.4167 | [0.9906, 1.8428] | 2.617e-05 |
| QMIX Unanimous Ensemble | Extended QMIX Replica Mean | network_coverage | 0.0008 | [-0.0000, 0.0017] | 4.669e-01 |
| QMIX Unanimous Ensemble | Entropy Threshold | raw_episode_reward | -4.9915 | [-8.2251, -1.7578] | 5.280e-02 |
| QMIX Unanimous Ensemble | Entropy Threshold | event_recall | -0.0032 | [-0.0069, 0.0006] | 6.586e-01 |
| QMIX Unanimous Ensemble | Entropy Threshold | total_energy_consumption | -0.7225 | [-0.8316, -0.6134] | 5.032e-10 |
| QMIX Unanimous Ensemble | Entropy Threshold | mean_aoi | 0.3641 | [0.1768, 0.5514] | 9.006e-03 |
| QMIX Unanimous Ensemble | Entropy Threshold | redundant_sampling | -3.5500 | [-4.7308, -2.3692] | 9.221e-05 |
| QMIX Unanimous Ensemble | Entropy Threshold | network_coverage | 0.0010 | [-0.0017, 0.0037] | 1.000e+00 |

## Decision

Promote the highest-reward candidate only when its paired reward 95% CI versus the three-replica Extended mean is above zero, recall advantage is at least -0.005, energy advantage at least -0.25, and redundancy advantage at least -2.0.

Promoted candidate: `QMIX Unanimous Ensemble`.
