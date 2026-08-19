# Extended QMIX Critical Evaluation

Split: `final`. Scenario: `volatile`. Regime: `coordinated`. Paired environment seeds: `3001`--`3030` (30 seeds).

Extended QMIX changes only training duration relative to Improved QMIX. The reward and environment are unchanged. Learned-policy means average three training replicas per environment seed before the displayed t intervals.

| Policy | Reward mean [95% CI] | Recall mean [95% CI] | Energy mean [95% CI] | AoI mean [95% CI] | Redundant samples | Channel blocks |
|---|---:|---:|---:|---:|---:|---:|
| Entropy Threshold | 156.32 [139.07, 173.58] | 0.609 [0.589, 0.629] | 10.74 [10.30, 11.19] | 5.62 [5.28, 5.96] | 46.90 | 34.40 |
| Battery + Entropy | 147.65 [130.36, 164.94] | 0.598 [0.578, 0.619] | 10.58 [10.15, 11.01] | 5.78 [5.45, 6.12] | 45.63 | 33.47 |
| Published QMIX | 27.59 [8.99, 46.19] | 0.473 [0.448, 0.498] | 13.95 [13.66, 14.25] | 8.61 [8.38, 8.85] | 61.74 | 53.19 |
| Improved QMIX | 119.09 [102.71, 135.48] | 0.575 [0.554, 0.595] | 12.69 [12.32, 13.07] | 5.16 [4.81, 5.50] | 58.27 | 42.08 |
| Extended QMIX | 146.76 [129.39, 164.13] | 0.602 [0.583, 0.622] | 11.72 [11.31, 12.13] | 5.07 [4.76, 5.38] | 52.60 | 39.66 |

## Training-seed variability

| Policy | Training seed | Reward mean | Recall mean | Energy mean | AoI mean |
|---|---:|---:|---:|---:|---:|
| Extended QMIX | 101 | 146.15 | 0.602 | 11.76 | 5.07 |
| Extended QMIX | 102 | 147.65 | 0.604 | 11.78 | 5.00 |
| Extended QMIX | 103 | 146.48 | 0.602 | 11.64 | 5.13 |
| Improved QMIX | 101 | 121.47 | 0.577 | 12.69 | 5.12 |
| Improved QMIX | 102 | 117.87 | 0.572 | 12.49 | 5.54 |
| Improved QMIX | 103 | 117.95 | 0.574 | 12.91 | 4.81 |

## Paired comparisons

Positive advantage means Extended QMIX is better in the engineering direction. The environment-seed interval averages replicas; the two-way bootstrap resamples both training and environment seeds. Holm adjustment covers all comparisons below.

| Comparator | Metric | Advantage | Environment CI | Two-way bootstrap CI | Holm p |
|---|---|---:|---:|---:|---:|
| Improved QMIX | raw_episode_reward | 27.6658 | [23.9034, 31.4283] | [23.2727, 32.3083] | 3.142e-14 |
| Improved QMIX | event_recall | 0.0277 | [0.0236, 0.0319] | [0.0226, 0.0331] | 2.932e-13 |
| Improved QMIX | total_energy_consumption | 0.9711 | [0.8482, 1.0940] | [0.7006, 1.2683] | 5.362e-15 |
| Improved QMIX | mean_aoi | 0.0898 | [-0.0788, 0.2584] | [-0.3238, 0.5390] | 5.700e-01 |
| Improved QMIX | redundant_sampling | 5.6667 | [4.3289, 7.0044] | [3.5333, 7.8000] | 9.233e-09 |
| Improved QMIX | network_coverage | 0.0069 | [0.0049, 0.0090] | [0.0042, 0.0098] | 7.770e-07 |
| Entropy Threshold | raw_episode_reward | -9.5621 | [-12.5019, -6.6223] | [-12.4452, -6.6127] | 1.083e-06 |
| Entropy Threshold | event_recall | -0.0067 | [-0.0100, -0.0034] | [-0.0101, -0.0034] | 8.560e-04 |
| Entropy Threshold | total_energy_consumption | -0.9794 | [-1.0698, -0.8891] | [-1.0833, -0.8661] | 1.199e-18 |
| Entropy Threshold | mean_aoi | 0.5520 | [0.4228, 0.6812] | [0.4127, 0.6968] | 9.003e-09 |
| Entropy Threshold | redundant_sampling | -5.7000 | [-6.6102, -4.7898] | [-6.6222, -4.7333] | 1.464e-12 |
| Entropy Threshold | network_coverage | -0.0008 | [-0.0025, 0.0009] | [-0.0026, 0.0008] | 5.700e-01 |

## Interpretation boundary

This final split was evaluated after the profile and model hashes were frozen. No further tuning on these seeds is valid. Three training seeds remain a small sample for algorithm-level generalisation, so training-seed conclusions are cautious.
