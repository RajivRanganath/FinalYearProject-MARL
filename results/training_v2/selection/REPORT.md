# Extended QMIX Critical Evaluation

Split: `selection`. Scenario: `volatile`. Regime: `coordinated`. Paired environment seeds: `211`--`230` (20 seeds).

Extended QMIX changes only training duration relative to Improved QMIX. The reward and environment are unchanged. Learned-policy means average three training replicas per environment seed before the displayed t intervals.

| Policy | Reward mean [95% CI] | Recall mean [95% CI] | Energy mean [95% CI] | AoI mean [95% CI] | Redundant samples | Channel blocks |
|---|---:|---:|---:|---:|---:|---:|
| Entropy Threshold | 159.72 [141.08, 178.37] | 0.599 [0.577, 0.620] | 10.92 [10.46, 11.39] | 5.95 [5.36, 6.55] | 48.55 | 36.80 |
| Battery + Entropy | 150.21 [130.90, 169.53] | 0.588 [0.565, 0.610] | 10.73 [10.28, 11.19] | 6.27 [5.66, 6.89] | 47.55 | 35.65 |
| Published QMIX | 15.10 [-5.14, 35.34] | 0.451 [0.427, 0.475] | 13.83 [13.55, 14.12] | 9.00 [8.67, 9.34] | 61.87 | 51.78 |
| Improved QMIX | 123.92 [105.44, 142.40] | 0.566 [0.545, 0.588] | 12.71 [12.29, 13.12] | 5.71 [5.17, 6.25] | 57.72 | 41.37 |
| Extended QMIX | 153.87 [135.86, 171.88] | 0.596 [0.576, 0.617] | 11.90 [11.46, 12.33] | 5.56 [5.04, 6.09] | 53.22 | 40.00 |

## Training-seed variability

| Policy | Training seed | Reward mean | Recall mean | Energy mean | AoI mean |
|---|---:|---:|---:|---:|---:|
| Extended QMIX | 101 | 151.85 | 0.594 | 11.91 | 5.67 |
| Extended QMIX | 102 | 155.15 | 0.598 | 11.97 | 5.44 |
| Extended QMIX | 103 | 154.61 | 0.597 | 11.80 | 5.59 |
| Improved QMIX | 101 | 128.16 | 0.571 | 12.75 | 5.67 |
| Improved QMIX | 102 | 117.45 | 0.559 | 12.50 | 6.11 |
| Improved QMIX | 103 | 126.16 | 0.569 | 12.88 | 5.35 |

## Paired comparisons

Positive advantage means Extended QMIX is better in the engineering direction. The environment-seed interval averages replicas; the two-way bootstrap resamples both training and environment seeds. Holm adjustment covers all comparisons below.

| Comparator | Metric | Advantage | Environment CI | Two-way bootstrap CI | Holm p |
|---|---|---:|---:|---:|---:|
| Improved QMIX | raw_episode_reward | 29.9470 | [25.2486, 34.6454] | [22.5322, 38.3640] | 4.246e-10 |
| Improved QMIX | event_recall | 0.0303 | [0.0255, 0.0351] | [0.0217, 0.0398] | 5.101e-10 |
| Improved QMIX | total_energy_consumption | 0.8117 | [0.7281, 0.8952] | [0.5349, 1.0717] | 2.597e-13 |
| Improved QMIX | mean_aoi | 0.1421 | [-0.0255, 0.3097] | [-0.2764, 0.6443] | 2.760e-01 |
| Improved QMIX | redundant_sampling | 4.5000 | [2.7332, 6.2668] | [2.4667, 6.5500] | 2.290e-04 |
| Improved QMIX | network_coverage | 0.0082 | [0.0049, 0.0114] | [0.0042, 0.0123] | 2.290e-04 |
| Entropy Threshold | raw_episode_reward | -5.8517 | [-9.3576, -2.3458] | [-9.6412, -2.2664] | 9.724e-03 |
| Entropy Threshold | event_recall | -0.0021 | [-0.0063, 0.0022] | [-0.0066, 0.0022] | 6.497e-01 |
| Entropy Threshold | total_energy_consumption | -0.9717 | [-1.0574, -0.8860] | [-1.0859, -0.8558] | 1.675e-14 |
| Entropy Threshold | mean_aoi | 0.3862 | [0.2372, 0.5353] | [0.2203, 0.5910] | 2.185e-04 |
| Entropy Threshold | redundant_sampling | -4.6667 | [-5.7451, -3.5882] | [-5.8667, -3.4667] | 2.025e-07 |
| Entropy Threshold | network_coverage | -0.0003 | [-0.0023, 0.0018] | [-0.0021, 0.0019] | 7.858e-01 |

## Promotion decision

Rule: Promote only if mean reward improves and every one of the three paired training replicas improves on predeclared selection seeds 211--230.

Reward advantages by training seed: `{'101': 23.688416666666683, '102': 37.70451666666665, '103': 28.44808333333333}`. Promote Extended QMIX: **True**.
