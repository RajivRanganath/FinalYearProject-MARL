# Refined QMIX Weight-Update Evaluation

Split: `selection`. Refined QMIX is a low-learning-rate continuation of each Extended checkpoint; environment and reward are unchanged.

| Policy | Reward | Recall | Energy | AoI | Redundancy |
|---|---:|---:|---:|---:|---:|
| Extended QMIX | 145.29 | 0.603 | 11.77 | 5.37 | 51.83 |
| Refined QMIX | 146.73 | 0.605 | 11.66 | 5.43 | 51.33 |
| Entropy Threshold | 154.35 | 0.608 | 10.80 | 5.89 | 46.65 |
| Battery + Entropy | 146.89 | 0.600 | 10.66 | 6.06 | 46.00 |

## Training-replica reward means

| Policy | Seed | Reward |
|---|---:|---:|
| Extended QMIX | 101 | 143.29 |
| Extended QMIX | 102 | 146.27 |
| Extended QMIX | 103 | 146.32 |
| Refined QMIX | 101 | 147.09 |
| Refined QMIX | 102 | 145.96 |
| Refined QMIX | 103 | 147.14 |

## Paired comparisons

| Comparator | Metric | Advantage | Environment CI | Two-way bootstrap CI | Holm p |
|---|---|---:|---:|---:|---:|
| Extended QMIX | raw_episode_reward | 1.4363 | [0.4606, 2.4121] | [-0.7400, 3.8998] | 4.306e-02 |
| Extended QMIX | event_recall | 0.0014 | [0.0001, 0.0028] | [-0.0010, 0.0044] | 1.665e-01 |
| Extended QMIX | total_energy_consumption | 0.1025 | [0.0679, 0.1371] | [0.0592, 0.1475] | 5.955e-05 |
| Extended QMIX | mean_aoi | -0.0664 | [-0.1231, -0.0096] | [-0.1278, 0.0002] | 1.211e-01 |
| Extended QMIX | redundant_sampling | 0.5000 | [0.0967, 0.9033] | [-0.1500, 1.2000] | 1.067e-01 |
| Extended QMIX | network_coverage | 0.0004 | [-0.0002, 0.0010] | [-0.0005, 0.0011] | 4.934e-01 |
| Entropy Threshold | raw_episode_reward | -7.6228 | [-11.9257, -3.3199] | [-11.6971, -3.5112] | 1.194e-02 |
| Entropy Threshold | event_recall | -0.0038 | [-0.0094, 0.0017] | [-0.0091, 0.0014] | 4.934e-01 |
| Entropy Threshold | total_energy_consumption | -0.8625 | [-0.9540, -0.7710] | [-0.9725, -0.7608] | 4.850e-13 |
| Entropy Threshold | mean_aoi | 0.4529 | [0.2650, 0.6407] | [0.2787, 0.6461] | 6.465e-04 |
| Entropy Threshold | redundant_sampling | -4.6833 | [-5.8633, -3.5034] | [-5.9167, -3.4500] | 1.048e-06 |
| Entropy Threshold | network_coverage | -0.0008 | [-0.0032, 0.0016] | [-0.0030, 0.0015] | 5.116e-01 |

## Promotion boundary

Promote only if refined mean paired reward has a 95% CI above zero, all three matched training replicas improve mean reward, mean recall advantage is at least -0.002, mean energy advantage is at least -0.10, and mean redundancy advantage is at least -1.0 on seeds 251--270.

Promote Refined QMIX: **False**.
