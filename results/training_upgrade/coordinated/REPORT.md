# Training Upgrade Evaluation

Scenario: `volatile`. Regime: `coordinated`. Fresh paired holdout: `2001`--`2030` (30 seeds).

The upgraded profile was selected using validation seeds 201--210. The original 1001--1030 test set was not reused for this claim.

| Policy | Reward mean [95% CI] | Recall mean [95% CI] | Energy mean [95% CI] | AoI mean [95% CI] |
|---|---:|---:|---:|---:|
| Entropy Threshold | 168.22 [151.37, 185.07] | 0.607 [0.591, 0.623] | 11.20 [10.66, 11.74] | 5.56 [5.25, 5.87] |
| Published QMIX | 21.94 [6.64, 37.24] | 0.456 [0.438, 0.475] | 14.15 [13.84, 14.46] | 8.90 [8.62, 9.18] |
| Upgraded QMIX | 123.18 [106.57, 139.80] | 0.567 [0.549, 0.585] | 12.98 [12.57, 13.40] | 5.33 [4.98, 5.68] |

Positive paired advantage means the upgraded policy is better in the engineering direction; energy and AoI signs are reversed because lower is better. Tests are exploratory and unadjusted for multiple comparisons.

| Comparator | Metric | Upgraded advantage [95% CI] | t | p | Cohen dz |
|---|---|---:|---:|---:|---:|
| Published QMIX | raw_episode_reward | 101.2461 [93.3575, 109.1348] | 26.249 | 9.249e-22 | 4.792 |
| Published QMIX | event_recall | 0.1103 [0.1003, 0.1204] | 22.485 | 6.714e-20 | 4.105 |
| Published QMIX | total_energy_consumption | 1.1678 [0.9884, 1.3471] | 13.317 | 6.931e-14 | 2.431 |
| Published QMIX | mean_aoi | 3.5690 [3.2824, 3.8556] | 25.472 | 2.135e-21 | 4.651 |
| Entropy Threshold | raw_episode_reward | -45.0366 [-51.0810, -38.9923] | -15.239 | 2.232e-15 | -2.782 |
| Entropy Threshold | event_recall | -0.0402 [-0.0470, -0.0334] | -12.123 | 7.074e-13 | -2.213 |
| Entropy Threshold | total_energy_consumption | -1.7856 [-1.9491, -1.6220] | -22.326 | 8.161e-20 | -4.076 |
| Entropy Threshold | mean_aoi | 0.2299 [-0.0391, 0.4988] | 1.748 | 9.104e-02 | 0.319 |
