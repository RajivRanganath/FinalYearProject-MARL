# Retrained QMIX Ablation Study

Scenario: `volatile`. Regime: `coordinated`. Training seeds: [101, 102, 103]. Paired held-out environment seeds: 30 (`1001`-`1030`).

Each variant was retrained from scratch after removing exactly the named component. Every resulting policy was evaluated in the same full coordinated environment and under the same full reward, so raw rewards are comparable. Training-seed replicas were averaged within each environment seed before confidence intervals and paired tests were calculated.

| Variant | Reward mean [95% CI] | Recall mean [95% CI] | Energy mean [95% CI] | Mean AoI [95% CI] |
|---|---:|---:|---:|---:|
| Full QMIX | 24.38 [9.06, 39.70] | 0.467 [0.446, 0.488] | 13.92 [13.64, 14.19] | 9.08 [8.82, 9.33] |
| No agent ID | 86.51 [71.84, 101.17] | 0.534 [0.514, 0.555] | 12.63 [12.29, 12.97] | 6.86 [6.53, 7.19] |
| No AoI term | 28.32 [13.22, 43.42] | 0.471 [0.450, 0.491] | 13.74 [13.47, 14.01] | 8.92 [8.66, 9.18] |
| No coordination constraint | -190.50 [-206.18, -174.81] | 0.320 [0.301, 0.339] | 19.31 [19.23, 19.40] | 9.01 [8.86, 9.17] |
| No energy term | -14.81 [-30.34, 0.73] | 0.429 [0.407, 0.450] | 14.84 [14.62, 15.06] | 10.03 [9.82, 10.24] |
| No neighbor signal | 64.85 [50.81, 78.89] | 0.513 [0.492, 0.535] | 12.55 [12.19, 12.91] | 7.78 [7.42, 8.15] |
| No redundancy penalty | -93.75 [-109.32, -78.18] | 0.372 [0.352, 0.393] | 17.47 [17.35, 17.60] | 10.21 [10.00, 10.42] |

Paired comparisons use a two-sided one-sample t-test on the 30 full-minus-ablated seed differences. The accompanying CSV reports the statistic, p-value, and paired Cohen's dz; no causal importance claim should be made from a point estimate or p-value alone.
