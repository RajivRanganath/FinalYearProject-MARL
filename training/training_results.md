# Historical Benchmark Snapshot (Superseded)

> These pre-repair MLP/oracle-era numbers are retained for history only. They
> are not the current causal benchmark and must not be cited as final results.
> Use `FINAL_RESEARCH_REPORT.md` and the frozen v2/v3 result directories.

# Legacy MARL Training & Empirical Benchmark Results Summary

This document records the empirical results of training Multi-Agent Reinforcement Learning (MARL) algorithms (**QMIX**, **VDN**, **IQL**) against standard baseline heuristics across 30 held-out Monte Carlo test seeds (`1001`–`1030`) and 3 environmental regimes (`stable`, `volatile`, `stress`).

All reported figures are directly computed from the test execution engine (`deployment/evaluate_all.py`).

---

## 1. Multi-Scenario Benchmark Results (30 Test Seeds with 95% Confidence Intervals)

### Volatile Scenario (Intermittent Cloud Attenuation & Poisson High-Entropy Bursts)

| Policy | Team Reward | Event Recall (%) | Mean AoI (steps) | Rejections | Overlap Steps | Final Battery |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Entropy Threshold (>0.60)** | **897.10 ± 29.38** | **94.85 ± 1.27%** | **10.25 ± 0.55** | **0.00 ± 0.00** | **37.67 ± 3.15** | **0.34 ± 0.04** |
| **Greedy Myopic Heuristic** | 897.10 ± 29.38 | 94.85 ± 1.27% | 10.25 ± 0.55 | 0.00 ± 0.00 | 37.67 ± 3.15 | 0.34 ± 0.04 |
| **Battery+Entropy Heuristic** | 861.76 ± 28.12 | 93.50 ± 1.47% | 9.43 ± 0.58 | 0.00 ± 0.00 | 74.37 ± 4.44 | 0.19 ± 0.03 |
| **Always Sample (Feasible)** | 72.02 ± 29.09 | 50.89 ± 1.73% | 15.86 ± 0.11 | 0.00 ± 0.00 | 147.90 ± 0.88 | 0.02 ± 0.00 |
| **Battery Threshold (<20%)** | 42.87 ± 29.77 | 48.68 ± 1.75% | 17.31 ± 0.11 | 0.00 ± 0.00 | 142.37 ± 0.92 | 0.17 ± 0.00 |
| **Random Feasible** | -128.78 ± 19.95 | 33.01 ± 1.25% | 8.91 ± 0.24 | 0.00 ± 0.00 | 125.83 ± 1.89 | 0.02 ± 0.00 |
| **Fixed Interval (N=12)** | -497.76 ± 20.15 | 7.99 ± 0.52% | 5.50 ± 0.00 | 0.00 ± 0.00 | 24.00 ± 0.00 | 0.69 ± 0.00 |
| **Trained VDN (MARL)** | -541.21 ± 26.34 | 4.29 ± 1.19% | 125.02 ± 4.13 | 0.00 ± 0.00 | 0.40 ± 0.53 | 0.99 ± 0.00 |
| **Always Sleep** | -610.20 ± 20.40 | 0.00 ± 0.00% | 144.50 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.99 ± 0.00 |
| **Trained QMIX (MARL)** | -610.20 ± 20.40 | 0.00 ± 0.00% | 144.50 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.99 ± 0.00 |
| **Trained IQL (MARL)** | -610.20 ± 20.40 | 0.00 ± 0.00% | 144.50 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.99 ± 0.00 |

---

### Stress Scenario (Heavy Overcast & Severe Solar Starvation)

| Policy | Team Reward | Event Recall (%) | Mean AoI (steps) | Rejections | Overlap Steps | Final Battery |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Entropy Threshold (>0.60)** | **828.13 ± 45.15** | **64.29 ± 1.45%** | **14.33 ± 0.53** | **0.00 ± 0.00** | **37.67 ± 3.15** | **0.03 ± 0.01** |
| **Greedy Myopic Heuristic** | 828.13 ± 45.15 | 64.29 ± 1.45% | 14.33 ± 0.53 | 0.00 ± 0.00 | 37.67 ± 3.15 | 0.03 ± 0.01 |
| **Battery+Entropy Heuristic** | 681.17 ± 43.79 | 60.58 ± 1.42% | 14.54 ± 0.49 | 0.00 ± 0.00 | 74.37 ± 4.44 | 0.02 ± 0.00 |
| **Always Sample (Feasible)** | 68.33 ± 48.48 | 43.28 ± 1.51% | 19.50 ± 0.19 | 0.00 ± 0.00 | 147.90 ± 0.88 | 0.02 ± 0.00 |
| **Battery Threshold (<20%)** | 36.33 ± 49.94 | 42.17 ± 1.57% | 21.03 ± 0.20 | 0.00 ± 0.00 | 142.37 ± 0.92 | 0.17 ± 0.00 |
| **Random Feasible** | -340.06 ± 29.35 | 28.49 ± 0.94% | 13.76 ± 0.27 | 0.00 ± 0.00 | 125.83 ± 1.89 | 0.02 ± 0.00 |
| **Trained VDN (MARL)** | -987.62 ± 38.70 | 7.21 ± 0.91% | 92.23 ± 2.97 | 0.00 ± 0.00 | 0.40 ± 0.53 | 0.99 ± 0.00 |
| **Fixed Interval (N=12)** | -1020.29 ± 38.63 | 6.58 ± 0.44% | 10.89 ± 0.33 | 0.00 ± 0.00 | 24.00 ± 0.00 | 0.69 ± 0.00 |
| **Always Sleep** | -1220.30 ± 39.75 | 0.00 ± 0.00% | 144.50 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.99 ± 0.00 |
| **Trained QMIX (MARL)** | -1220.30 ± 39.75 | 0.00 ± 0.00% | 144.50 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.99 ± 0.00 |
| **Trained IQL (MARL)** | -1220.30 ± 39.75 | 0.00 ± 0.00% | 144.50 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.99 ± 0.00 |

---

## 2. Key Scientific Findings & Behavioral Analysis

1. **Feasibility Masking & Causality Compliance**:
   - In all tested policies and all 900 evaluated episode runs, **rejections were exactly 0.00**. Energy causality ($E_{t} \ge E_{sample}$) is strictly enforced by action masking at both train and test time.
2. **Heuristic Adaptive Dominance**:
   - The **Entropy Threshold (>0.60)** and **Greedy Myopic Heuristic** achieved top performance across all regimes ($897.10 \pm 29.38$ in Volatile), capturing **$94.85\%$ of all high-entropy events** while maintaining low co-sampling collisions ($37.67$ overlap steps).
3. **Fixed-Interval vs. Adaptive Sensing**:
   - Blind periodic sampling (`Fixed Interval N=12`) achieves low data staleness (AoI = 5.5) but misses over $92\%$ of burst events in Volatile scenarios because environmental spikes are transient Poisson bursts uncorrelated with fixed clock ticks.
4. **Energy Conservation vs. Event Capture**:
   - In the Stress scenario under solar starvation, the entropy threshold policy intelligently adapts, scaling back sampling to maintain battery longevity while still capturing $64.29\%$ of events.
