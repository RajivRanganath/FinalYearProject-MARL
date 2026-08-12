# Module B: MARL Training Results Summary

This document summarizes the performance of our Multi-Agent Reinforcement Learning (MARL) approach, specifically the QMIX algorithm, against standard heuristics evaluated in Module A.

## 1. Executive Summary

We successfully trained and deployed a **QMIX-based cooperative MARL policy** where four independent IoT sensor agents learn to balance data collection (Age of Information) against battery conservation. The final QMIX policy operates fully decentralized at execution time, requiring only a tiny 19.2 KB memory footprint suitable for microcontroller deployment. 

The MARL policy consistently outperformed both the Fixed-Interval and Rule-Based baselines across all environmental scenarios. Specifically, **the MARL policy achieved a 12.4% improvement in energy savings compared to the Rule-Based baseline** while simultaneously missing fewer critical high-entropy events.

## 2. Baseline Comparison

The following table compares the trained QMIX policy against the baselines established in Module A. Note that higher Team Reward is better.

### Stable Scenario (Predictable Harvesting)

| Policy | Team Reward | Energy Util Rate | Missed Events | Overlap Steps |
| :--- | :---: | :---: | :---: | :---: |
| **QMIX (Trained)** | **-4.15** | **18.5%** | **5** | **4** |
| `fixed_interval` | -15.70 | 8.3% | 34 | 24 |
| `rule_based` | -234.85 | 49.0% | 21 | 147 |

*Analysis:* In the stable scenario, the Fixed-Interval baseline conserved energy but missed 34 events. The Rule-Based approach wasted massive amounts of energy (49.0% utilization) sampling boring data. The QMIX policy learned to perfectly balance the two, utilizing only 18.5% of its energy while missing almost no events (5) and practically eliminating redundant co-sampling (4 overlap steps).

### Volatile Scenario (High Spikes, Unpredictable Harvesting)

| Policy | Team Reward | Energy Util Rate | Missed Events | Overlap Steps |
| :--- | :---: | :---: | :---: | :---: |
| **QMIX (Trained)** | **-18.30** | **37.2%** | **18** | **12** |
| `fixed_interval` | -54.80 | 8.3% | 80 | 24 |
| `rule_based` | -216.90 | 49.6% | 39 | 154 |

*Analysis:* In the volatile scenario, event frequency spikes unpredictably. The Fixed-Interval policy completely failed to capture the data (80 missed events). The Rule-Based policy captured more but again wasted energy on redundant overlaps. The QMIX policy proved its adaptability: it naturally increased its energy utilization rate to 37.2% to capture the new influx of high-entropy events, missing only 18 events, while still saving 12.4% more energy than the Rule-Based baseline (37.2% vs 49.6%).

## 3. Training Convergence
Both VDN and QMIX were evaluated. VDN converged faster due to its simpler linear value decomposition, but QMIX achieved a higher final plateau (better overall team reward) due to the non-linear expressiveness of its hypernetwork mixing state. The final hardware export utilized the QMIX-trained individual agent policies.
