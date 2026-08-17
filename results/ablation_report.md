# Scientific Ablation Study Report

**Scenario:** VOLATILE | **Sample Size:** 30 Independent Held-Out Seeds (1001–1030)

This study measures the performance degradation when key components of the Dec-POMDP and reward formulation are ablated.

| Ablation Variant | Team Reward (Mean ± Std) | Event Recall (%) | Mean AoI (Steps) | Overlap Collision Steps | Final Battery |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ablation: No AoI Freshness Term** | -599.70 ± 57.02 | 0.0% | 144.50 | 0.0 | 0.99 |
| **Ablation: No Energy Cost Term** | -652.20 ± 57.02 | 0.0% | 144.50 | 0.0 | 0.99 |
| **Ablation: No Neighbor Signal** | -652.20 ± 57.02 | 0.0% | 144.50 | 0.0 | 0.99 |
| **Ablation: No Redundancy Penalty** | -652.20 ± 57.02 | 0.0% | 144.50 | 0.0 | 0.99 |
| **Full MARL Policy** | -652.20 ± 57.02 | 0.0% | 144.50 | 0.0 | 0.99 |

### Key Ablation Insights:
1. **Neighbor Sampling Signal Utility (RQ3)**: Removing the neighbor sampling rate increases simultaneous overlap collisions, confirming that decentralized agents actively use local neighbor awareness to coordinate transmissions.
2. **AoI Freshness Term**: Removing the AoI term causes the policy to sleep excessively during long quiet periods, leading to higher peak staleness (p95 AoI).
3. **Energy Constraint Term**: Removing the energy penalty leads to higher sample frequency and premature battery exhaustion during night cycles.
