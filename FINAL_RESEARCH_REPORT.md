# Final Research Report: When Does Cooperative MARL Help Energy-Harvesting Sensor Networks?

## Executive finding

No overall engineering advantage from QMIX over the strongest causal adaptive heuristic was observed in either evaluated regime. The learned policies were repaired from an Always-Sleep collapse and all selected checkpoints demonstrably chose both actions, but the causal event-proxy threshold remained better on reward, event recall, energy use, and AoI. In the independent regime, the best learned policy was VDN (reward 9.50) versus the threshold policy (190.28). In the coordinated regime, the best learned policy was QMIX (24.38) versus the threshold policy (159.03). The coordinated regime therefore created genuine inter-agent dependence, but not enough complexity for value decomposition to beat a well-aligned local proxy rule at the original training budget and network scale. A later split-locked continuation substantially improved QMIX. A separately locked unanimous-vote deployment ensemble then closed the reward gap to a statistically unresolved 2.27 units and had a slightly higher but statistically unresolved recall point estimate, while still using more energy and producing more redundant samples than the threshold.

## Post-report validation-driven training upgrade

After the original benchmark was frozen, a separate training-only upgrade fixed terminal checkpoint loss, increased replay updates for 288-step episodes, and used the homogeneous shared-policy configuration supported by the earlier ablations. No environment physics, reward weight, baseline, or original test result was changed. The three selected validation rewards were seed 101: 127.12, seed 102: 123.35, seed 103: 121.57. Because seeds 1001--1030 had already informed the ablation analysis, the upgraded claim uses the predeclared fresh holdout 2001--2030.

| Policy | Reward [95% CI] | Recall [95% CI] | Energy [95% CI] | Mean AoI [95% CI] |
|---|---:|---:|---:|---:|
| Published QMIX | 21.94 [6.64, 37.24] | 0.456 [0.438, 0.475] | 14.15 [13.84, 14.46] | 8.90 [8.62, 9.18] |
| Upgraded QMIX | 123.18 [106.57, 139.80] | 0.567 [0.549, 0.585] | 12.98 [12.57, 13.40] | 5.33 [4.98, 5.68] |
| Entropy Threshold | 168.22 [151.37, 185.07] | 0.607 [0.591, 0.623] | 11.20 [10.66, 11.74] | 5.56 [5.25, 5.87] |

The upgraded policy gains 101.25 paired reward units over published QMIX (95% CI 93.36 to 109.13; p=9.25e-22, dz=4.79). It nevertheless remains behind the causal threshold by 45.04 reward units (p=2.23e-15). The upgrade therefore strengthens QMIX substantially without changing the central negative-result conclusion. Full paired confidence intervals and effect sizes are in [the training-upgrade report](results/training_upgrade/coordinated/REPORT.md).

## Split-locked extended-training result

A further one-factor experiment restored agent identity while keeping the 180k-step configuration fixed. It was screened out after the first training replica: its seed-101 validation reward was 121.12, versus 127.12 for the matched no-identity model. The retained hypothesis changed only training duration: each of the three no-identity QMIX replicas was warm-started from its validation-selected 180k checkpoint and trained to a 360k horizon. Replay, RNG progress, learner counters, and target history were not restored. Validation selected seed 101: step 360288, reward 155.89, seed 102: step 360288, reward 156.88, seed 103: step 289440, reward 159.13; seed 103 was selected before the terminal checkpoint rather than automatically taking the last model.

The predeclared promotion gate used only selection seeds 211--230 and required both a positive mean reward difference and improvement in every matched training replica. It passed with replica advantages seed 101: 23.69, seed 102: 37.70, seed 103: 28.45 (mean 29.95). Only then was the untouched final split 3001--3030 evaluated once.

| Policy | Reward [95% CI] | Recall | Energy | Mean AoI | Redundant samples |
|---|---:|---:|---:|---:|---:|
| Published QMIX | 27.59 [8.99, 46.19] | 0.473 | 13.95 | 8.61 | 61.74 |
| Improved QMIX | 119.09 [102.71, 135.48] | 0.575 | 12.69 | 5.16 | 58.27 |
| Extended QMIX | 146.76 [129.39, 164.13] | 0.602 | 11.72 | 5.07 | 52.60 |
| Entropy Threshold | 156.32 [139.07, 173.58] | 0.609 | 10.74 | 5.62 | 46.90 |

Extended QMIX improved over Improved QMIX by 27.67 paired reward units; its two-way bootstrap 95% interval was [23.27, 32.31] and Holm-adjusted p=3.14e-14. Final reward was stable across the three training replicas (seed 101: 146.15, seed 102: 147.65, seed 103: 146.48). The stronger training result still did not beat the threshold: Extended QMIX was worse by 9.56 reward units (Holm p=1.08e-06), 0.007 recall, and 0.98 energy units, while improving mean AoI by 0.55. The final environment seeds are now consumed and frozen; they must not be used for further selection or tuning. See [the frozen final report](results/training_v2/final/REPORT.md).

The final manifest records the exact 30 seeds plus SHA-256 fingerprints for 12 primary model files and six listed evaluation/source files. A later dependency-closure audit found that those six hashes omitted behavior-changing transitive files, including the energy model and PettingZoo wrapper. The frozen numbers remain internally reproducible from their raw CSVs, but the fingerprint claim is partial rather than complete. It records Git SHA `2d920fa26d1781341a1f260b7d8309fadb134d87` and the worktree-dirty disclosure.

## Split-locked deployment ensemble

A final bounded iteration left all trained weights, environment dynamics, and reward terms unchanged. It predeclared two scale-independent deployment rules over the three Extended QMIX replicas: majority voting and unanimous voting for SAMPLE. Selection seeds 231--250 rejected majority voting because its reward interval crossed zero. Unanimous voting passed every promotion gate, gaining 4.19 reward units with selection CI [2.58, 5.81], while also improving recall, energy, and redundancy. Only that rule was evaluated on the untouched final seeds 4001--4030.

The historical promotion predicate reported but did not require the candidate-family Holm-adjusted reward p-value. The selected unanimous rule nevertheless has `p_holm = 6.20e-05` and passes the corrected future guard, so this omission does not change the frozen winner or final result.

| Policy | Reward | Recall | Energy | Mean AoI | Redundant samples |
|---|---:|---:|---:|---:|---:|
| Extended QMIX Replica Mean | 153.38 | 0.608 | 11.78 | 5.10 | 53.84 |
| QMIX Unanimous Ensemble | 157.43 | 0.612 | 11.55 | 5.21 | 52.70 |
| Entropy Threshold | 159.70 | 0.611 | 10.82 | 5.64 | 48.30 |
| Battery + Entropy | 150.45 | 0.600 | 10.62 | 5.82 | 47.13 |

On the final split, unanimous voting improved over the Extended replica mean by 4.05 reward units (95% CI [2.92, 5.18], Holm p=3.93e-07). It also improved recall, energy, redundancy, and coverage, but worsened AoI by 0.11. Relative to the threshold, the remaining reward difference was -2.27 (95% CI [-4.92, 0.39], Holm p=0.182); this is neither evidence of a reward win nor an equivalence test. Recall differed by only 0.001 with an interval crossing zero. The ensemble retained a clear 0.73-unit energy disadvantage and 4.40 additional redundant samples, while its AoI was 0.42 lower.

The ensemble requires three recurrent-model evaluations per decision, approximately tripling model storage and inference work relative to one QMIX replica. Its final intervals vary environment seeds but condition on this one selected three-model set, so they do not measure variability across independently retrained ensembles. It is therefore a stronger simulation/deployment policy but not automatically a better TinyML choice. The final seeds are consumed and frozen. See [the v3 final report](results/training_v3/final/REPORT.md) and [artifact audit](results/training_v3/audit.json).

## Strict weight-update attempt: invalidated protocol and not promoted

Refined QMIX was intended to be a `1e-4` continuation of all three Extended checkpoints to a 540k-step horizon. A later artifact-level audit found that `Optimizer.load_state_dict` restored the source checkpoint parameter-group rate after the new optimizer was constructed. Every one of the 37 Refined `opt.th` files, including all three selected checkpoints, records `3e-4`. The run therefore did **not** test the declared low-learning-rate hypothesis. A separate resume audit did correctly resolve seed 103 to its validation-selected source step 289440.

| Policy | Reward | Recall | Energy | Mean AoI | Redundancy |
|---|---:|---:|---:|---:|---:|
| Extended QMIX | 145.29 | 0.603 | 11.77 | 5.37 | 51.83 |
| Refined QMIX | 146.73 | 0.605 | 11.66 | 5.43 | 51.33 |
| Entropy Threshold | 154.35 | 0.608 | 10.80 | 5.89 | 46.65 |

The historical same-rate selection calculation improved by 1.44 and the environment-seed CI lower bound was 0.46. However, the two-way bootstrap lower bound was -0.74, and matched replica advantages were seed 101: +3.80, seed 102: -0.31, seed 103: +0.82. Because seed 102 regressed and the bootstrap crossed zero, the predeclared all-replica gate rejected the candidate even before the LR defect was discovered. These numbers now describe only a rejected same-rate warm-start and cannot answer the low-LR question. Locked seeds 5001--5030 were not evaluated. See [the invalidation record](results/training_v4/INVALIDATED.json); the original [selection report](results/training_v4/selection/REPORT.md) is retained as a frozen historical artifact whose low-LR sentence is superseded.

## Source drift after the results were produced

Two source repairs postdate every run and every evaluation in this repository, so the committed source does not bit-exactly regenerate the committed numbers. The larger repair changes SAMPLE feasibility from `battery >= sample_energy_cost` to `battery >= sample_energy_cost + sleep_energy_cost + proxy_monitor_energy_cost`, because the previous rule admitted a sample without accounting for the unavoidable same-step sleep and proxy-monitor draw. It governs the action mask for every agent in every run.

The gap is measured rather than assumed. The probe replays the real environment under both rules on the same 30 held-out seeds, using the policy that saturates the battery floor and therefore bounds how far the rules can diverge.

| Regime | Mean reward delta | Max abs delta | Seeds differing | Agent-steps in disagreement band |
|---|---:|---:|---:|---:|
| independent | +0.1833 | 1.0000 | 10/30 | 11/34560 (0.032%) |
| coordinated | -0.0666 | 4.1980 | 7/30 | 7/34560 (0.020%) |

The two rules can only disagree while the battery sits inside a band 0.017% of capacity wide. Against a headline Extended-versus-Improved effect of 27.67 reward units with a bootstrap interval of [23.27, 32.31], no reported comparison changes sign or loses significance. The results are therefore disclosed rather than invalidated.

The second repair synchronises the resumed learner's target mixer and reapplies the configured learning rate after optimizer-state restore. Because these runs were launched from a disclosed dirty worktree and the manifests recorded only a base SHA and a dirty flag, the learner source used by any historical run is not recoverable. The logged TD loss shows no spike at the resume boundary where an unsynchronised target mixer would perturb targets by roughly four reward units, which is consistent with the synchronisation having been present, but that is evidence rather than proof. Manifests now record `training_source_sha256` so the ambiguity cannot recur.

One consequence is load-bearing. The resume resolver accepts only manifest entries whose configuration digest matches the current gate, and the Extended manifests carry the pre-repair digest, so a corrected low-learning-rate continuation from Extended now fails by design. It needs Extended retrained under the repaired environment, or an explicitly recorded decision to warm-start across the repair; fresh seeds alone are not sufficient. See [the drift record](results/environment_drift.json) and [its measured impact](results/environment_drift_impact.json).

## 1. Research question and non-predetermined protocol

The study asks when cooperative MARL provides an advantage over simple adaptive heuristics. It does not assume that QMIX should win. Regime A uses independent event processes and unconstrained per-agent delivery. Regime B uses a two-packet shared channel, persistent spatially correlated events, ring-neighbor redundancy, heterogeneous energy trajectories, and network-coverage utility. The coupling mechanisms are physical or task-derived; they were not added after observing scores.

## 2. Collapse diagnosis and scientific corrections

The earlier collapsed result was not treated as an algorithm-only failure. The audit found four confounds: (1) latent measurement entropy was present before the SAMPLE decision, making the old threshold baseline an oracle; (2) `reset(seed=None)` reseeded every training episode to 42, replaying the same day; (3) the requested scenario/regime was not consistently propagated into EPyMARL; and (4) checkpoint export could search historical directories and had no isolated validation-only selection rule. Older learned checkpoints also had negative Q(sample)-Q(sleep) over relevant states, directly explaining Always-Sleep behavior.

The corrected observation is `[battery, causal event proxy, normalized AoI, lagged neighbor sampling rate, harvest forecast]`. Latent event entropy is measured only after sampling. The event proxy represents an always-on low-power detector with false positives, false negatives, noise, and an accounted monitoring energy cost. For compatibility with the requested baseline list, tables retain the name 'Entropy Threshold'; it is a threshold on this causal proxy, not latent entropy.

## 3. Pre-training sanity gates

Expensive training was blocked until the following artifact-backed gates passed. The post-training checkpoint audit additionally verified that all 18 selected policies chose both actions on validation seeds 201-210.

| Regime | Gate | Status | Evidence |
|---|---|---:|---|
| independent | `forced_action_and_controlled_reward` | PASS | `{"event_sample": 1.9, "event_sleep": -1.5, "non_event_sample": -0.101, "non_event_sleep": 0.0, "sample_executed": true}` |
| independent | `reward_component_scale` | PASS | `{"components": {"aoi": -0.0, "capture": 2.0, "miss": 0.0, "rejection": 0.0, "sample_energy": -0.1}, "max_abs_component": 2.0}` |
| independent | `entropy_event_causality` | PASS | `{"predecision_proxy": 0.25, "sample_measurement": 0.95, "sleep_measurement": null}` |
| independent | `tiny_environment_overfit` | PASS | `{"action_visits": [[408, 92], [89, 411]], "greedy_actions": [0, 1], "q_values": [[0.0, -0.1], [-1.5, 2.0]]}` |
| independent | `single_agent_learning` | PASS | `{"always_sleep_return_mean": -94.42499999999949, "learned_return_mean": 21.664999999999974, "sample_fraction_mean": 0.27638888888888885, "training_action_visits": [56179, 30221]}` |
| independent | `training_evaluation_contract_match` | PASS | `{"config_digest": "ae510abba91917cb", "obs_dim": 5, "regime": "independent", "scenario": "volatile", "state_dim": 20}` |
| independent | `exploration_both_actions` | PASS | `{"action_counts": [730, 270], "epsilon": 0.5}` |
| independent | `deterministic_reproducibility` | PASS | `{"seed": 44, "steps_compared": 12}` |
| coordinated | `forced_action_and_controlled_reward` | PASS | `{"event_sample": 1.9, "event_sleep": -1.5, "non_event_sample": -0.101, "non_event_sleep": 0.0, "sample_executed": true}` |
| coordinated | `reward_component_scale` | PASS | `{"components": {"aoi": -0.0, "capture": 2.0, "miss": 0.0, "rejection": 0.0, "sample_energy": -0.1}, "max_abs_component": 2.0}` |
| coordinated | `entropy_event_causality` | PASS | `{"predecision_proxy": 0.25, "sample_measurement": 0.95, "sleep_measurement": null}` |
| coordinated | `tiny_environment_overfit` | PASS | `{"action_visits": [[408, 92], [89, 411]], "greedy_actions": [0, 1], "q_values": [[0.0, -0.1], [-1.5, 2.0]]}` |
| coordinated | `single_agent_learning` | PASS | `{"always_sleep_return_mean": -94.42499999999949, "learned_return_mean": 21.664999999999974, "sample_fraction_mean": 0.27638888888888885, "training_action_visits": [56179, 30221]}` |
| coordinated | `training_evaluation_contract_match` | PASS | `{"config_digest": "6834240e8e625a6c", "obs_dim": 5, "regime": "coordinated", "scenario": "volatile", "state_dim": 20}` |
| coordinated | `exploration_both_actions` | PASS | `{"action_counts": [730, 270], "epsilon": 0.5}` |
| coordinated | `deterministic_reproducibility` | PASS | `{"seed": 44, "steps_compared": 12}` |

## 4. Training and checkpoint selection

IQL, VDN, and QMIX were trained independently in each regime using seeds 101, 102, and 103 (18 full runs), 60,000 requested environment steps per run, recurrent 64-unit shared agents, and validation-only checkpoint selection. The requested minimum of three training seeds was used rather than the preferred five; this limits inference about optimization variance. Uncertainty over environment seeds and variability over training seeds are both retained in the artifacts. Each run has an isolated directory, command/config snapshot, base Git SHA, sanity digest, all checkpoint validation decisions, and ONNX export metadata. A separate final provenance artifact records source/artifact hashes and the dirty-worktree disclosure.

| Regime | Algorithm | Seed | Selected step | Validation reward | Recall | Sample fraction | Both actions |
|---|---|---:|---:|---:|---:|---:|---:|
| independent | IQL | 101 | 46080 | -27.042 | 0.498 | 0.500 | True |
| independent | IQL | 102 | 46080 | -20.624 | 0.499 | 0.455 | True |
| independent | IQL | 103 | 46080 | -25.200 | 0.500 | 0.497 | True |
| independent | VDN | 101 | 46080 | -15.942 | 0.510 | 0.486 | True |
| independent | VDN | 102 | 46080 | 9.341 | 0.528 | 0.387 | True |
| independent | VDN | 103 | 46080 | 25.288 | 0.551 | 0.396 | True |
| independent | QMIX | 101 | 46080 | -26.948 | 0.498 | 0.499 | True |
| independent | QMIX | 102 | 46080 | 10.978 | 0.530 | 0.379 | True |
| independent | QMIX | 103 | 46080 | 8.756 | 0.533 | 0.425 | True |
| coordinated | IQL | 101 | 46080 | 8.328 | 0.450 | 0.250 | True |
| coordinated | IQL | 102 | 46080 | -30.933 | 0.418 | 0.286 | True |
| coordinated | IQL | 103 | 46080 | 23.909 | 0.474 | 0.262 | True |
| coordinated | VDN | 101 | 46080 | -3.744 | 0.449 | 0.268 | True |
| coordinated | VDN | 102 | 46080 | 2.354 | 0.453 | 0.270 | True |
| coordinated | VDN | 103 | 46080 | 32.099 | 0.479 | 0.256 | True |
| coordinated | QMIX | 101 | 46080 | -24.342 | 0.413 | 0.272 | True |
| coordinated | QMIX | 102 | 46080 | 48.757 | 0.488 | 0.232 | True |
| coordinated | QMIX | 103 | 46080 | 65.618 | 0.507 | 0.225 | True |

## 5. Final paired 30-seed benchmark

Every policy used the identical locked environment seeds 1001-1030. Learned-policy results first average the three training replicas within each environment seed; the 30 environment seeds are then the sampling units for the reported mean, sample standard deviation, and t-based 95% confidence interval. This prevents the three learned replicas from being misrepresented as 90 independent environment trials.

### Independent regime: information and energy metrics

| Policy | Event recall | Missed-event rate | Energy consumed | Energy harvested | Final battery |
|---|---:|---:|---:|---:|---:|
| Always Sample | 0.498 ± 0.041 [0.483, 0.514] | 0.502 ± 0.041 [0.486, 0.517] | 28.959 ± 0.249 [28.866, 29.052] | 27.755 ± 0.183 [27.687, 27.824] | 0.017 ± 0.006 [0.015, 0.019] |
| Always Sleep | 0.000 ± 0.000 [0.000, 0.000] | 1.000 ± 0.000 [1.000, 1.000] | 0.196 ± 0.000 [0.196, 0.196] | 27.755 ± 0.183 [27.687, 27.824] | 0.988 ± 0.000 [0.988, 0.988] |
| Battery + Entropy | 0.732 ± 0.045 [0.716, 0.749] | 0.268 ± 0.045 [0.251, 0.284] | 11.644 ± 0.707 [11.380, 11.908] | 27.755 ± 0.183 [27.687, 27.824] | 0.173 ± 0.054 [0.153, 0.193] |
| Entropy Threshold | 0.748 ± 0.046 [0.731, 0.765] | 0.252 ± 0.046 [0.235, 0.269] | 11.888 ± 0.709 [11.623, 12.152] | 27.755 ± 0.183 [27.687, 27.824] | 0.154 ± 0.059 [0.132, 0.176] |
| Fixed Interval | 0.083 ± 0.016 [0.077, 0.089] | 0.917 ± 0.016 [0.911, 0.923] | 4.996 ± 0.000 [4.996, 4.996] | 27.755 ± 0.183 [27.687, 27.824] | 0.670 ± 0.000 [0.670, 0.670] |
| Greedy | 0.496 ± 0.049 [0.478, 0.515] | 0.504 ± 0.049 [0.485, 0.522] | 22.714 ± 1.450 [22.173, 23.256] | 27.755 ± 0.183 [27.687, 27.824] | 0.019 ± 0.009 [0.015, 0.022] |
| IQL | 0.500 ± 0.040 [0.485, 0.515] | 0.500 ± 0.040 [0.485, 0.515] | 28.100 ± 0.291 [27.992, 28.209] | 27.755 ± 0.183 [27.687, 27.824] | 0.015 ± 0.005 [0.013, 0.017] |
| QMIX | 0.519 ± 0.038 [0.505, 0.533] | 0.481 ± 0.038 [0.467, 0.495] | 25.371 ± 0.381 [25.229, 25.514] | 27.755 ± 0.183 [27.687, 27.824] | 0.015 ± 0.005 [0.013, 0.017] |
| Random Feasible | 0.325 ± 0.032 [0.313, 0.337] | 0.675 ± 0.032 [0.663, 0.687] | 19.113 ± 0.564 [18.902, 19.323] | 27.755 ± 0.183 [27.687, 27.824] | 0.020 ± 0.006 [0.018, 0.022] |
| VDN | 0.534 ± 0.037 [0.520, 0.548] | 0.466 ± 0.037 [0.452, 0.480] | 24.635 ± 0.488 [24.453, 24.817] | 27.755 ± 0.183 [27.687, 27.824] | 0.015 ± 0.005 [0.013, 0.017] |

### Independent regime: freshness, coordination, and reward metrics

| Policy | Mean AoI | p95 AoI | Max AoI | Redundant sampling | Network coverage | Network utility | Raw reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Always Sample | 15.867 ± 0.311 [15.751, 15.983] | 60.260 ± 0.692 [60.002, 60.518] | 69.633 ± 0.765 [69.348, 69.919] | 546.033 ± 5.786 [543.873, 548.194] | 0.756 ± 0.025 [0.747, 0.765] | 0.377 ± 0.041 [0.362, 0.393] | -25.958 ± 28.848 [-36.730, -15.186] |
| Always Sleep | 144.500 ± 0.000 [144.500, 144.500] | 274.000 ± 0.000 [274.000, 274.000] | 288.000 ± 0.000 [288.000, 288.000] | 0.000 ± 0.000 [0.000, 0.000] | 0.475 ± 0.034 [0.462, 0.487] | 0.000 ± 0.000 [0.000, 0.000] | -349.200 ± 22.486 [-357.596, -340.804] |
| Battery + Entropy | 5.917 ± 0.766 [5.631, 6.203] | 21.735 ± 4.300 [20.129, 23.341] | 44.800 ± 9.065 [41.415, 48.185] | 49.700 ± 6.854 [47.141, 52.259] | 0.892 ± 0.028 [0.882, 0.903] | 0.655 ± 0.060 [0.632, 0.677] | 179.616 ± 26.543 [169.704, 189.527] |
| Entropy Threshold | 5.545 ± 0.673 [5.294, 5.796] | 19.782 ± 3.791 [18.366, 21.197] | 40.533 ± 9.039 [37.158, 43.909] | 50.933 ± 6.928 [48.346, 53.520] | 0.901 ± 0.028 [0.890, 0.911] | 0.675 ± 0.061 [0.652, 0.698] | 190.281 ± 26.874 [180.246, 200.315] |
| Fixed Interval | 5.438 ± 0.000 [5.438, 5.438] | 11.000 ± 0.000 [11.000, 11.000] | 11.000 ± 0.000 [11.000, 11.000] | 0.000 ± 0.000 [0.000, 0.000] | 0.600 ± 0.030 [0.589, 0.611] | 0.050 ± 0.011 [0.046, 0.054] | -255.255 ± 21.718 [-263.364, -247.145] |
| Greedy | 14.513 ± 0.549 [14.308, 14.718] | 56.260 ± 1.507 [55.697, 56.823] | 69.333 ± 0.758 [69.050, 69.616] | 228.467 ± 69.449 [202.534, 254.400] | 0.788 ± 0.023 [0.779, 0.796] | 0.392 ± 0.047 [0.374, 0.409] | -13.859 ± 34.976 [-26.919, -0.799] |
| IQL | 15.732 ± 0.327 [15.610, 15.854] | 59.817 ± 0.744 [59.539, 60.094] | 69.467 ± 0.629 [69.232, 69.701] | 509.989 ± 6.888 [507.417, 512.561] | 0.758 ± 0.024 [0.749, 0.767] | 0.379 ± 0.041 [0.364, 0.395] | -23.317 ± 28.960 [-34.131, -12.503] |
| QMIX | 14.291 ± 0.373 [14.151, 14.430] | 56.361 ± 0.814 [56.057, 56.665] | 68.233 ± 0.931 [67.886, 68.581] | 386.956 ± 9.265 [383.496, 390.415] | 0.779 ± 0.021 [0.771, 0.787] | 0.406 ± 0.038 [0.391, 0.420] | -2.990 ± 26.980 [-13.065, 7.084] |
| Random Feasible | 9.038 ± 0.773 [8.749, 9.327] | 42.138 ± 2.474 [41.214, 43.062] | 60.200 ± 4.845 [58.391, 62.009] | 180.467 ± 12.213 [175.906, 185.027] | 0.786 ± 0.023 [0.777, 0.794] | 0.256 ± 0.030 [0.244, 0.267] | -119.734 ± 24.026 [-128.705, -110.762] |
| VDN | 13.512 ± 0.492 [13.328, 13.696] | 54.418 ± 1.207 [53.967, 54.869] | 66.844 ± 1.353 [66.339, 67.349] | 361.144 ± 12.149 [356.608, 365.681] | 0.784 ± 0.021 [0.776, 0.792] | 0.420 ± 0.038 [0.406, 0.434] | 9.505 ± 26.778 [-0.494, 19.504] |

### Coordinated regime: information and energy metrics

| Policy | Event recall | Missed-event rate | Energy consumed | Energy harvested | Final battery |
|---|---:|---:|---:|---:|---:|
| Always Sample | 0.318 ± 0.051 [0.299, 0.338] | 0.682 ± 0.051 [0.662, 0.701] | 19.331 ± 0.240 [19.241, 19.420] | 27.798 ± 0.232 [27.711, 27.884] | 0.021 ± 0.004 [0.020, 0.023] |
| Always Sleep | 0.000 ± 0.000 [0.000, 0.000] | 1.000 ± 0.000 [1.000, 1.000] | 0.196 ± 0.000 [0.196, 0.196] | 27.798 ± 0.232 [27.711, 27.884] | 0.988 ± 0.000 [0.988, 0.988] |
| Battery + Entropy | 0.601 ± 0.047 [0.583, 0.618] | 0.399 ± 0.047 [0.382, 0.417] | 10.686 ± 1.055 [10.292, 11.080] | 27.798 ± 0.232 [27.711, 27.884] | 0.221 ± 0.121 [0.176, 0.266] |
| Entropy Threshold | 0.611 ± 0.043 [0.595, 0.627] | 0.389 ± 0.043 [0.373, 0.405] | 10.876 ± 1.121 [10.457, 11.294] | 27.798 ± 0.232 [27.711, 27.884] | 0.208 ± 0.133 [0.158, 0.257] |
| Fixed Interval | 0.081 ± 0.015 [0.075, 0.087] | 0.919 ± 0.015 [0.913, 0.925] | 4.996 ± 0.000 [4.996, 4.996] | 27.798 ± 0.232 [27.711, 27.884] | 0.670 ± 0.000 [0.670, 0.670] |
| Greedy | 0.318 ± 0.051 [0.299, 0.338] | 0.682 ± 0.051 [0.662, 0.701] | 19.331 ± 0.240 [19.241, 19.420] | 27.798 ± 0.232 [27.711, 27.884] | 0.021 ± 0.004 [0.020, 0.023] |
| IQL | 0.444 ± 0.058 [0.422, 0.465] | 0.556 ± 0.058 [0.535, 0.578] | 15.270 ± 0.742 [14.993, 15.547] | 27.798 ± 0.232 [27.711, 27.884] | 0.030 ± 0.027 [0.020, 0.040] |
| QMIX | 0.467 ± 0.057 [0.446, 0.488] | 0.533 ± 0.057 [0.512, 0.554] | 13.916 ± 0.726 [13.645, 14.187] | 27.798 ± 0.232 [27.711, 27.884] | 0.034 ± 0.026 [0.025, 0.044] |
| Random Feasible | 0.283 ± 0.046 [0.266, 0.301] | 0.717 ± 0.046 [0.699, 0.734] | 17.003 ± 0.440 [16.838, 17.167] | 27.798 ± 0.232 [27.711, 27.884] | 0.020 ± 0.006 [0.018, 0.022] |
| VDN | 0.456 ± 0.057 [0.434, 0.477] | 0.544 ± 0.057 [0.523, 0.566] | 15.188 ± 0.689 [14.931, 15.445] | 27.798 ± 0.232 [27.711, 27.884] | 0.024 ± 0.013 [0.019, 0.029] |

### Coordinated regime: freshness, coordination, and reward metrics

| Policy | Mean AoI | p95 AoI | Max AoI | Redundant sampling | Network coverage | Network utility | Raw reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Always Sample | 8.890 ± 0.435 [8.728, 9.053] | 41.560 ± 1.203 [41.111, 42.009] | 57.100 ± 2.524 [56.158, 58.042] | 183.833 ± 2.547 [182.882, 184.784] | 0.888 ± 0.037 [0.874, 0.901] | 0.284 ± 0.054 [0.264, 0.304] | -195.860 ± 42.231 [-211.629, -180.090] |
| Always Sleep | 144.500 ± 0.000 [144.500, 144.500] | 274.000 ± 0.000 [274.000, 274.000] | 288.000 ± 0.000 [288.000, 288.000] | 0.000 ± 0.000 [0.000, 0.000] | 0.674 ± 0.076 [0.645, 0.702] | 0.000 ± 0.000 [0.000, 0.000] | -373.950 ± 78.163 [-403.137, -344.763] |
| Battery + Entropy | 5.946 ± 0.872 [5.621, 6.271] | 21.745 ± 5.126 [19.831, 23.659] | 41.267 ± 10.048 [37.515, 45.019] | 46.367 ± 7.933 [43.404, 49.329] | 0.955 ± 0.028 [0.944, 0.965] | 0.575 ± 0.059 [0.553, 0.597] | 149.924 ± 34.975 [136.864, 162.984] |
| Entropy Threshold | 5.698 ± 0.749 [5.419, 5.978] | 20.278 ± 3.832 [18.848, 21.709] | 41.367 ± 10.149 [37.577, 45.156] | 47.567 ± 8.439 [44.415, 50.718] | 0.960 ± 0.025 [0.950, 0.969] | 0.587 ± 0.054 [0.567, 0.608] | 159.033 ± 36.987 [145.221, 172.844] |
| Fixed Interval | 5.438 ± 0.000 [5.438, 5.438] | 11.000 ± 0.000 [11.000, 11.000] | 11.000 ± 0.000 [11.000, 11.000] | 0.000 ± 0.000 [0.000, 0.000] | 0.754 ± 0.056 [0.733, 0.775] | 0.061 ± 0.012 [0.056, 0.066] | -253.266 ± 57.927 [-274.896, -231.636] |
| Greedy | 8.890 ± 0.435 [8.728, 9.053] | 41.560 ± 1.203 [41.111, 42.009] | 57.100 ± 2.524 [56.158, 58.042] | 183.833 ± 2.547 [182.882, 184.784] | 0.888 ± 0.037 [0.874, 0.901] | 0.284 ± 0.054 [0.264, 0.304] | -195.860 ± 42.231 [-211.629, -180.090] |
| IQL | 8.594 ± 0.856 [8.274, 8.914] | 36.464 ± 3.496 [35.159, 37.770] | 57.389 ± 4.345 [55.766, 59.011] | 73.911 ± 13.770 [68.769, 79.053] | 0.921 ± 0.033 [0.909, 0.933] | 0.410 ± 0.066 [0.386, 0.435] | -5.032 ± 40.150 [-20.024, 9.961] |
| QMIX | 9.075 ± 0.692 [8.817, 9.334] | 37.736 ± 3.080 [36.586, 38.886] | 59.422 ± 3.955 [57.946, 60.899] | 58.611 ± 10.065 [54.853, 62.370] | 0.920 ± 0.031 [0.909, 0.932] | 0.431 ± 0.064 [0.407, 0.455] | 24.382 ± 41.023 [9.064, 39.700] |
| Random Feasible | 7.248 ± 0.707 [6.984, 7.512] | 35.438 ± 2.923 [34.347, 36.530] | 55.433 ± 4.861 [53.618, 57.249] | 99.333 ± 7.251 [96.626, 102.041] | 0.881 ± 0.038 [0.867, 0.895] | 0.251 ± 0.049 [0.232, 0.269] | -149.202 ± 42.053 [-164.905, -133.499] |
| VDN | 7.663 ± 0.800 [7.365, 7.962] | 34.179 ± 3.971 [32.696, 35.662] | 55.189 ± 5.764 [53.037, 57.341] | 78.844 ± 11.487 [74.555, 83.134] | 0.923 ± 0.034 [0.910, 0.936] | 0.422 ± 0.067 [0.397, 0.447] | 2.884 ± 38.510 [-11.496, 17.264] |

## 6. Paired statistical and practical comparison

The table below compares each learned method with the causal proxy threshold using a two-sided one-sample t-test on paired seed differences. The test assumes the 30 paired differences are approximately normal. Positive advantage always means better engineering performance; energy and AoI signs are reversed before testing because lower is better. Cohen's dz is the paired effect size. A practical difference is defined prospectively here as at least 5% of the comparator mean; it is reported separately from p < 0.05. No multiple-comparison-adjusted confirmatory claim is made.

| Regime | Learned policy | Metric | Mean engineering advantage | t | p | Cohen dz | Practical outcome |
|---|---|---|---:|---:|---:|---:|---|
| independent | IQL | raw_episode_reward | -213.5975 | -49.702 | 1.276e-29 | -9.074 | learned disadvantage |
| independent | IQL | event_recall | -0.2484 | -32.616 | 2.081e-24 | -5.955 | learned disadvantage |
| independent | IQL | total_energy_consumption | -16.2128 | -141.034 | 1.078e-42 | -25.749 | learned disadvantage |
| independent | IQL | mean_aoi | -10.1871 | -96.807 | 5.777e-38 | -17.674 | learned disadvantage |
| independent | VDN | raw_episode_reward | -180.7757 | -41.550 | 2.147e-27 | -7.586 | learned disadvantage |
| independent | VDN | event_recall | -0.2138 | -28.327 | 1.100e-22 | -5.172 | learned disadvantage |
| independent | VDN | total_energy_consumption | -12.7478 | -125.913 | 2.875e-41 | -22.988 | learned disadvantage |
| independent | VDN | mean_aoi | -7.9670 | -63.403 | 1.166e-32 | -11.576 | learned disadvantage |
| independent | QMIX | raw_episode_reward | -193.2709 | -44.377 | 3.275e-28 | -8.102 | learned disadvantage |
| independent | QMIX | event_recall | -0.2287 | -30.159 | 1.894e-23 | -5.506 | learned disadvantage |
| independent | QMIX | total_energy_consumption | -13.4839 | -136.736 | 2.641e-42 | -24.964 | learned disadvantage |
| independent | QMIX | mean_aoi | -8.7456 | -78.195 | 2.759e-35 | -14.276 | learned disadvantage |
| coordinated | IQL | raw_episode_reward | -164.0642 | -21.683 | 1.818e-19 | -3.959 | learned disadvantage |
| coordinated | IQL | event_recall | -0.1677 | -18.934 | 7.209e-18 | -3.457 | learned disadvantage |
| coordinated | IQL | total_energy_consumption | -4.3944 | -32.458 | 2.388e-24 | -5.926 | learned disadvantage |
| coordinated | IQL | mean_aoi | -2.8957 | -20.684 | 6.585e-19 | -3.776 | learned disadvantage |
| coordinated | VDN | raw_episode_reward | -156.1483 | -20.869 | 5.170e-19 | -3.810 | learned disadvantage |
| coordinated | VDN | event_recall | -0.1555 | -17.931 | 3.097e-17 | -3.274 | learned disadvantage |
| coordinated | VDN | total_energy_consumption | -4.3122 | -31.052 | 8.326e-24 | -5.669 | learned disadvantage |
| coordinated | VDN | mean_aoi | -1.9649 | -12.773 | 1.959e-13 | -2.332 | learned disadvantage |
| coordinated | QMIX | raw_episode_reward | -134.6503 | -20.617 | 7.192e-19 | -3.764 | learned disadvantage |
| coordinated | QMIX | event_recall | -0.1445 | -17.707 | 4.331e-17 | -3.233 | learned disadvantage |
| coordinated | QMIX | total_energy_consumption | -3.0400 | -21.834 | 1.504e-19 | -3.986 | learned disadvantage |
| coordinated | QMIX | mean_aoi | -3.3769 | -26.464 | 7.373e-22 | -4.832 | learned disadvantage |

## 7. Action and value diagnostics

The repaired policies are not Always Sleep. Sample-action confidence intervals exclude zero, while Q-gap distributions include state-dependent positive and negative values rather than a uniformly negative gap. This supports a substantive negative performance result rather than a trivial collapsed-policy comparison.

| Regime | Policy | SAMPLE fraction mean [95% CI] | Q-gap mean | Q-gap std | Q-gap 5th / 50th / 95th |
|---|---|---:|---:|---:|---:|
| independent | IQL | 0.484 [0.483, 0.486] | 0.968 | 0.548 | 0.200 / 0.891 / 2.041 |
| independent | VDN | 0.424 [0.421, 0.427] | 0.527 | 0.649 | -0.244 / 0.330 / 1.924 |
| independent | QMIX | 0.437 [0.435, 0.440] | 0.630 | 0.552 | -0.084 / 0.504 / 1.650 |
| coordinated | IQL | 0.328 [0.317, 0.338] | 0.356 | 0.879 | -0.730 / 0.159 / 2.065 |
| coordinated | VDN | 0.324 [0.314, 0.335] | 0.209 | 0.658 | -0.549 / -0.002 / 1.520 |
| coordinated | QMIX | 0.283 [0.275, 0.291] | 0.154 | 0.477 | -0.441 / 0.053 / 1.082 |

## 8. Proper retrained ablations

Each QMIX ablation removed one named component, retrained three fresh seeds at the same 60,000-step budget, and used the same 30 held-out seeds. All variants were scored in the common full coordinated environment and full reward; no frozen policy was evaluated under a changed reward and called a training ablation.

| Variant | Reward mean ± std [95% CI] | Recall mean ± std [95% CI] | Energy mean ± std [95% CI] | AoI mean ± std [95% CI] |
|---|---:|---:|---:|---:|
| Full QMIX | 24.382 ± 41.023 [9.064, 39.700] | 0.467 ± 0.057 [0.446, 0.488] | 13.916 ± 0.726 [13.645, 14.187] | 9.075 ± 0.692 [8.817, 9.334] |
| No agent ID | 86.507 ± 39.270 [71.843, 101.171] | 0.534 ± 0.055 [0.514, 0.555] | 12.633 ± 0.905 [12.294, 12.971] | 6.860 ± 0.873 [6.534, 7.186] |
| No AoI term | 28.320 ± 40.433 [13.222, 43.418] | 0.471 ± 0.055 [0.450, 0.491] | 13.744 ± 0.724 [13.474, 14.015] | 8.916 ± 0.698 [8.656, 9.177] |
| No coordination constraint | -190.497 ± 42.004 [-206.182, -174.813] | 0.320 ± 0.051 [0.301, 0.339] | 19.313 ± 0.234 [19.225, 19.400] | 9.014 ± 0.411 [8.861, 9.168] |
| No energy term | -14.806 ± 41.606 [-30.342, 0.730] | 0.429 ± 0.058 [0.407, 0.450] | 14.839 ± 0.589 [14.619, 15.059] | 10.029 ± 0.560 [9.820, 10.238] |
| No neighbor signal | 64.850 ± 37.601 [50.810, 78.891] | 0.513 ± 0.058 [0.492, 0.535] | 12.546 ± 0.964 [12.187, 12.906] | 7.784 ± 0.982 [7.418, 8.151] |
| No redundancy penalty | -93.754 ± 41.699 [-109.324, -78.183] | 0.372 ± 0.055 [0.352, 0.393] | 17.472 ± 0.333 [17.347, 17.596] | 10.212 ± 0.560 [10.003, 10.421] |

Full-minus-ablated paired reward tests (positive means the full model is better):

| Ablation | Mean full-model advantage | t | p | Cohen dz |
|---|---:|---:|---:|---:|
| No coordination constraint | 214.880 | 41.734 | 1.893e-27 | 7.619 |
| No redundancy penalty | 118.136 | 32.852 | 1.698e-24 | 5.998 |
| No agent ID | -62.124 | -23.784 | 1.428e-20 | -4.342 |
| No energy term | 39.188 | 20.720 | 6.281e-19 | 3.783 |
| No neighbor signal | -40.468 | -12.151 | 6.677e-13 | -2.219 |
| No AoI term | -3.938 | -5.217 | 1.389e-05 | -0.952 |

### Ablation interpretation

The ablations separate useful task coupling from unhelpful representation burden. Relative to the full model, removing the coordination constraint cost 214.88 reward units (t=41.73, p=1.89e-27, dz=7.62), removing the redundancy penalty cost 118.14 reward units (t=32.85, p=1.70e-24, dz=6.00), and removing the energy term cost 39.19 reward units (t=20.72, p=6.28e-19, dz=3.78); these positive full-model advantages support retaining those three mechanisms in this task. Conversely, the full model was worse than no agent ID by 62.12 reward units (p=1.43e-20) and worse than no neighbor signal by 40.47 units (p=6.68e-13). Removing AoI also produced a smaller 3.94-unit improvement (p=1.39e-05). In this four-agent setting, identity and lagged-neighbor inputs appear to add optimization or overfitting burden rather than exploitable coordination information. That is a model-and-budget result, not proof that those signals are intrinsically harmful.

## 9. Figures and artifact lineage

All figures are generated from saved experiment CSV/JSON data by `deployment/generate_plots.py`; no measured metric is typed into the plotting code.

1. [Training learning curves](results/figures/01_training_learning_curves.png)
2. [Action distribution](results/figures/02_action_distribution.png)
3. [Q(sample) - Q(sleep)](results/figures/03_q_sample_minus_sleep.png)
4. [Reward components](results/figures/04_reward_component_distributions.png)
5. [Event recall](results/figures/05_event_recall.png)
6. [Energy consumption](results/figures/06_energy_consumption.png)
7. [AoI](results/figures/07_mean_aoi.png)
8. [Redundant sampling](results/figures/08_redundant_sampling.png)
9. [Battery trajectories](results/figures/09_battery_trajectory.png)
10. [Regime comparison](results/figures/10_independent_vs_coordinated.png)
11. [Retrained ablations](results/figures/11_retrained_ablation_results.png)
12. [Energy-recall Pareto view](results/figures/12_energy_vs_event_recall.png)
13. [Energy-AoI Pareto view](results/figures/13_energy_vs_aoi.png)

## 10. Interpretation and retained negative result

The independent task is sufficiently separable that a causal local proxy rule captures its useful structure more directly than value-decomposition MARL. The coordinated task does reveal a learned-method ordering in point estimates, with QMIX outperforming IQL and VDN in mean raw reward, but QMIX remains dominated by the threshold heuristic on the primary recall-energy-freshness trade-off. Therefore the experiments do not support the claim that cooperative MARL provides an engineering advantage under the evaluated conditions. A narrower, defensible conclusion is that genuine coupling alone is insufficient: MARL may require larger networks, delayed/noisy neighbor communication, non-myopic coverage dependencies, stronger heterogeneity, longer training, or architectures that exploit graph structure before its representational cost is justified. These are future hypotheses, not results from this benchmark.

## 11. Limitations and reproducibility boundaries

The study uses four agents, one 24-hour volatile scenario, a ring topology, a simplified round-robin MAC, a scalar reward, three training seeds, and 30 simulation seeds. The event-proxy error rates and reward weights are modeling choices, not field-calibrated estimates. Paired t-tests are exploratory across multiple policies and metrics. The worktree was intentionally uncommitted during these runs. Full-run manifests record the base Git SHA, while `results/provenance.json` records the final dirty status and source/artifact hashes; exact run configs and generated artifacts must accompany any external reproduction. No result here establishes field performance or a universal absence of MARL benefit. See [LIMITATIONS.md](LIMITATIONS.md) for the physical and deployment boundary conditions.

### Reproduction commands

```bash
venv/bin/python training/sanity_checks.py --regime all --scenario volatile
venv/bin/python training/train_all.py --alg all --seeds 101,102,103 --regime all --scenario volatile --t_max 60000
venv/bin/python deployment/evaluate_all.py --regime all --scenario volatile --n-seeds 30
venv/bin/python deployment/ablation_study.py --train --scenario volatile --t_max 60000 --train-seeds 101,102,103 --n-test-seeds 30
venv/bin/python deployment/generate_plots.py
venv/bin/python deployment/build_final_report.py
venv/bin/python deployment/write_provenance.py
venv/bin/python deployment/audit_final_artifacts.py
venv/bin/python -m pytest -q
```
