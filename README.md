# Cooperative MARL for Energy-Harvesting Sensor Scheduling

This project asks a deliberately neutral research question:

> When does cooperative MARL provide an advantage over simple adaptive heuristics for energy-harvesting sensor networks?

The final experiments do not force QMIX to win. After repairing a learned-policy collapse and removing an entropy oracle from the decision-time observation, IQL, VDN, and QMIX all learn non-trivial SAMPLE/SLEEP policies. Nevertheless, a causal low-power event-proxy threshold outperforms the learned methods in both evaluated regimes. In the coordinated regime QMIX has the strongest learned-policy reward point estimate, but it still does not beat the strongest heuristic on the recall-energy-AoI trade-off.

A subsequent validation-driven training upgrade fixes terminal checkpoint loss, adds replay updates for the long 288-step episodes, and uses a homogeneous shared policy without the harmful lagged-neighbor input. On a fresh predeclared 30-seed holdout (`2001`--`2030`), coordinated QMIX improves from 21.94 to 123.18 mean reward, recall from 0.456 to 0.567, energy from 14.15 to 12.98, and mean AoI from 8.90 to 5.33. The causal threshold still leads on reward (168.22), recall (0.607), and energy (11.20), so the upgraded result remains scientifically negative rather than forcing QMIX to win. See [the training-upgrade report](results/training_upgrade/coordinated/REPORT.md).

The complete measured result, confidence intervals, paired tests, ablations, limitations, and negative-result interpretation are in [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md).

## Scientific design

Each sensor observes five causal, deployable features:

1. residual battery;
2. a noisy always-on event proxy whose energy cost is accounted for;
3. normalized Age of Information;
4. lagged neighbor sampling rate;
5. time-of-day harvest forecast.

Latent measurement entropy is unavailable before the action and is measured only after SAMPLE. The benchmark label `Entropy Threshold` is retained for compatibility with the research protocol, but it thresholds the causal event proxy and is not an oracle.

Two environments are trained and evaluated separately:

- `independent`: independent local events, no shared-channel bottleneck, and no coordination reward.
- `coordinated`: persistent spatially correlated events, a two-packet shared channel, ring-neighbor redundancy, heterogeneous harvest trajectories, and network coverage utility.

The benchmark includes Always Sleep, Always Sample when feasible, Random Feasible, Fixed Interval, Entropy Threshold, Battery + Entropy, Greedy, IQL, VDN, and QMIX. Every policy is evaluated on the same 30 seeds (`1001`–`1030`).

## Reproduce the final pipeline

Create a Python environment and install [requirements.txt](requirements.txt), then run:

```bash
venv/bin/python training/sanity_checks.py --regime all --scenario volatile
venv/bin/python training/train_all.py --alg all --seeds 101,102,103 --regime all --scenario volatile --t_max 60000
venv/bin/python deployment/evaluate_all.py --regime all --scenario volatile --n-seeds 30
venv/bin/python deployment/ablation_study.py --train --scenario volatile --t_max 60000 --train-seeds 101,102,103 --n-test-seeds 30
venv/bin/python deployment/generate_plots.py
venv/bin/python deployment/build_final_report.py
venv/bin/python deployment/write_provenance.py
venv/bin/python deployment/audit_final_artifacts.py
venv/bin/pytest -q
```

Training is blocked unless controlled rewards, action semantics, reward scales, causal observations, tiny overfit, single-agent learning, configuration parity, exploration, and deterministic replay checks pass. Each training run is isolated and records its command, configuration digest, sanity artifact, validation-only checkpoint decision, and ONNX metadata.

## Repository map

```text
environment/                 causal sensor and two-regime network environments
training/sanity_checks.py    pre-training scientific gates
training/train_all.py        isolated IQL/VDN/QMIX multi-seed training
training/policy_runtime.py   stateful GRU checkpoint and ONNX inference
deployment/evaluate_all.py   paired 30-seed, ten-policy benchmark
deployment/ablation_study.py retrain-and-evaluate QMIX ablations
deployment/generate_plots.py 13 data-derived scientific figures
deployment/build_final_report.py artifact-derived final report
results/sanity/              gate evidence
results/experiments/         configs, validation decisions, checkpoints, logs
results/learned_models/      selected full-policy ONNX exports
results/ablation_models/     selected retrained-ablation ONNX exports
results/final/               raw, per-seed, summary, statistical, Q, and trajectory data
results/ablations/           ablation data and paired comparisons
results/figures/             final plots and source manifest
results/provenance.json      source and primary-artifact SHA-256 archive
hardware_eval/               separate analytical TinyML feasibility tooling
```

## Reproducibility notes

- Training seeds: `101`, `102`, `103`.
- Validation seeds: `201`–`210`; checkpoint selection uses only mean validation reward.
- Locked test seeds: `1001`–`1030`.
- Learned replicas are averaged within each environment seed before the 30-seed confidence interval is calculated.
- Paired tests use identical environment seeds and report the statistic, p-value, paired Cohen's dz, and a separate 5% engineering-practicality threshold.
- Reward components, actions, Q-gaps, trajectories, and checkpoint decisions are saved rather than inferred from headline scores.

See [LIMITATIONS.md](LIMITATIONS.md) before making field-deployment or general MARL claims.

## Validation-driven training upgrade

The historical `baseline` profile remains available unchanged. The upgraded coordinated QMIX run is isolated under `results/upgrade_*` and does not overwrite the published Phase 25--33 artifacts:

```bash
venv/bin/python training/train_all.py --profile improved --alg qmix --seeds 101,102,103 --regime coordinated --scenario volatile
venv/bin/python deployment/evaluate_training_upgrade.py --regime coordinated --scenario volatile --n-seeds 30
```

Profile selection and checkpoint selection use validation seeds `201`--`210`. Because the original test set had already informed the ablation analysis, upgraded claims use the separately declared seeds `2001`--`2030`.
