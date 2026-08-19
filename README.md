# Cooperative MARL for Energy-Harvesting Sensor Scheduling

This project asks a deliberately neutral research question:

> When does cooperative MARL provide an advantage over simple adaptive heuristics for energy-harvesting sensor networks?

The final experiments do not force QMIX to win. After repairing a learned-policy collapse and removing an entropy oracle from the decision-time observation, IQL, VDN, and QMIX all learn non-trivial SAMPLE/SLEEP policies. Nevertheless, a causal low-power event-proxy threshold outperforms the learned methods in both evaluated regimes. In the coordinated regime QMIX has the strongest learned-policy reward point estimate, but it still does not beat the strongest heuristic on the recall-energy-AoI trade-off.

A subsequent validation-driven training upgrade fixes terminal checkpoint loss, adds replay updates for the long 288-step episodes, and uses a homogeneous shared policy without the harmful lagged-neighbor input. On a fresh predeclared 30-seed holdout (`2001`--`2030`), coordinated QMIX improves from 21.94 to 123.18 mean reward, recall from 0.456 to 0.567, energy from 14.15 to 12.98, and mean AoI from 8.90 to 5.33. The causal threshold still leads on reward (168.22), recall (0.607), and energy (11.20), so the upgraded result remains scientifically negative rather than forcing QMIX to win. See [the training-upgrade report](results/training_upgrade/coordinated/REPORT.md).

A second split-locked experiment warm-starts the same three QMIX replicas from 180k to a 360k-step horizon. On the untouched final seeds `3001`--`3030`, Extended QMIX improves over Improved QMIX from 119.09 to 146.76 reward, from 0.575 to 0.602 recall, and from 12.69 to 11.72 energy units. It still trails the causal threshold on reward (156.32), recall (0.609), energy (10.74), and redundant sampling, although it has lower mean AoI (5.07 versus 5.62). See [the frozen split-locked report](results/training_v2/final/REPORT.md).

A third split-locked iteration leaves training and the environment unchanged and combines the three Extended QMIX replicas at deployment. On selection seeds `231`--`250`, unanimous SAMPLE voting passed the predeclared promotion rule while majority voting failed. On untouched final seeds `4001`--`4030`, the unanimous ensemble improves reward from 153.38 to 157.43 versus the Extended replica mean and improves recall, energy, redundancy, and coverage, with a small AoI tradeoff. Its reward gap to the threshold is only 2.27 units and statistically unresolved, but it still consumes 0.73 more energy units and produces 4.40 more redundant samples. It also requires three model evaluations per decision. See [the v3 final report](results/training_v3/final/REPORT.md).

An actual weight-updating continuation was also tested strictly, but a later artifact audit found that it was mislabeled: loading the source optimizer restored `3e-4`, overriding the declared `1e-4`. All 37 saved continuation checkpoints prove the effective rate stayed `3e-4`. The candidate already failed its selection gate, and final seeds `5001`--`5030` remain untouched. Its selection numbers are retained only as a rejected same-rate warm-start, not as evidence about low-LR refinement; see [the invalidation record](results/training_v4/INVALIDATED.json).

Two source repairs landed after every run and every evaluation above had already been produced, so the committed source does not bit-exactly regenerate the committed numbers. The larger one makes SAMPLE feasibility charge the unavoidable same-step sleep and proxy-monitor energy, which the previous rule ignored. Measured under the policy that saturates the battery floor, it moves mean episode reward by less than 0.2 units with a worst single seed of 4.2, against a headline Extended-versus-Improved effect of 27.67 with a bootstrap interval of `[23.27, 32.31]`, so no reported comparison changes sign or loses significance. The results are disclosed rather than invalidated; see [the drift record](results/environment_drift.json), its [measured impact](results/environment_drift_impact.json), and [LIMITATIONS.md](LIMITATIONS.md).

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
venv/bin/python -m pytest -q
```

Training is blocked unless controlled rewards, action semantics, reward scales, causal observations, tiny overfit, single-agent learning, configuration parity, exploration, and deterministic replay checks pass. Each training run is isolated and records its command, configuration digest, sanity artifact, validation-only checkpoint decision, and ONNX metadata.

## Repository map

```text
environment/                 causal sensor and two-regime network environments
training/sanity_checks.py    pre-training scientific gates
training/train_all.py        isolated IQL/VDN/QMIX multi-seed training
training/policy_runtime.py   stateful GRU checkpoint and ONNX inference
deployment/evaluate_all.py   paired 30-seed, ten-policy benchmark
deployment/evaluate_training_v2.py split-locked selection/final evaluator
deployment/evaluate_training_v3.py split-locked ensemble promotion evaluator
deployment/ablation_study.py retrain-and-evaluate QMIX ablations
deployment/generate_plots.py 13 data-derived scientific figures
deployment/build_final_report.py artifact-derived final report
results/sanity/              gate evidence
results/experiments/         configs, validation decisions, checkpoints, logs
results/learned_models/      selected full-policy ONNX exports
results/ablation_models/     selected retrained-ablation ONNX exports
results/final/               raw, per-seed, summary, statistical, Q, and trajectory data
results/ablations/           ablation data and paired comparisons
results/training_v2/         frozen selection and final continuation results
results/training_v3/         frozen ensemble selection, final results, and audit
results/figures/             final plots and source manifest
results/environment_drift.json        post-hoc source-repair disclosure record
results/environment_drift_impact.json measured effect of the feasibility repair
results/provenance.json      source and primary-artifact SHA-256 archive
hardware_eval/               graph accounting and provisional hardware estimates (not board validation)
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

## Split-locked extended training

The `extended` profile changes only the training horizon. It warm-starts from validation-selected `improved` checkpoints, preserves the architecture and optimizer settings, and revalidates checkpoints rather than assuming the terminal model is best. Online weights and optimizer moments are restored and both target networks are bootstrapped from the restored online weights; the replay buffer, RNG progress, and learner counters are not restored, so this is not a bitwise continuation:

```bash
venv/bin/python training/train_all.py --profile extended --resume-profile improved --alg qmix --seeds 101,102,103 --regime coordinated --scenario volatile
venv/bin/python deployment/evaluate_training_v2.py --split selection --scenario volatile
```

Selection seeds `211`--`230` were used for the predeclared all-three-replicas promotion rule. The final seeds `3001`--`3030` were evaluated exactly once after model and analysis fingerprints were frozen. That final split is now consumed and must not be rerun or used for further model selection.

## Split-locked deployment ensemble

The deployment-only v3 experiment compares majority and unanimous votes across the three Extended QMIX recurrent policies:

```bash
venv/bin/python deployment/evaluate_training_v3.py --split selection
venv/bin/python deployment/audit_training_v3.py
```

Selection seeds `231`--`250` are development-only. Final seeds `4001`--`4030` were evaluated once after unanimous voting was promoted and model/analysis hashes were frozen. They are now consumed; the final command is intentionally omitted from the reproduction workflow to prevent accidental reuse.

## Source drift and what the corrected low-LR experiment now requires

Two repairs postdate every artifact in this repository: the sample-feasibility rule in `environment/single_agent_env.py` and the resumed-learner target-mixer and learning-rate handling in `training/epymarl/src/learners/q_learner.py`. Both are retained. Their scope, evidence, and measured effect are recorded in [results/environment_drift.json](results/environment_drift.json), and the impact measurement is reproducible:

```bash
venv/bin/python deployment/measure_environment_drift.py
venv/bin/python deployment/audit_final_artifacts.py
```

The audit now recomputes the environment digest instead of trusting the stored gate artifact, and fails if any manifest carries a pre-repair digest that the drift record does not declare. Training manifests additionally record `training_source_sha256` over the learner, controller, runner, agent, mixer, launcher, profile, and gate sources, because a `git_sha` taken from a disclosed dirty worktree cannot identify the code that actually ran.

One consequence is load-bearing for future work. `train_all.py` resolves a resume checkpoint only among manifest entries whose `config_digest` matches the current gate, and the Extended manifests record the pre-repair digest. A `refined` warm-start from Extended therefore raises `RuntimeError` for all three seeds. That guard is correct: it refuses to continue training across an environment change. A corrected low-learning-rate experiment consequently needs more than fresh selection and final seeds. It needs Extended checkpoints retrained under the repaired environment, or an explicit, separately recorded decision to warm-start across the repair. The guard must not be bypassed silently to make the resume succeed.
