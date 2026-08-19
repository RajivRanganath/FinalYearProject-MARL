# Engineering and Scientific Limitations

These boundaries apply to the final independent/coordinated volatile-scenario study. They are part of the result, not caveats to hide after reporting it.

## Scope of the empirical claim

The experiments cover four agents, one ring topology, one 24-hour volatile scenario, three independent training seeds per algorithm, and 30 held-out simulation seeds. They do not establish that QMIX is generally inferior to heuristics, nor that cooperative MARL cannot help larger or more complex networks. The supported result is narrower: no engineering advantage over the strongest causal proxy heuristic was observed under the evaluated conditions.

Only simulation performance is measured. No field sensor network, radio stack, solar installation, or target microcontroller executes the final policy in this study.

## Event observability and proxy model

True measurement entropy is latent before the decision and is measured only after SAMPLE. Policies instead receive a noisy low-power event proxy. Its false-positive rate, false-negative rate, noise level, and monitoring cost are modeling assumptions rather than values calibrated from a named physical sensor. Results may change materially with a weaker, delayed, drifting, or more expensive detector.

The benchmark name `Entropy Threshold` is retained to match the requested comparison list, but the implementation thresholds this causal event proxy. It must not be described as a threshold on pre-observed latent entropy.

## Network and communication model

The coordinated regime uses a two-packet shared channel with deterministic rotating priority, a ring neighborhood, lagged neighbor sampling rates, persistent spatially correlated events, redundancy costs, and coverage utility. It omits path loss, fading, capture effects, retransmission, CSMA/CA backoff, queueing, packet corruption, clock drift, and explicit neighbor-message energy. Neighbor activity is received without loss or delay beyond the defined lag.

The channel, redundancy, and coverage mechanisms create genuine joint dependence, but the four-agent task remains small and may be solvable by a well-aligned local rule. More complex coordination should be a new pre-registered experiment rather than a post-hoc mechanism added to make MARL win.

## Energy and sensing model

Battery state uses linear state-of-charge integration and nominal voltage. It omits nonlinear discharge curves, internal resistance, temperature effects, aging, leakage variation, and voltage sag under burst current.

Solar harvest uses an idealized time-of-day envelope with stochastic cloud attenuation. It omits seasonal sun geometry, panel orientation, shading, MPPT behavior, conversion loss, and weather data calibration. A time-of-day harvest forecast is available to policies; realized future harvest is not.

The environment advances in five-minute steps. Sub-second phenomena would require a separate wake-up interrupt or continuous-time sensing layer.

The normalized simulation reservoir deliberately accelerates charge/discharge dynamics within a one-day episode: a normalized SAMPLE cost of `0.05` is not a direct conversion of the approximately 14.5 mJ component estimate into a full 500 mAh cell. Consequently, benchmark energy is reported in normalized simulation units and must not be presented as measured joules or battery lifetime.

## Reward and metric dependence

The objective is a scalarization of capture, miss, sample energy, AoI, rejection, redundancy, contention, and coverage terms. The weights encode one engineering preference and are not fitted to deployment stakeholder utilities. A different safety or maintenance objective can change policy rankings.

Raw reward is comparable across policies within a regime because every benchmark policy uses the same environment and objective. Raw reward should not be compared naively across regimes because event processes and coordination components differ. Recall, energy, AoI, and other physical metrics should accompany every reward claim.

## Training and model selection

Three training seeds meet the requested minimum but provide limited evidence about optimization variance. The original study used a 60,000-step budget; the post-report coordinated-QMIX experiments used 180,000 steps and then a one-factor continuation to a 360,000-step horizon. The recurrent 64-unit architecture, optimizer settings, and validation-reward selection rule remain finite design choices rather than proof of convergence or optimality.

Checkpoint selection uses ten validation seeds (`201`–`210`) and never uses the locked test seeds (`1001`–`1030`). This avoids direct test leakage but does not eliminate researcher degrees of freedom in earlier environment and reward design.

The continuation experiment used separate selection seeds (`211`–`230`) and an untouched final split (`3001`–`3030`). The final split was evaluated once only after the model and analysis hashes were frozen. It is now consumed and cannot support additional tuning, model selection, or an independent future confirmation claim.

The deployment-ensemble experiment used development seeds `231`–`250` and a further untouched final split `4001`–`4030`. That final split has also been evaluated once and is consumed. The statistically unresolved reward difference from the threshold is not an equivalence or non-inferiority result; a prospectively powered test with an explicit engineering margin would be required for either claim.

The v3 confidence intervals vary environment seeds while conditioning on the same selected set of three trained QMIX replicas. They test generalization across simulated worlds, not stability across independently trained three-model ensembles. More training seeds would be needed to quantify that second source of uncertainty.

IQL, VDN, and QMIX use parameter-shared recurrent agents. Results do not extend automatically to graph neural networks, actor-critic methods, offline RL, model-based control, longer training, or hyperparameter sweeps.

## Statistics

Confidence intervals use 30 held-out environment seeds as sampling units after averaging the three learned training replicas within each environment seed. Paired t-tests assume the paired differences are approximately normal. Multiple policies and metrics are explored, so unadjusted p-values should not be treated as a family-wise confirmatory analysis.

The split-locked continuation analysis additionally reports Holm-adjusted p-values over its 12 predeclared comparisons and a two-way bootstrap that resamples both training and environment seeds. With only three training replicas, that bootstrap is still a coarse characterization of optimization uncertainty and should not be read as population-level evidence over arbitrary training runs.

The historical v3 screen tried two ensemble voting rules and reported a Holm-adjusted reward p-value, but its promotion predicate did not require that adjusted value. The selected unanimous rule nevertheless passes the corrected guard (`p_holm = 6.20e-05`), so the frozen selection outcome is unchanged. Future code now requires `p_holm <= 0.05`; this must not be retroactively described as part of the original predeclared gate.

The 5% practical threshold is an explicit reporting convention, not a stakeholder-validated minimum clinically or economically important difference. Statistical support and practical engineering relevance are reported separately.

Simulation seeds create paired stochastic worlds, not independent real deployments. Generalization across seasons, sites, hardware, and event distributions remains untested.

## Ablations

Each ablation is retrained with three seeds and scored in the common full coordinated evaluation environment. This is stronger than applying a frozen policy to a modified reward, but an ablation can change optimization difficulty as well as remove information or inductive bias. A null ablation difference does not prove that a component is universally useless, and a difference does not by itself identify a unique causal mechanism.

## TinyML tooling boundary

The previous hardware report was invalid: it counted a legacy three-layer MLP instead of the selected recurrent GRU policy, and its failed quantizer silently compared the float model with itself. The repaired tooling now derives dimensions from the real ONNX graph and fails closed. A replacement host ONNX Runtime check genuinely quantized one Extended replica (2.89x serialized compression and 99.78% action agreement over 10,000 synthetic recurrent inputs), but it is neither whole-ensemble nor microcontroller validation. Memory, latency, energy, and device rankings remain analytical projections. There is still no ONNX-to-micro-runtime conversion, target build, operator-support proof, tensor-arena measurement, board timing, or power trace.

The current recurrent ONNX models are larger than older legacy MLP artifacts in this repository. Hardware conclusions derived from an old parameter count or model file must be regenerated against the selected final ONNX exports.

The unanimous deployment ensemble runs all three Extended QMIX recurrent replicas for every agent decision. It therefore uses roughly three times the model storage and inference work of one replica before accounting for orchestration overhead. Its simulation improvement must not be described as a TinyML deployment improvement until latency, RAM, flash, and energy are measured on target hardware.

## Reproducibility state

Run manifests record the base Git SHA and that the worktree was dirty during the final experiments. They also save exact commands, configurations, validation decisions, and output paths. A publication archive should include the complete diff or a commit containing these changes before claiming bit-for-bit code provenance.

The split-locked final manifest records SHA-256 fingerprints for 12 primary model files and six listed analysis/environment files. A later audit found that the list did not cover the full transitive evaluation contract (for example, the energy model and PettingZoo wrapper), so it must not be described as complete source closure. The dirty-worktree disclosure further means the repository should be committed before external archival.

The v3 selection and final manifests similarly freeze six Extended-model files and six analysis/environment files. The saved v3 audit verifies row cardinality, split disjointness, the comparison family, final reward replication, and current hash agreement.

The later Refined QMIX experiment updated all three model weights but did not use its declared `1e-4` rate: all saved optimizer states remained at `3e-4`. It also failed its predeclared promotion rule. Its locked final split remains untouched. The experiment is invalid for the low-LR hypothesis and its selection mean must not be presented as a confirmed improvement.

The current environment now requires enough battery for sampling plus same-step sleep and proxy-monitor costs. The prior `battery >= sample_cost` boundary was used by every result in this repository, not only by the frozen v2/v3 splits: the repair postdates the Phase 25--33 benchmark, the ablation suite, the training-upgrade holdout, all four split-locked iterations, and every exported ONNX policy. All of them are historical snapshot evidence, not current-code validation after this repair.

The size of that gap is measured rather than assumed. `deployment/measure_environment_drift.py` replays the real multi-agent environment under both rules on the 30 held-out seeds, using the policy that saturates the battery floor and therefore bounds how often the rules can disagree. The two rules can only differ while the battery sits inside a band 0.017% of capacity wide, which is entered on roughly 0.02--0.03% of agent-steps. Mean episode reward moves by `+0.18` (independent) and `-0.07` (coordinated), with a worst single seed of `4.20`. Against a headline Extended-versus-Improved effect of `27.67` with a bootstrap interval of `[23.27, 32.31]`, no reported comparison changes sign or loses significance. This is a reason to disclose the drift, not a reason to treat the frozen numbers as regenerable.

Two further consequences follow. First, `train_all.py` only resolves a resume checkpoint among manifest entries whose configuration digest matches the current gate, so the corrected low-learning-rate experiment can no longer warm-start from the Extended checkpoints: it needs Extended retrained under the repaired environment, or an explicitly recorded decision to warm-start across the repair. Fresh selection and final seeds alone are not sufficient. Second, because these runs were launched from a dirty worktree and the manifests recorded only a base SHA and a dirty flag, the learner source used by any historical run is not recoverable from the archive. The resumed-learner target-mixer synchronisation cannot be proved present or absent for the Extended runs; the logged TD loss shows no spike at the resume boundary where an unsynchronised target mixer would perturb targets by roughly four reward units, which is consistent with it having been present but is not proof. Manifests now record `training_source_sha256` so this ambiguity cannot recur.
