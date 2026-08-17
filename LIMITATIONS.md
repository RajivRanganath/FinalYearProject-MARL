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

Three training seeds meet the requested minimum but provide limited evidence about optimization variance. The 60,000-step budget, recurrent 64-unit architecture, optimizer settings, and validation-reward selection rule are a finite design choice rather than a proof of convergence or optimality.

Checkpoint selection uses ten validation seeds (`201`–`210`) and never uses the locked test seeds (`1001`–`1030`). This avoids direct test leakage but does not eliminate researcher degrees of freedom in earlier environment and reward design.

IQL, VDN, and QMIX use parameter-shared recurrent agents. Results do not extend automatically to graph neural networks, actor-critic methods, offline RL, model-based control, longer training, or hyperparameter sweeps.

## Statistics

Confidence intervals use 30 held-out environment seeds as sampling units after averaging the three learned training replicas within each environment seed. Paired t-tests assume the paired differences are approximately normal. Multiple policies and metrics are explored, so unadjusted p-values should not be treated as a family-wise confirmatory analysis.

The 5% practical threshold is an explicit reporting convention, not a stakeholder-validated minimum clinically or economically important difference. Statistical support and practical engineering relevance are reported separately.

Simulation seeds create paired stochastic worlds, not independent real deployments. Generalization across seasons, sites, hardware, and event distributions remains untested.

## Ablations

Each ablation is retrained with three seeds and scored in the common full coordinated evaluation environment. This is stronger than applying a frozen policy to a modified reward, but an ablation can change optimization difficulty as well as remove information or inductive bias. A null ablation difference does not prove that a component is universally useless, and a difference does not by itself identify a unique causal mechanism.

## TinyML tooling boundary

The separate `hardware_eval/` utilities measure model dimensions and host-runtime properties but estimate several microcontroller latency and energy quantities from device specifications. Those estimates are not measurements on target silicon and should not be called cycle-accurate hardware validation without board-level power and timing experiments.

The current recurrent ONNX models are larger than older legacy MLP artifacts in this repository. Hardware conclusions derived from an old parameter count or model file must be regenerated against the selected final ONNX exports.

## Reproducibility state

Run manifests record the base Git SHA and that the worktree was dirty during the final experiments. They also save exact commands, configurations, validation decisions, and output paths. A publication archive should include the complete diff or a commit containing these changes before claiming bit-for-bit code provenance.
