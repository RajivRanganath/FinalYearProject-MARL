# Module B — MARL Algorithm and Training

Paste the master prompt first, then this.

## Your Role

You are responsible for building the cooperative multi agent reinforcement learning pipeline that decides, for each of 4 IoT sensor agents, whether to Sample Now or Sleep at every timestep.

## Goal

Train a policy that lets 4 agents cooperatively minimize wasted energy while keeping Age of Information low, using only local observations (`residual_energy`, `data_entropy`, `neighbor_sampling_rate`), with no agent having access to the global state of the system. The policy must outperform both a fixed interval baseline and a simple rule based baseline, and must be small enough to realistically run on a microcontroller once exported.

## Phase 1, Framework Setup and Sanity Check

Install EPyMARL (preferred, since it has QMIX and VDN implemented and maintained) or PyMARL2 as a fallback if EPyMARL has dependency conflicts in your environment. Before touching the project's actual environment, run one of the framework's built in toy examples end to end and confirm training runs without errors and reward improves over time. This confirms your local setup, GPU or CPU configuration, and dependencies are correct before you introduce any project specific complexity.

## Phase 2, Mock Environment

Do not wait for Module A's environment to be finished. Build a mock environment now that exactly matches the shared interface contract:

- 4 agents
- Observation per agent: `[residual_energy, data_entropy, neighbor_sampling_rate]`, each a float normalized 0 to 1, in that exact order
- Action space per agent: discrete, 2 values, 0 for Sleep or Wait, 1 for Sample Now
- PettingZoo ParallelEnv API, `reset()` and `step()` with dict keyed by agent id

The mock can return random observations and a simple placeholder reward function (for example, small positive reward for sampling when a random entropy value is high, small negative reward otherwise). Its only purpose is to let you build, run, and debug the full training loop without blocking on Module A. Document clearly in your code that this is a mock and must be swapped out later.

## Phase 3, Implement VDN First

Value Decomposition Networks are your starting point because the architecture is simpler to reason about and debug. VDN assumes the team Q value is just the sum of each individual agent's Q value, which makes credit assignment straightforward, if the team does well, every agent's individual Q value gets nudged up proportionally.

Requirements:
- Each agent has its own local Q network, but weights can be shared across agents since all 4 sensors face structurally the same problem, this reduces parameters and training time
- Use an epsilon greedy exploration schedule, decaying over training, so agents explore Sample versus Sleep sufficiently early on rather than collapsing to a fixed policy too fast
- Train against the mock environment first, confirm reward improves and the policy is not just collapsing to always Sleep or always Sample, both are common failure modes in energy penalized reward functions and need to be checked for explicitly

## Phase 4, Implement QMIX

QMIX is the primary algorithm the project is built around and needs to be implemented properly, not just used as a black box from the library.

Requirements:
- Understand and be able to explain the mixing network, it takes each agent's individual Q value as input and combines them into a single team Q value, using weights that are constrained to be non negative, this constraint is what guarantees that improving any individual agent's Q value never decreases the team Q value, which is what makes the whole approach mathematically sound for cooperative credit assignment
- The mixing network itself is typically conditioned on global state during centralized training, meaning it can see everything, this is allowed and expected under CTDE, only the individual agent policies need to be decentralized at execution time, the mixing network is discarded after training and is not part of what eventually runs on hardware
- Add clear code comments explaining this training versus execution asymmetry, since it is a common point of confusion when the project is explained to a panel
- Compare QMIX's convergence and final performance against your VDN baseline, QMIX should generally do at least as well or better given its more expressive mixing function, if it does not, investigate hyperparameters before assuming the environment or reward function is the issue

## Phase 5, Reward and Hyperparameter Tuning

- Track episode reward, energy waste percentage, and Age of Information stability across training, not just raw reward, since raw reward alone can hide whether the policy is actually solving the problem the project claims to solve
- If training is unstable or not converging, systematically check learning rate, batch size, replay buffer size, and target network update frequency before assuming a fundamental issue, these are the most common causes of QMIX instability
- If you suspect the environment or reward function itself has an issue, such as unintended bias or a degenerate optimum like always sleeping, flag this to whoever owns Module A rather than trying to work around it silently in the training code, since the fix likely belongs in the environment, not the training script

## Phase 6, Swap in the Real Environment

Once Module A delivers their PettingZoo environment (single scenario first, then both stable and volatile scenarios once available), retrain both VDN and QMIX against it. Track and report, at minimum:

- Episode reward over training steps, for both algorithms
- Percentage of energy saved compared to a fixed interval sampling baseline (sample every N timesteps regardless of state, N should be configurable and documented)
- Percentage of energy saved compared to a simple rule based baseline (if battery below 20 percent, sleep, otherwise sample), this is the baseline referenced in the project's panel Q&A material claiming an 8.6 percent improvement, so this comparison needs to be run properly and the actual measured number reported honestly, even if it differs from the number in the original script
- Age of Information stability across an episode, for both the trained policy and both baselines
- Performance under both the stable and volatile scenarios from Module A, since the project's core claim is that MARL adapts to changing conditions better than static rules, this needs to be demonstrated with actual numbers, not just asserted

## Phase 7, Model Size Awareness

Throughout training, keep the eventual hardware constraint in mind. A rough guideline, if the network has more than a few hundred thousand parameters, it is likely too large for a microcontroller class device and will need to be pruned down or redesigned with smaller hidden layers. Do not wait until export to discover this, check parameter count early and periodically during development, and if the architecture needs to shrink, prefer reducing hidden layer width over removing layers entirely, since depth often matters more than width for policy expressiveness in small state spaces like this one.

## Phase 8, Export

Export the final trained policy network, meaning the individual agent policy network, not the mixing network, since the mixing network is training only and never runs on hardware, to ONNX format. Verify the exported model produces the same outputs as the original PyTorch model on a handful of test inputs, to confirm nothing broke in the export process.

Document alongside the export:
- Full architecture, number of layers, layer sizes, activation functions
- Total parameter count
- Input shape and expected input normalization (confirm inputs are expected in the 0 to 1 normalized range matching the shared interface contract)
- Output shape and how to interpret it, meaning how the raw output maps to the Sample or Sleep decision, argmax, threshold, or otherwise

This documentation is not optional, Module C cannot estimate memory footprint or latency accurately without it.

## Deliverables

1. Working VDN and QMIX training code in the `training/` folder, clearly separated and both runnable independently
2. Training curves showing convergence for both algorithms, against the mock environment initially and the real environment once available
3. A comparison table showing energy savings and Age of Information stability for the trained QMIX policy against the fixed interval baseline and the rule based baseline, across both the stable and volatile scenarios
4. ONNX exported policy file, verified to match the original model's outputs
5. Architecture documentation file for Module C, including parameter count and input and output specifications
6. A brief written summary of results in plain language, suitable for direct inclusion in a project report or a panel presentation slide, including an honest statement of the actual measured improvement percentage over baselines, whatever that number turns out to be

If parameter count is too high for any candidate hardware device once Module C reports back, be prepared to retrain a smaller version of the network rather than treating the original as final, this handoff and possible retraining loop between Module B and Module C should be expected as normal, not treated as a failure.
