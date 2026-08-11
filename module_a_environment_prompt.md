# Module A — Environment and Simulation Core

Paste the master prompt first, then this.

## Your Role

You are responsible for building the simulated world that the IoT sensor agents live in. You are not building any AI or neural network code, you are building the physics, rules, and dynamics of the environment itself.

## Goal

Build a Dec POMDP, Decentralized Partially Observable Markov Decision Process, simulation of 4 solar powered IoT sensor nodes, each independently deciding whether to Sample Now or Sleep at each timestep, under strict energy harvesting constraints, wrapped as a PettingZoo ParallelEnv so Module B can plug any MARL algorithm into it without custom glue code.

## Phase 1, Single Agent Foundation

Before building the multi agent version, build and fully test a single agent environment first. This single sensor should have:

**Battery state**: a float between 0 and 1 representing residual energy.

**Harvesting model**: battery increases each timestep based on a solar harvesting function. Do not use a flat constant rate, model something closer to reality, such as a day and night cycle (zero harvesting at night, a bell curve peak around midday) with some added randomness to represent cloud cover or weather variability. This randomness matters since the project's core argument is that MARL adapts to unpredictable conditions better than a fixed rule, so the environment needs genuine unpredictability built in.

**Data entropy generator**: a float between 0 and 1 representing how volatile or interesting the sensor's local data currently is. Model this as a signal that spends most of its time low (boring, stable periods) with occasional spikes (an event worth capturing). This is what creates the tension the agent has to learn to navigate, sample too rarely and you miss the spike, sample too often and you waste energy during boring periods.

**Energy causality enforcement**: the Sample action must be rejected or have no effect if the battery does not have enough energy to cover it. A sensor is never allowed to go into energy debt. Make this a hard constraint in the step function, not something the reward function just discourages.

**Reward function**: give positive reward for sampling during a genuine high entropy event, a penalty for sampling during a low entropy period since that is wasted energy, and a penalty for missing a high entropy event by sleeping through it. Make sure the reward function correctly reflects the tradeoff the project claims to solve, wasted energy versus stale data.

Test this single agent environment with a random policy and with a fixed interval policy (sample every N steps regardless of state), and confirm battery levels and rewards behave sensibly across a full episode before moving to the multi agent version.

## Phase 2, Multi Agent Extension

Extend to 4 agents, each with its own independent battery and entropy signal. This is where partial observability has to be implemented carefully, since it is central to the project's claims:

- Each agent's observation must contain only its own `residual_energy`, its own `data_entropy`, and a `neighbor_sampling_rate` value, in that exact order, matching the shared interface contract.
- `neighbor_sampling_rate` should reflect how often nearby agents have sampled recently, for example a rolling average over the last several timesteps, but it must NOT leak neighbor battery levels or neighbor entropy values to the observing agent. Agents can coordinate loosely through this one shared signal, but they cannot see each other's full state. If you accidentally leak more information than this, the project's core claim of decentralized, partially observable coordination breaks.
- Reward should have both an individual component (did this agent make a good sample or sleep decision) and shared benefit from avoiding redundant simultaneous sampling across neighbors, since avoiding neighbor overlap is one of the stated goals in the project's state space design.

## Phase 3, PettingZoo Wrapping

Wrap the final multi agent environment in the PettingZoo ParallelEnv API. `reset()` must return a dict of observations keyed by agent id. `step(actions)` must accept a dict of actions keyed by agent id and return observations, rewards, terminations, truncations, and infos, all keyed by agent id. Use the exact state vector order and action space defined in the shared interface contract, do not improvise a different shape.

## Phase 4, Scenario Variants

Build at least 2 distinct environment configurations for testing generalization, since the panel Q&A material claims MARL adapts to changing conditions and this needs supporting evidence:

- A **stable scenario**, low entropy volatility, predictable harvesting.
- A **volatile scenario**, frequent entropy spikes, less predictable harvesting (more simulated cloud cover variability).

These should be selectable via the shared config file, not hardcoded separately, so Module B can train and evaluate against both.

## Deliverables

1. Single agent environment, tested and documented, in the `environment/` folder
2. Multi agent, 4 agent, PettingZoo compatible environment
3. Two scenario configurations, stable and volatile
4. A short results file showing reward and battery behavior under a random policy and a fixed interval policy, across both scenarios, to serve as baseline evidence before Module B's trained policy is compared against it
5. Full documentation of every parameter (harvesting curve shape, entropy spike frequency, reward weights) so Module B can tune training effectively and Module C understands the environment assumptions behind any energy estimates

If Module B reports that training is not converging or behaving oddly, be prepared to review whether the reward function or the harvesting model has an unintended bias, this is a common source of bugs in energy harvesting RL environments.
