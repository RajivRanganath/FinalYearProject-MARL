# Master Prompt — MARL Adaptive IoT Sampling Project

Give this to all three coding agents (Codex, Antigravity, Claude Code) before their module specific prompt.

## Project Context, Full Detail

We are building a Multi Agent Reinforcement Learning system for adaptive IoT sampling rate control under energy harvesting constraints. This is a final year engineering project (BNMIT, Computer Science and Engineering) with three collaborators, each using their own AI coding agent and each owning one of three modules. You are helping build ONE of these three modules. Before writing any code, state clearly which module you understand yourself to be responsible for, and do not write code belonging to another module even if it seems faster to do so yourself.

## The Problem We Are Solving

Remote IoT sensors, deployed in places like agricultural fields or on bridges for structural monitoring, typically waste 60 to 80 percent of their harvested solar energy transmitting redundant data, for example reporting the same reading every minute even when nothing has changed. We want sensors that learn when data is actually worth sampling, so they can run indefinitely on harvested energy alone without battery replacement, a state called Energy Neutral Operation. The system must also avoid the opposite failure, sampling too rarely and missing genuine events, measured through a metric called Age of Information, which tracks how stale the most recent data is.

## Why Multi Agent Reinforcement Learning

A simple rule based approach, such as if battery is below 20 percent then sleep, is static and does not adapt to unpredictable solar harvesting conditions or changing data volatility. Reinforcement learning allows the system to learn a flexible policy. Multi agent specifically matters because the 4 sensor nodes should coordinate to avoid redundant simultaneous sampling, without a central coordinator, since each node only observes its own local situation. This is formally modeled as a Dec POMDP, a Decentralized Partially Observable Markov Decision Process.

## Project Phasing and Novelty, Important Context for All Three Modules

Most existing literature in this space either simulates MARL for IoT sampling without ever validating it against real hardware constraints, or assumes a single microcontroller target without justifying that choice. This project's approach and novelty is a two part claim: first, a systematic comparative hardware feasibility framework that evaluates a trained MARL policy against the memory, compute, and energy profiles of multiple candidate microcontroller platforms before committing to one, and second, physical deployment of the cooperative policy on whichever platform is empirically justified as the best fit, rather than assumed upfront. This means the project currently has three sequential phases:

- **Phase 1**, software and simulation only, covering the environment, the MARL training, and the comparative hardware evaluation framework. This is the current active phase and what all three of you are building right now.
- **Phase 2**, hardware selection, the output of Module C's evaluation framework, a ranked recommendation with justification.
- **Phase 3**, physical implementation on the selected hardware, quantizing and flashing the trained policy onto real nodes. This phase has not started and none of you should write code assuming specific physical hardware is already available.

## The Three Modules, Full Summary

**Module A, Environment and Simulation Core.** Builds the Dec POMDP simulation world, meaning the solar harvesting dynamics, the data entropy signal for each sensor, the energy causality constraint, and the reward function, wrapped as a PettingZoo ParallelEnv. Builds a single agent version first, then extends to 4 agents with partial observability correctly enforced, meaning agents can see a neighbor sampling rate signal but never each other's battery or entropy directly. Also builds at least 2 distinct scenario configurations, a stable low volatility one and a volatile high entropy spike one, since the project needs evidence that MARL adapts to changing conditions rather than just asserting it. Module A contains no learning or neural network code.

**Module B, MARL Algorithm and Training.** Builds the cooperative reinforcement learning pipeline using EPyMARL or PyMARL2, implementing VDN first as a simpler sanity check, then QMIX as the primary algorithm, using a mixing network with non negative weights to combine individual agent Q values into a team Q value under Centralized Training, Decentralized Execution. Trains against Module A's environment, or a mock of it early on that matches the same interface, and must produce measured comparisons against a fixed interval baseline and a rule based baseline, across both of Module A's scenarios. Exports the final individual agent policy, not the mixing network, to ONNX format, with full architecture documentation.

**Module C, Hardware Profiling and Evaluation.** Builds a specification database for candidate microcontroller platforms, currently ESP32, Bharat Pi (note, this is built on ESP32 silicon itself and will likely score similarly to ESP32 on raw compute and memory, its comparison value is at the board and ecosystem level, not architectural), Raspberry Pi Pico RP2040, and Arduino Nano 33 BLE Sense. Simulates int8 quantization of Module B's exported model, estimates memory footprint against each device's available SRAM, estimates inference latency from published per chip benchmarks, estimates energy cost per inference against each device's power draw and an assumed solar harvesting budget, and produces a weighted, justified ranking of which device the team should physically build with in Phase 3.

## Shared Interface Contract, Do Not Deviate Without Team Agreement

**State vector per agent, in this exact order** (every module must use this exact ordering and normalization):
1. `residual_energy`, float, normalized 0 to 1
2. `data_entropy`, float, normalized 0 to 1
3. `neighbor_sampling_rate`, float, normalized 0 to 1

**Action space per agent**: discrete, 2 values.
- `0` = Sleep or Wait
- `1` = Sample Now

**Number of agents**: 4, matching the eventual physical node count referenced throughout the project's methodology.

**Episode structure**: one episode represents one simulated day, broken into a configurable number of timesteps, default 288, meaning roughly 5 minute intervals across 24 hours. This constant lives in the shared config file only, never hardcode a different value inside an individual module.

**Environment API**: PettingZoo ParallelEnv style. `reset()` returns a dict of observations keyed by agent id. `step(actions)` accepts a dict of actions keyed by agent id and returns observations, rewards, terminations, truncations, and infos, all keyed by agent id.

**Scenario configurations**: at minimum a stable scenario and a volatile scenario, selectable through the shared config file, used consistently by Module A when building environments, Module B when training and reporting comparisons, and Module C when the eventual energy budget assumptions need to be grounded in a specific scenario.

**Model export format**: ONNX. Regardless of what framework is used for training, the final trained individual agent policy network, not the mixing network used only during centralized training, must be exportable to ONNX so Module C can consume it independently of whatever training framework produced it.

**Baselines required for comparison**, used consistently across the project:
- Fixed interval baseline, sample every N timesteps regardless of state, N configurable and documented
- Rule based baseline, if battery below 20 percent then sleep, otherwise sample

## Repo Structure

```
environment/       Module A's code
training/          Module B's code
hardware_eval/     Module C's code
shared_config.py or shared_config.yaml   constants used by all three,
  including state vector order, episode length, agent count, scenario
  definitions, and baseline parameters
README.md          setup instructions per module, and instructions for
  how to run the mock stand ins for modules that are not yet finished
```

## Rules for All Agents

1. Read `shared_config.py` or `shared_config.yaml` before writing any code that touches state shape, action shape, episode length, or scenario parameters. Never hardcode values that already exist there, and never introduce a second source of truth for a constant that belongs there.
2. If your module depends on another module that is not ready yet, build against a mock or stub that matches the interface contract exactly, not a simplified or approximate version of it. This is what allows all three modules to develop in parallel rather than waiting on each other. Clearly label mock code as a mock in comments, so it is obvious what needs to be swapped out later.
3. Document every function, every constant, and every non obvious design decision. A teammate using a different AI agent than you, and possibly with less context on the reasoning behind a choice, needs to be able to understand and modify your code without asking you to explain it verbally.
4. Do not change the shared interface contract unilaterally. If your module genuinely needs a change to the state vector, the action space, the episode structure, or a baseline definition, flag this clearly and explain the reasoning, do not silently implement a divergent version and let the mismatch surface later during integration.
5. Prefer well established libraries over custom implementations where one reasonably exists, EPyMARL for MARL training, PettingZoo for the environment API, TensorFlow Lite or ONNX Runtime tools for quantization simulation, since this is a time constrained student project with a fixed deadline, not a research lab with months to spend building infrastructure from scratch.
6. Be honest in all reported numbers. Any performance comparison, energy savings percentage, or latency estimate must be clearly labeled as either measured (from actual simulation output) or estimated (based on published specifications, since no physical hardware exists yet in this phase). Do not present an estimate as a measurement, this distinction matters for the project's academic integrity and for what can be defended honestly in front of a panel.
7. When in doubt about scope, do less rather than more. It is better to flag "this decision belongs to a different module, here is what I assumed for now" than to quietly make a decision that should have been a team discussion.

## Cross Platform Compatibility, Module A and C on Windows, Module B on Mac

This team is working across two operating systems, so every agent must follow these rules to avoid environment mismatches at integration time.

1. Never hardcode file paths with OS specific separators. Always use Python's `pathlib.Path` for any file or folder path, never a raw string with backslashes or forward slashes typed manually.
2. Use a virtual environment with a single pinned `requirements.txt` shared across the whole repo. In week 1, each collaborator must run the exact same install command in a clean virtual environment and confirm it completes without errors on their machine, before building anything on top of it. If an install fails on one OS, flag it to the team immediately rather than working around it locally.
3. Add a `.gitattributes` file at the repo root that normalizes line endings (LF) across the project, to avoid noisy diffs and broken scripts caused by Windows CRLF versus Mac LF differences.
4. Module B, training on Mac, will likely use CPU or Apple's MPS backend rather than CUDA. This is expected and fine, the policy network in this project is small enough that GPU acceleration is not required for reasonable training times. Do not add CUDA specific code that would fail to run on Mac.
5. The actual handoff artifact between Module B and the other two modules is the ONNX exported model file, which is fully platform independent. Module C on Windows will load this file with no compatibility concerns, so cross platform work is really only about local development environment consistency, not the final integration artifact.
6. Any shell scripts (for setup, running training, or evaluation) should be written in Python rather than OS specific shell scripts (`.sh` or `.bat`), so the same script runs unmodified on both Windows and Mac.

## Integration Checkpoints

There are two points where modules must sync with each other's real output rather than mocks: once Module A's environment is stable enough for Module B to train against for real, and once Module B's trained model is exported and ready for Module C to evaluate for real. Both transitions should be explicitly confirmed between collaborators, not assumed to have silently happened.

Confirm which module you are building, and confirm you have read and understood the shared interface contract above, before writing any code.
