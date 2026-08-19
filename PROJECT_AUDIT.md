# Comprehensive Project Audit: MARL Adaptive IoT Energy-Harvesting & TinyML Evaluation

> Historical snapshot: this audit predates the current recurrent-policy export,
> split-locked training upgrades, and repaired hardware tooling. Use README.md,
> FINAL_RESEARCH_REPORT.md, LIMITATIONS.md, and the current audit scripts for the
> present repository state.

**Repository:** `FinalYearProject-MARL`  
**Audit Date:** August 17, 2026  
**Auditor Role:** Senior MARL, IoT Energy-Harvesting & TinyML Research Engineer  

---

## 1. Executive Summary

This audit reviews every component of the codebase—from single-agent physical simulation, multi-agent PettingZoo wrappers, and EPyMARL training pipelines, to ONNX model export, baseline benchmarking, and TinyML hardware feasibility evaluation.

While the core conceptual framework is sound, the audit identified critical flaws in:
1. **Temporal ordering of environment state transitions and rewards** (Markov property violation).
2. **Ambiguous and conflicting input/observation contract definitions** across modules.
3. **Absence of Action Feasibility Masking** leading to sample rejection loops.
4. **Complete absence of executable code in Module C** (Hardware Evaluation).
5. **Disconnection between training reward shaping and evaluation metrics**.
6. **Lack of physical units** grounding the simulation in real-world Joules, mW, and mAh.

---

## 2. Detailed Findings by Severity

### 🔴 CRITICAL FINDINGS

#### [CRIT-01] Temporal Markov Alignment Bug in Environment Step Functions
* **Locations:** `environment/single_agent_env.py` (lines 203–220), `environment/multi_agent_env.py` (line 181)
* **Root Cause:** In `SingleAgentSensorEnv.step(action)`, the method executes the harvesting/sampling energy update, then immediately calls `_update_data_entropy()` generating $\text{entropy}_{t+1}$, and then calculates the step reward using $\text{entropy}_{t+1}$ rather than the observation $\text{entropy}_t$ upon which the agent made its decision.
* **Impact:** Agents that correctly sampled a high-entropy spike at $t$ were penalized for wasting energy if $\text{entropy}_{t+1}$ decayed back to baseline. Conversely, agents sleeping during low entropy at $t$ were penalized if a new spike spawned at $t+1$.
* **Required Fix:** Restructure `step()` to evaluate action $a_t$ against state $s_t = (\text{battery}_t, \text{entropy}_t)$, compute reward $R(s_t, a_t)$, and subsequently transition to $s_{t+1} = (\text{battery}_{t+1}, \text{entropy}_{t+1})$.

#### [CRIT-02] Observation vs. Model Input Contract Inconsistency
* **Locations:** `shared_config.py`, `training/export_onnx.py`, `training/verify_onnx.py`, `hardware_eval/architecture.md`, `deployment/simulate_deployment.py`
* **Root Cause:** `shared_config.py` defines `STATE_DIM = 3` (`[residual_energy, data_entropy, neighbor_sampling_rate]`). However, EPyMARL appends a 4-dimensional one-hot agent ID, making the neural network input tensor dimension 7 (`MODEL_INPUT_DIM = 7`). `hardware_eval/architecture.md` states the input is 3, while `export_onnx.py` builds an input shape of 7. Furthermore, `verify_onnx.py` hardcodes `hidden_dim = 64` while `export_onnx.py` uses `hidden_dim = 128`.
* **Impact:** Module C cannot profile the model correctly, and documentation contradicts the exported ONNX graph.
* **Required Fix:** Formally standardize constants in `shared_config.py`: `ENV_OBS_DIM = 3`, `NUM_AGENTS = 4`, `NUM_ACTIONS = 2`, `MODEL_INPUT_DIM = 7` (or 3 for decentralized models without ID concatenation), and `HIDDEN_DIM = 128`. Add automated contract validation assertions.

#### [CRIT-03] Module C (Hardware Profiling) Is Completely Unimplemented
* **Locations:** `hardware_eval/`
* **Root Cause:** `hardware_eval/` contains only a stub `architecture.md` file (which ends with a syntax typo) and zero executable Python scripts.
* **Impact:** The project's core claim of a comparative TinyML hardware evaluation framework across ESP32, Bharat Pi, RP2040, and Arduino Nano 33 BLE Sense is currently unsupported by executable code.
* **Required Fix:** Implement complete, reproducible scripts: `device_specs.json`, `quantize_model.py`, `model_analysis.py`, `memory_estimator.py`, `latency_estimator.py`, `energy_estimator.py`, `rank_devices.py`, and `final_hardware_report.md`.

#### [CRIT-04] Action Feasibility Masking Is Disabled
* **Locations:** `training/env_wrapper.py` (lines 151–157)
* **Root Cause:** `IoTSensorEnvWrapper.get_avail_agent_actions(agent_id)` returns `[1, 1]` unconditionally for all states, even when residual energy is below the sampling threshold ($E_{battery} < E_{cost}$).
* **Impact:** The Q-learning learner is forced to learn feasibility via penalties rather than structural masking, causing agents to repeatedly request impossible samples and accumulate rejection penalties.
* **Required Fix:** Implement action availability masking: return `[1, 1]` when $E_{battery} \ge E_{cost}$ and `[1, 0]` when $E_{battery} < E_{cost}$.

---

### 🟠 HIGH FINDINGS

#### [HIGH-01] Absence of Physical Units in Energy & Solar Model
* **Locations:** `environment/single_agent_env.py`
* **Root Cause:** Energy parameters are arbitrary dimensionless floats ($E_{sample} = 0.05$, $P_{max} = 0.08$).
* **Impact:** Findings cannot be defended in an engineering viva regarding real-world battery size (e.g., 500 mAh LiPo, 3.7V = 6660 mJ), sensor active power (e.g., 50 mA for 100 ms = 18.5 mJ), sleep current (e.g., 15 µA), and solar panel output (e.g., 50x50 mm solar cell yielding 10–50 mW).
* **Required Fix:** Create a parameterized energy model with explicitly documented physical units and derivations mapping physical Joules to the normalized $[0, 1]$ interval.

#### [HIGH-02] Solar Cloud Attenuation Lacks Temporal Autocorrelation
* **Locations:** `environment/single_agent_env.py` (line 111)
* **Root Cause:** Cloud noise is modeled as independent Gaussian noise $\mathcal{N}(0, \sigma^2)$ at every 5-minute timestep.
* **Impact:** Produces unrealistic high-frequency flickering rather than realistic weather patterns (cloud fronts lasting 30–120 minutes).
* **Required Fix:** Implement an autoregressive AR(1) cloud attenuation model: $c_{t} = \alpha c_{t-1} + (1-\alpha)\mu + \epsilon_t$.

#### [HIGH-03] Data Entropy Model Lacks Multi-Step Event Persistence
* **Locations:** `environment/single_agent_env.py` (lines 118–135)
* **Root Cause:** High-entropy spikes are generated as single-step impulses that immediately decay.
* **Impact:** Environmental events in real IoT deployments (e.g., structural vibrations, temperature surges, agricultural pests) persist over multiple timesteps.
* **Required Fix:** Implement a Poisson/Markov burst generator supporting multi-step event durations and spatial correlation across neighboring nodes.

#### [HIGH-04] Age of Information (AoI) Not Formally Integrated as an Objective
* **Locations:** `environment/multi_agent_env.py`, `deployment/simulate_deployment.py`
* **Root Cause:** AoI is only calculated as an evaluation metric in `simulate_deployment.py` and is omitted from the environment state and reward formulation.
* **Impact:** The policy cannot directly optimize for freshness bounds ($p95$ AoI, max staleness).
* **Required Fix:** Formally integrate AoI tracking into the environment and include a normalized AoI freshness term in the canonical reward function.

#### [HIGH-05] Training Reward Shaping vs. Deployment Evaluation Disconnect
* **Locations:** `training/env_wrapper.py` (lines 52–89)
* **Root Cause:** The environment wrapper introduced custom training-only reward shaping offsets that differed from the environment's actual reward return.
* **Impact:** The RL algorithm optimizes a surrogate reward function that does not match the evaluation benchmark.
* **Required Fix:** Unify the reward function into a single, canonical, mathematically grounded formulation used consistently in both training and evaluation.

---

### 🟡 MEDIUM FINDINGS

#### [MED-01] Cherry-Picked Seed Evaluation & Inadequate Sample Size
* **Locations:** `deployment/simulate_deployment.py`
* **Root Cause:** Evaluation was executed across only 5 seeds (42–46).
* **Impact:** 5 seeds are insufficient to establish statistical significance or compute 95% confidence intervals.
* **Required Fix:** Establish a rigorous evaluation protocol with a fixed split of 30+ independent held-out test seeds, reporting mean $\pm$ standard deviation and 95% confidence intervals with paired statistical tests.

#### [MED-02] Incomplete Multi-Algorithm Benchmark Comparison
* **Locations:** `training/train_qmix.py`, `training/train_vdn.py`
* **Root Cause:** `train_qmix.py` was invoking `iql` under the hood. Head-to-head empirical comparisons between IQL, VDN, and true QMIX across identical training budgets and seeds were not systematically completed.
* **Impact:** Cannot rigorously answer RQ2 ("Does QMIX outperform independent and simpler cooperative methods?").
* **Required Fix:** Run systematic multi-seed training and evaluation for IQL, VDN, and QMIX.

#### [MED-03] Missing Quantitative Ablation Studies
* **Locations:** Repository-wide
* **Root Cause:** No empirical ablation experiments exist to test the individual utility of the neighbor sampling rate, coordination penalty, AoI penalty, and network recurrence.
* **Impact:** Claims regarding the benefit of partial observability and neighbor rate coordination lack experimental backing.
* **Required Fix:** Execute controlled ablation runs and record results in `results/ablation_results.csv` and `results/ablation_report.md`.

#### [MED-04] Absence of a Formal Automated PyTest Suite
* **Locations:** Root repository
* **Root Cause:** Tests exist as scattered ad-hoc `if __name__ == "__main__":` blocks inside environment files rather than a structured `tests/` directory.
* **Impact:** Regressions cannot be prevented automatically during refactoring or CI execution.
* **Required Fix:** Create a complete `tests/` suite covering PettingZoo API compliance, observation shapes, action availability, battery causality, reward alignment, and ONNX numerical parity.

---

### 🔵 LOW FINDINGS

#### [LOW-01] Missing Version Pinning in `requirements.txt`
* **Locations:** `requirements.txt`
* **Root Cause:** Dependency versions are unpinned, risking environment drift across operating systems.
* **Impact:** Fresh installations on Windows or Linux could install incompatible PyTorch/ONNX versions.
* **Required Fix:** Pin all core dependencies (`torch`, `onnx`, `onnxruntime`, `pettingzoo`, `gymnasium`, `numpy`, `scipy`, `pandas`, `matplotlib`, `pytest`).

#### [LOW-02] Stale & Inconsistent Markdown Documentation
* **Locations:** `README.md`, `training/architecture.md`, `hardware_eval/architecture.md`
* **Root Cause:** Markdown documentation contains outdated claims and contradicts actual code parameters.
* **Impact:** Undermines project credibility during academic examination.
* **Required Fix:** Rewrite all documentation directly from verified code outputs and automated evaluation logs.

---

## 3. Component Status Matrix

| Component | Status | Code Present | Verified Working | Tests Exist | Action Required |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Shared Config & Interfaces** | 🟡 Inconsistent | Yes | Partial | No | Standardize dimensions & add contract assertions |
| **Single-Agent Env Physics** | 🔴 Buggy | Yes | Yes (with bug) | Partial | Fix temporal Markov bug & add physical energy model |
| **Multi-Agent Env Dec-POMDP** | 🟡 Incomplete | Yes | Partial | Partial | Add spatial topology & multi-step burst generator |
| **PettingZoo ParallelEnv** | 🟢 Functional | Yes | Yes | Partial | Run official PettingZoo API test harness |
| **EPyMARL Integration** | 🟡 Needs Tuning | Yes | Yes | No | Enable action masking & unify reward formulation |
| **IQL Training Pipeline** | 🟢 Functional | Yes | Yes | No | Multi-seed training with validation checkpointing |
| **VDN Training Pipeline** | 🟡 Needs Verification | Yes | Untested | No | Run multi-seed benchmark |
| **QMIX Training Pipeline** | 🟡 Needs Verification | Yes | Untested | No | Run true QMIX multi-seed benchmark |
| **ONNX Policy Export** | 🟢 Functional | Yes | Yes | Partial | Document exact graph & verify numerical precision |
| **Quantization Pipeline** | 🔴 Missing | No | No | No | Implement Int8 post-training quantization |
| **Module C Hardware Profiling**| 🔴 Missing | No | No | No | Implement full hardware feasibility engine |
| **Baseline Evaluation Suite** | 🟡 Insufficient | Yes | Partial | No | Expand to 10 baselines across 30+ held-out seeds |
| **Ablation Studies** | 🔴 Missing | No | No | No | Run neighbor rate, AoI, and redundancy ablations |
| **Documentation & README** | 🟡 Inconsistent | Yes | Stale | N/A | Rewrite from automated experimental data |

---

## 4. Remediation Roadmap

1. **Stage 1**: Contract Standardization & Physical Energy Formulation (`shared_config.py`, `environment/energy_model.py`).
2. **Stage 2**: Environment Repair & Statistical Scenario Validation (`single_agent_env.py`, `multi_agent_env.py`, `pettingzoo_env.py`).
3. **Stage 3**: Automated Test Suite & PettingZoo Compliance (`tests/`).
4. **Stage 4**: Multi-Algorithm Training & Validation Checkpointing (IQL, VDN, QMIX across independent training seeds).
5. **Stage 5**: Comprehensive Evaluation Pipeline & Baselines (10 baselines, 30+ test seeds, Pareto analysis).
6. **Stage 6**: Module C Implementation (Device DB, Quantization, Memory, Latency, Energy profiling).
7. **Stage 7**: Ablation Studies, Professional Figures, Limitations & Final Documentation.
