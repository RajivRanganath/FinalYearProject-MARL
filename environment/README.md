# Module A — Environment and Simulation Core

This directory contains the complete Dec-POMDP simulation environment for the MARL Adaptive IoT Sampling project.

> [!NOTE]
> **Module A Completion Summary**:
> All 4 deliverables specified in `module_a_environment_prompt.md` are fully implemented, tested, and documented.

---

## 1. Constants & Shared Interface Contract

All core global constants are imported directly from [`shared_config.py`](../shared_config.py) at the repository root to guarantee a single source of truth across all project modules:

* **`NUM_AGENTS`** (`4`): Number of decentralized sensor nodes (`agent_0` through `agent_3`).
* **`EPISODE_LENGTH_TIMESTEPS`** (`288`): Number of timesteps per episode representing a 24-hour day in 5-minute intervals.
* **`STATE_DIM`** (`3`): Dimension of each agent's local state vector.
* **`ACTION_SLEEP`** (`0`): Discrete action value representing Sleep / Wait (no-op).
* **`ACTION_SAMPLE`** (`1`): Discrete action value representing Sample Now (sensor reading & transmission).
* **`BASELINE_FIXED_INTERVAL_N`** (`12`): Step interval for the fixed-interval baseline policy (samples once per hour).
* **`BASELINE_RULE_BATTERY_THRESHOLD`** (`0.20`): Battery threshold ($20\%$) for the rule-based baseline policy.
* **`SEED`** (`42`): Global random seed for reproducible simulation runs.

---

## 2. Partial Observability Boundary & State Vector

Each agent operates as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)** node.

### Full 3D Observation Vector

In Phase 2 & 3 multi-agent mode, the observation returned for each agent $i \in \{0, 1, 2, 3\}$ is a 3-dimensional NumPy float32 array:
$$\mathbf{s}_{t, i} = [\text{residual\_energy}_i, \text{data\_entropy}_i, \text{neighbor\_sampling\_rate}_i]$$

* **`residual_energy`** ($[0.0, 1.0]$): Agent $i$'s own battery state of charge. Initialized randomly between $0.5$ and $1.0$ at `reset()`.
* **`data_entropy`** ($[0.0, 1.0]$): Agent $i$'s own local data volatility/interest level.
* **`neighbor_sampling_rate`** ($[0.0, 1.0]$): Rolling average sampling rate of the **other 3 agents** (excluding agent $i$) over a window of $W = 12$ timesteps.

> [!CAUTION]
> **Strict Partial Observability Boundary**:
> 1. Agent $i$ can **ONLY** observe its own battery, its own entropy, and the scalar `neighbor_sampling_rate`.
> 2. Agent $i$ **NEVER** receives neighbor battery levels or neighbor data entropy values.
> 3. Disregarding or leaking neighbor internal states violates the Dec-POMDP formulation and breaks the project's core claim of decentralized decision-making under partial observability.

---

## 3. Neighbor Sampling Rate Calculation

For a target agent $i$, let $O_i = \{j \in \{0, 1, 2, 3\} \mid j \neq i\}$ be the set of the 3 neighboring agents.

Over a rolling window of $W = 12$ timesteps, the maximum possible executed sample count across all neighbors is $|O_i| \times W = 3 \times 12 = 36$.

The normalized scalar `neighbor_sampling_rate` for agent $i$ at timestep $t$ is computed as:

$$\text{neighbor\_sampling\_rate}_{t, i} = \frac{\sum_{j \in O_i} \sum_{\tau = t - W + 1}^{t} a_{j, \tau}}{3 \times W}$$

where $a_{j, \tau} \in \{0, 1\}$ is 1 if neighbor $j$ successfully executed a sample at timestep $\tau$, and 0 otherwise.

---

## 4. Multi-Agent Physical Mechanics & Scenarios

1. **Independent Solar Harvesting**: Each of the 4 nodes experiences solar daylight cycles (sunrise at step 72, peak at step 144, sunset at step 216). Additive Gaussian cloud noise $N_c \sim \mathcal{N}(0, \sigma^2)$ is generated **independently per node**, reflecting local micro-variations in shading and weather.
2. **Independent Data Entropy Spikes**: Each node monitors a distinct geographic zone. Data entropy spikes occur independently per agent with probability `spike_prob`, decaying exponentially ($\times 0.60$) per step.
3. **Hard Energy Causality Enforcement**: If an agent attempts to sample when $\text{battery} < 0.05$, the action is **rejected** (overridden to Sleep/no-op). Battery level cannot drop below 0.0.

---

## 5. Joint Reward Function & Coordination Penalty

Each agent $i$ receives a total scalar reward comprising an **individual component** and a **coordination component**:

$$R_{t, i} = R_{\text{ind}, i} - R_{\text{coord}, i}$$

### Individual Component ($R_{\text{ind}, i}$)

$$R_{\text{ind}, i} = \begin{cases} 
+1.00 & \text{if Action}_i = \text{Sample (Executed) and Entropy}_i \ge 0.60 \\
-0.20 & \text{if Action}_i = \text{Sample (Executed) and Entropy}_i < 0.60 \quad (\text{Wasted Energy}) \\
-0.80 & \text{if Action}_i = \text{Sleep (or Rejected) and Entropy}_i \ge 0.60 \quad (\text{Missed Event}) \\
+0.05 & \text{if Action}_i = \text{Sleep and Entropy}_i < 0.60 \quad (\text{Correct Restraint})
\end{cases}$$

### Coordination Component ($R_{\text{coord}, i}$)

To discourage redundant simultaneous sampling across neighboring sensors covering overlapping fields:

$$R_{\text{coord}, i} = \begin{cases}
w_{\text{coord}} \times N_{\text{co-samplers}} & \text{if Action}_i = \text{Sample (Executed)} \\
0.0 & \text{otherwise}
\end{cases}$$

* **$N_{\text{co-samplers}}$**: Number of **other** neighboring agents $j \neq i$ that also successfully executed a Sample in the same timestep $t$.
* **Weight $w_{\text{coord}} = 0.10$**: Modest penalty scalar per co-sampling neighbor.

---

## 6. PettingZoo Wrapper (`IoTSensorEnv`) & Module B Integration

Implemented in [`pettingzoo_env.py`](file:///c:/Users/VISMAYA/Desktop/marl/FinalYearProject-MARL/environment/pettingzoo_env.py), the `IoTSensorEnv` class subclasses `pettingzoo.ParallelEnv` and wraps `MultiAgentSensorEnv`.

### Drop-In Compatibility Confirmation

`IoTSensorEnv` is a **100% true drop-in replacement** for [`training/mock_env.py`](file:///c:/Users/VISMAYA/Desktop/marl/FinalYearProject-MARL/training/mock_env.py)'s `MockIoTSensorEnv`.

To swap from the mock environment to the real physics simulation in Module B's [`training/env_wrapper.py`](file:///c:/Users/VISMAYA/Desktop/marl/FinalYearProject-MARL/training/env_wrapper.py):

```python
# In training/env_wrapper.py:
# Replace:
# from mock_env import MockIoTSensorEnv
# self.env = MockIoTSensorEnv()

# With:
from environment.pettingzoo_env import IoTSensorEnv
self.env = IoTSensorEnv(scenario="stable")  # Or scenario="volatile"
```

No method signature, return shape, or dictionary key changes are required on Module B's side.

---

## 7. Summary of Module A Deliverables

| Deliverable | File Path | Purpose / Description |
| :--- | :--- | :--- |
| **Deliverable 1: Single-Agent Foundation** | [`environment/single_agent_env.py`](file:///c:/Users/VISMAYA/Desktop/marl/FinalYearProject-MARL/environment/single_agent_env.py) | Single node Dec-POMDP physics (harvesting, entropy spikes, energy causality). |
| **Deliverable 2: Multi-Agent Environment** | [`environment/multi_agent_env.py`](file:///c:/Users/VISMAYA/Desktop/marl/FinalYearProject-MARL/environment/multi_agent_env.py) | 4-agent environment with rolling `neighbor_sampling_rate` & partial observability. |
| **Deliverable 3: PettingZoo ParallelEnv Wrapper** | [`environment/pettingzoo_env.py`](file:///c:/Users/VISMAYA/Desktop/marl/FinalYearProject-MARL/environment/pettingzoo_env.py) | PettingZoo wrapper matching `MockIoTSensorEnv` interface exactly. |
| **Deliverable 4: Scenario Configurations** | [`shared_config.py`](file:///c:/Users/VISMAYA/Desktop/marl/FinalYearProject-MARL/shared_config.py) | `stable` vs `volatile` scenario parameters. |
| **Deliverable 5: Baseline Results Report** | [`environment/baseline_results.py`](file:///c:/Users/VISMAYA/Desktop/marl/FinalYearProject-MARL/environment/baseline_results.py)<br>[`environment/baseline_results.md`](file:///c:/Users/VISMAYA/Desktop/marl/FinalYearProject-MARL/environment/baseline_results.md) | Measured benchmarks for 3 baseline policies across both scenarios. |

---

## 8. Running Simulation & Verification Scripts

Execute the scripts directly from the repository root:

```bash
# Phase 1: Single-Agent Physics Test
python environment/single_agent_env.py

# Phase 2: Multi-Agent Partial Observability & Overlap Test
python environment/multi_agent_env.py

# Phase 3: PettingZoo ParallelEnv Wrapper Test
python environment/pettingzoo_env.py

# Phase 4: Baseline Benchmark Results Deliverable
python environment/baseline_results.py
```
