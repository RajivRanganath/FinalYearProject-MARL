# Multi-Agent Reinforcement Learning (MARL) for Energy-Harvesting IoT Sensing & TinyML Edge Deployment

[![PyTest Suite](https://img.shields.io/badge/pytest-11%20passed%20(100%25)-brightgreen.svg)](tests/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](requirements.txt)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Validated-blueviolet.svg)](training/policy.onnx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Summary & Research Motivation

In wireless sensor networks (WSNs) powered by ambient energy harvesting (solar photovoltaics), edge sensor nodes face an essential trade-off: **conserving finite battery energy to prevent node death** versus **sampling frequently to capture transient environmental events and maintain low Age of Information (AoI)**.

Traditional fixed-interval periodic sampling policies either exhaust stored energy during overcast days or fail to detect bursty high-entropy phenomena. This repository implements a complete, mathematically grounded engineering framework:
1. **Dec-POMDP Simulation Core**: A multi-agent energy-harvesting environment modeling diurnal solar irradiation with autoregressive cloud attenuation, Poisson anomaly bursts, spatial co-sampling collisions, and strict energy causality.
2. **Cooperative MARL Training Engine**: Multi-algorithm training (**QMIX**, **VDN**, **IQL**) under Centralized Training with Decentralized Execution (CTDE) exported to standard ONNX format.
3. **TinyML Microcontroller Profiling Engine**: Analytical and empirical hardware evaluation across real-world edge MCUs (**ESP32**, **Bharat Pi**, **RP2040**, **Arduino Nano 33 BLE Sense**), verifying cycle counts, SRAM/Flash footprints, and net energy conservation.

---

## 2. Mathematical Formulation & Architecture

```
                                  CENTRALIZED TRAINING (CTDE)
                   ┌────────────────────────────────────────────────────────┐
                   │ Global State s_t = [E_1..E_N, H_1..H_N, Rate_1..Rate_N]│
                   │               QMIX / VDN Value Mixer                   │
                   └──────────────────────────┬─────────────────────────────┘
                                              │ Gradient Updates
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
             Agent 1 Policy (MLP)                             Agent N Policy (MLP)
     Input: [E_1, H_1, Rate_neighbor, ID]             Input: [E_N, H_N, Rate_neighbor, ID]
                     │                                                 │
                     ▼                                                 ▼
             Action a_1 ∈ {0, 1}                              Action a_N ∈ {0, 1}
                     │                                                 │
 ════════════════════╪═════════════════════════════════════════════════╪════════════════════
                     ▼ DECENTRALIZED EXECUTION ON EMBEDDED SILICON      ▼
       ┌───────────────────────────┐                     ┌───────────────────────────┐
       │   ESP32 / RP2040 Node 1   │ <──── 1-Hop ──────> │   ESP32 / RP2040 Node N   │
       │  500 mAh Li-ion + 50mW PV │    Ring Channel     │  500 mAh Li-ion + 50mW PV │
       └───────────────────────────┘                     └───────────────────────────┘
```

### 2.1 Canonical Decentralized Reward Function
For each agent $i$ at decision step $t$, the decision-time reward $R_i(s_t, a_t)$ is evaluated before transitioning to state $s_{t+1}$:
$$R_i(s_t, a_t) = w_{info} \cdot \mathbb{I}_{event}(s_t) \cdot a_i - w_{energy} \cdot a_i - w_{aoi} \cdot \widetilde{\text{AoI}}_i - w_{red} \cdot \text{Redundancy}_i - w_{miss} \cdot \mathbb{I}_{event}(s_t) \cdot (1 - a_i)$$

Where:
* $a_i \in \{0: \text{SLEEP}, 1: \text{SAMPLE}\}$.
* $\mathbb{I}_{event}(s_t) = \mathbb{I}(H_t \ge 0.60)$ indicates an active high-entropy event.
* $\text{Redundancy}_i = \frac{1}{|N_i|} \sum_{j \in N_i} a_j$ penalizes concurrent neighbor sampling collisions.
* Feasibility constraint: $a_i = 1$ is strictly masked if $E_i(t) < E_{sample}$.

### 2.2 Physical Energy Model Parameters
* **Battery Capacity**: $500\text{ mAh} @ 3.7\text{ V} = 5550\text{ mWh} = 19,980\text{ J}$.
* **Active Sample Energy**: $3.8\text{ mA} @ 3.7\text{ V} \times 1.0\text{ s} = 14.06\text{ mJ}$ ($E_{sample} = 0.0039\text{ mWh}$).
* **Sleep Power Draw**: $15\ \mu\text{A} @ 3.7\text{ V} = 55.5\ \mu\text{W}$ ($E_{sleep} = 0.004625\text{ mWh}$ per 5-min step).
* **PV Energy Harvest**: $50\text{ mW}$ peak solar panel with semi-diurnal sinusoidal envelope and $\text{AR}(1)$ cloud attenuation ($\phi=0.85, \sigma=0.15$).

---

## 3. Project Structure

```
FinalYearProject-MARL/
├── shared_config.py               # Single source of truth for contracts, dimensions & seeds
├── environment/                   # MODULE A: Physical Simulation & Dec-POMDP Core
│   ├── energy_model.py            # Physical battery & diurnal solar harvesting
│   ├── single_agent_env.py        # Single sensor node Gym environment
│   ├── multi_agent_env.py         # Multi-agent spatial ring network environment
│   └── pettingzoo_env.py          # PettingZoo ParallelEnv standard compliance wrapper
├── training/                      # MODULE B: MARL Algorithms & ONNX Export Engine
│   ├── epymarl/                   # EPyMARL core library (QMIX, VDN, IQL)
│   ├── env_wrapper.py             # EPyMARL MultiAgentEnv environment bridge
│   ├── train_all.py               # Master multi-algorithm, multi-seed training suite
│   ├── export_onnx.py             # Dynamic PyTorch to ONNX export utility
│   └── policy.onnx                # Primary deployment ONNX policy (4,802 params, 7.3 KB)
├── hardware_eval/                 # MODULE C: TinyML Profiling Engine
│   ├── device_specs.json          # Verified datasheet specifications (ESP32, RP2040, etc.)
│   ├── model_analysis.py          # Layer-by-layer parameter & FLOP counter
│   ├── quantize_model.py          # FP32 to INT8 dynamic quantization engine
│   ├── memory_estimator.py        # SRAM runtime buffer & Flash footprint profiler
│   ├── latency_estimator.py       # Cycle-accurate inference latency estimator
│   ├── energy_estimator.py        # Micro-Joule inference energy model
│   └── rank_devices.py            # Master hardware ranking & evaluation report generator
├── deployment/                    # Benchmarking & Publication Visualizations
│   ├── evaluate_all.py            # 10-policy Monte Carlo evaluator (30 seeds, 95% CIs)
│   ├── ablation_study.py          # Controlled ablation study across 5 variants
│   └── generate_plots.py          # Pareto frontier & hardware bar chart generator
├── tests/                         # Automated PyTest Test Suite (11/11 Passing)
├── results/                       # Empirical Outputs, Reports, Models & Figures
│   ├── exported_models/           # Exported ONNX models across algorithms & seeds
│   ├── plots/                     # Publication PNG figures
│   ├── benchmark_summary_*.csv    # Measured benchmark CSV tables
│   └── ablation_results.csv       # Ablation metrics
├── LIMITATIONS.md                 # Scientific assumptions & boundary disclosures
└── README.md                      # Primary project documentation
```

---

## 4. Quick Start & Execution Guide

### 4.1 Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/RajivRanganath/FinalYearProject-MARL.git
cd FinalYearProject-MARL

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 4.2 Run Automated Test Suite
```bash
pytest tests/ -v
```

### 4.3 Train All MARL Policies (QMIX, VDN, IQL)
```bash
# Train all algorithms across 3 seeds (101, 102, 103)
python training/train_all.py --alg=all --t_max=60000

# Or train a single algorithm:
python training/train_all.py --alg=qmix --seed=101 --t_max=60000
```

### 4.4 Run 30-Seed Multi-Policy Benchmark Evaluation
```bash
python deployment/evaluate_all.py --scenario=all
```

### 4.5 Run Ablation Study & Generate Figures
```bash
# Run ablation study
python deployment/ablation_study.py

# Run TinyML hardware evaluation
python hardware_eval/rank_devices.py

# Generate publication plots
python deployment/generate_plots.py
```

---

## 5. Measured Empirical Results

### 5.1 Multi-Scenario Benchmark (30 Test Seeds with 95% CIs)

Measured across 30 held-out Monte Carlo seeds (`1001`–`1030`) under the **Volatile** environmental scenario ($T=288$ steps):

| Policy | Team Reward | Event Recall (%) | Mean AoI (steps) | Rejections | Overlap Steps | Final Battery |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Entropy Threshold (>0.60)** | **897.10 ± 29.38** | **94.85 ± 1.27%** | **10.25 ± 0.55** | **0.00 ± 0.00** | **37.67 ± 3.15** | **0.34 ± 0.04** |
| **Greedy Myopic Heuristic** | 897.10 ± 29.38 | 94.85 ± 1.27% | 10.25 ± 0.55 | 0.00 ± 0.00 | 37.67 ± 3.15 | 0.34 ± 0.04 |
| **Battery+Entropy Heuristic** | 861.76 ± 28.12 | 93.50 ± 1.47% | 9.43 ± 0.58 | 0.00 ± 0.00 | 74.37 ± 4.44 | 0.19 ± 0.03 |
| **Always Sample (Feasible)** | 72.02 ± 29.09 | 50.89 ± 1.73% | 15.86 ± 0.11 | 0.00 ± 0.00 | 147.90 ± 0.88 | 0.02 ± 0.00 |
| **Battery Threshold (<20%)** | 42.87 ± 29.77 | 48.68 ± 1.75% | 17.31 ± 0.11 | 0.00 ± 0.00 | 142.37 ± 0.92 | 0.17 ± 0.00 |
| **Random Feasible** | -128.78 ± 19.95 | 33.01 ± 1.25% | 8.91 ± 0.24 | 0.00 ± 0.00 | 125.83 ± 1.89 | 0.02 ± 0.00 |
| **Fixed Interval (N=12)** | -497.76 ± 20.15 | 7.99 ± 0.52% | 5.50 ± 0.00 | 0.00 ± 0.00 | 24.00 ± 0.00 | 0.69 ± 0.00 |
| **Trained VDN (MARL)** | -541.21 ± 26.34 | 4.29 ± 1.19% | 125.02 ± 4.13 | 0.00 ± 0.00 | 0.40 ± 0.53 | 0.99 ± 0.00 |
| **Always Sleep** | -610.20 ± 20.40 | 0.00 ± 0.00% | 144.50 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.99 ± 0.00 |
| **Trained QMIX (MARL)** | -610.20 ± 20.40 | 0.00 ± 0.00% | 144.50 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.99 ± 0.00 |
| **Trained IQL (MARL)** | -610.20 ± 20.40 | 0.00 ± 0.00% | 144.50 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.99 ± 0.00 |

### 5.2 Microcontroller Hardware Profiling Ranking

Evaluated using datasheet parameters from Espressif Systems, Raspberry Pi Ltd, and Nordic Semiconductor:

| Rank | Target Device | Core Architecture | Clock (MHz) | Active Power | INT8 Latency (µs) | INT8 Energy (µJ) | Net Energy Saved vs Sample | SRAM Util% | Flash Util% |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **ESP32-WROOM-32** | Xtensa Dual-Core LX6 | 240 | 224.4 mW | 40.0 µs | 8.98 µJ | **1,566x** | 0.12% | 0.12% |
| 2 | **Bharat Pi** | ESP32-WROOM-32 | 240 | 224.4 mW | 40.0 µs | 8.98 µJ | **1,566x** | 0.12% | 0.12% |
| 3 | **RP2040** | Dual ARM Cortex-M0+ | 133 | 72.6 mW | 72.2 µs | 5.24 µJ | **2,683x** | 0.24% | 0.23% |
| 4 | **Arduino Nano 33 BLE**| ARM Cortex-M4F | 64 | 15.0 mW | 150.1 µs | 2.25 µJ | **6,249x** | 0.24% | 0.47% |

* **Energy Savings Proof**: A physical sensor read consumes $14.06\text{ mJ} = 14,060\ \mu\text{J}$. An INT8 policy inference consumes $2.25 - 8.98\ \mu\text{J}$. Even after factoring in active computation, **avoiding a single redundant transmission saves >1,500x the energy required to make the decision**.

---

## 6. Scientific Limitations & Disclosure

Detailed scientific assumptions, electrochemical battery non-linearities, wireless MAC layer simplifications, and TinyML analytical estimation boundaries are transparently documented in [LIMITATIONS.md](LIMITATIONS.md).

---

## 7. Citation & Reproducibility

To cite this repository in academic or final-year engineering project reports:
```bibtex
@misc{marl_energy_iot_2026,
  author = {Rajiv Ranganath},
  title = {Multi-Agent Reinforcement Learning for Energy-Harvesting IoT Sensor Scheduling and TinyML Edge Profiling},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/RajivRanganath/FinalYearProject-MARL}}
}
```
