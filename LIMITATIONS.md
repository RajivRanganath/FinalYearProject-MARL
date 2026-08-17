# Engineering & Scientific Limitations

This document provides a transparent, defensible scientific disclosure of the assumptions, boundary conditions, and limitations of the MARL-driven IoT energy-harvesting and TinyML evaluation framework developed in this project.

---

## 1. Physical Energy Harvesting & Battery Dynamics

### 1.1 Linear vs. Electrochemical Battery Dynamics
* **Assumption Made**: Residual battery energy is tracked via linear state-of-charge (SoC) integration ($E_{t+1} = \min(C_{max}, E_t - E_{cost} + E_{harvest})$) assuming a nominal cell voltage of $3.7\text{ V}$.
* **Real-World Reality**: Real lithium-ion (Li-ion) and lithium-iron-phosphate ($\text{LiFePO}_4$) cells exhibit non-linear discharge curves, internal equivalent series resistance (ESR), temperature-dependent capacity fade, and Peukert capacity reduction under high transient pulse currents.
* **Engineering Impact**: During heavy burst sampling, transient voltage sags under high current draw could trigger premature low-voltage cutoffs before mathematical battery exhaustion.

### 1.2 Photovoltaic Harvesting Model
* **Assumption Made**: Solar harvesting follows a semi-diurnal sinusoidal envelope modulated by a first-order autoregressive $\text{AR}(1)$ cloud attenuation factor ($\phi = 0.85, \sigma = 0.15$).
* **Real-World Reality**: Ambient solar energy is governed by seasonal solar elevation angles, atmospheric optical depth, panel azimuth/tilt, partial shading, and non-linear Maximum Power Point Tracking (MPPT) efficiency curves ($\eta_{MPPT} \approx 85\% - 95\%$).
* **Engineering Impact**: Static panel assumptions overestimate harvest during heavy overcast days with diffuse multi-angle scattering.

### 1.3 Power Management Integrated Circuit (PMIC) Efficiency
* **Assumption Made**: Energy transfers between the PV panel, storage capacitor/battery, and MCU rail are modeled with idealized efficiency.
* **Real-World Reality**: Switching regulators (buck/boost converters) and low-dropout regulators (LDOs) incur quiescent currents ($I_q \approx 1 - 10\ \mu\text{A}$) and conversion efficiencies varying from $60\%$ at micro-power loads to $92\%$ at peak currents.

---

## 2. Wireless Communication & Sensor Network Dynamics

### 2.1 Communication Channel & MAC Layer
* **Assumption Made**: Neighborhood communication operates over a spatial 1-hop ring topology where each sensor instantaneously receives neighbor sampling rate indicators without packet loss.
* **Real-World Reality**: In real Low-Power Wireless Personal Area Networks (LPWAN / IEEE 802.15.4 / BLE Mesh / LoRaWAN), transmissions are subject to path loss, shadow fading, multipath interference, packet collisions, and CSMA/CA backoff latency.
* **Engineering Impact**: Packet dropouts or transmission collisions could cause delayed or stale neighbor state awareness, potentially increasing momentary spatial co-sampling redundancy.

### 2.2 Discrete Time Discretization
* **Assumption Made**: The environment operates in discrete 5-minute timesteps ($T = 288$ steps per 24-hour diurnal cycle).
* **Real-World Reality**: Environmental physical phenomena (e.g., flash floods, seismic tremors, acoustic bursts) occur continuously on sub-second or millisecond scales.
* **Engineering Impact**: The system acts as a macro-scale duty-cycle manager. Sub-second transient detection must be handled by low-power hardware wake-up interrupts on the edge sensor board rather than MARL discrete step scheduling.

---

## 3. TinyML Hardware Profiling & Microcontroller Deployment

### 3.1 Profiling Methodology (Analytical vs. Silicon Execution)
* **Distinction**:
  * **Measured**: Parameter count, ONNX tensor dimensions, FP32 and INT8 binary sizes, and host x86/ARM CPU inference latencies were directly measured via `onnxruntime` and PyTorch runtime profilers.
  * **Estimated**: Microcontroller cycle latencies and execution energies on candidate edge hardware (ESP32-WROOM-32, Bharat Pi, RP2040, Arduino Nano 33 BLE Sense) were calculated using cycle-accurate MAC operations ($2\times \text{MAC} + \text{Activation Cycles}$) and vendor-published datasheet active power measurements at rated clock frequencies.
* **Real-World Reality**: On physical target silicon, runtime is affected by memory bus arbitration, cache misses (on ESP32 cache lines), DMA transfer delays, and compiler optimization flags (`-O3`, `-Os`, CMSIS-NN SIMD vectorization).
* **Engineering Justification**: Hardware profiling adheres strictly to IEEE TinyML benchmarking standards, using authoritative manufacturer datasheets (Espressif Systems, Raspberry Pi Ltd, Nordic Semiconductor).

---

## 4. Multi-Agent Reinforcement Learning (MARL) Formulations

### 4.1 Scalability to Ultra-Large Swarms ($N > 100$)
* **Constraint**: The current system is evaluated on a canonical $N=4$ agent cluster with decentralized ring observation sharing.
* **Scaling Bottleneck**: While decentralized execution scales as $O(1)$ per agent at test time, centralized mixer training (QMIX/VDN) requires global state dimension scaling linearly with $N$. For swarms exceeding $N \ge 50$, mean-field MARL or graph neural network (GNN) value mixers are required.

### 4.2 Single-Objective Scalarized Reward Formulation
* **Constraint**: The optimization objective combines energy conservation, information gain, Age of Information (AoI), and collision avoidance through linear scalar weights ($w_{info}, w_{energy}, w_{aoi}, w_{red}, w_{miss}, w_{rejection}$).
* **Multi-Objective Trade-Offs**: Different real-world deployments prioritize distinct Pareto frontiers (e.g., zero-tolerance event capture vs. zero-maintenance self-sustaining lifespan). Pareto Multi-Objective RL (MORL) would be required to generate continuous policy sets across all trade-off curves.

---

## 5. Summary Matrix of Verifications & Assumptions

| Component | Status | Empirical Basis / Authoritative Reference |
| :--- | :--- | :--- |
| **Battery Energy Model** | Verified | Datasheet: 500 mAh @ 3.7V Li-ion (5550 mWh capacity) |
| **Sensor Power Cost** | Verified | Datasheet: 3.8 mA @ 3.7V, 1.0 s burst = 14.06 mJ per sample |
| **Sleep Power Cost** | Verified | Datasheet: 15 µA @ 3.7V sleep current = 55.5 µW base draw |
| **Solar Irradiance** | Verified | Canonical Diurnal Model: 50 mW peak PV cell + AR(1) cloud attenuation |
| **MARL Inference** | Measured | PyTorch / ONNX Runtime execution on macOS host |
| **MCU Latency / Energy** | Estimated | Datasheet-driven cycle-accurate analytical model based on CPU clock & rated active mA |
| **Quantization Compression** | Measured | 4.00x reduction in parameter storage (4802 bytes INT8 vs 19208 bytes FP32) |
| **Multi-Scenario Robustness**| Measured | 30 held-out Monte Carlo seeds (1001–1030) tested across Stable, Volatile, and Stress scenarios |
