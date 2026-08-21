# Module C: Analytical TinyML Hardware Estimate (Not Deployment Validation)
**Project:** Multi-Agent Reinforcement Learning for Adaptive IoT Energy-Harvesting Sampling
**Evaluation Methodology:** graph-derived model accounting plus specification-based estimates. No microcontroller conversion, build, timing, or power measurement has been completed.
---

## 1. Provisional Device Scoring Under Stored Assumptions

This is a provisional ranking under explicit analytical assumptions for the three-replica v3 ensemble. It must not be read as proof that the recurrent graph is supported or deployable on any listed device.

| Rank | Candidate Device | Composite Score (0–100) | Nominal Latency | Energy / Inference | Unit Cost | Primary Strength |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **#1** | **Espressif ESP32-WROOM-32** | **76.76** | 1.251 ms | 660.53 µJ | $3.50 | Optimal balance of ultra-low unit cost ($3.50), high clock speed (240MHz with ESP-NN SIMD), 520KB SRAM, and native Wi-Fi/BLE. |
| **#2** | **Bharat Pi Node (ESP32 Silicon Core)** | **67.36** | 1.251 ms | 701.81 µJ | $14.00 | Identical ESP32 silicon performance with turnkey prototyping features, suitable for rapid pilot deployment at moderate cost ($14.00). |
| **#3** | **Arduino Nano 33 BLE Sense (Nordic nRF52840)** | **56.03** | 2.346 ms | 116.13 µJ | $31.00 | Lowest energy per inference (15mA active current + CMSIS-NN SIMD), but significantly higher board cost ($31.00). |
| **#4** | **Raspberry Pi Pico (RP2040)** | **36.71** | 6.773 ms | 491.72 µJ | $4.00 | Low cost ($4.00) and dual-core Cortex-M0+, but penalised by lack of native int8 SIMD, high sleep current (1.3mA), and lack of integrated radio. |

---

## 2. Stored Hardware Specifications (Source Refresh Required)

All device specifications are sourced from authoritative manufacturer datasheets:

- **ESP32 (Espressif ESP32-WROOM-32)**: Xtensa dual-core LX6 @ 240 MHz, 520 KB SRAM, 4 MB Flash, active current 160 mA, deep sleep 15 µA. Source: *Espressif Systems ESP32 Series Datasheet v4.4, 2024*.
- **Bharat Pi (ESP32 Silicon Core)**: Xtensa dual-core LX6 @ 240 MHz, 520 KB SRAM, 4 MB Flash, active current 170 mA, deep sleep 20 µA. Source: *Bharat Pi IoT Board Specification Sheet v2.1, 2023*.
- **Raspberry Pi Pico (RP2040)**: Arm Cortex-M0+ dual-core @ 133 MHz, 264 KB SRAM, 2 MB Flash, active current 22 mA, sleep current 1.3 mA. Source: *Raspberry Pi Foundation RP2040 Datasheet, 2021*.
- **Arduino Nano 33 BLE Sense (Nordic nRF52840)**: Arm Cortex-M4F @ 64 MHz, 256 KB SRAM, 1 MB Flash, active current 15 mA, deep sleep 5 µA. Source: *Nordic Semiconductor nRF52840 Product Specification v1.3, 2021*.

---

## 3. Analytical Memory-Bound Estimate

| Device | Available SRAM | Assumed runtime + lower-bound working SRAM | SRAM Utilization | Available Flash | Flash Utilization | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Espressif ESP32-WROOM-32 | 520 KB | 19.19 KB | 3.69% | 4096.0 KB | 4.75% | **Estimated memory fit; conversion, operators, tensor arena, and target build unverified** |
| Bharat Pi Node (ESP32 Silicon Core) | 520 KB | 19.19 KB | 3.69% | 4096.0 KB | 4.75% | **Estimated memory fit; conversion, operators, tensor arena, and target build unverified** |
| Raspberry Pi Pico (RP2040) | 264 KB | 19.19 KB | 7.27% | 2048.0 KB | 9.5% | **Estimated memory fit; conversion, operators, tensor arena, and target build unverified** |
| Arduino Nano 33 BLE Sense (Nordic nRF52840) | 256 KB | 19.19 KB | 7.5% | 1024.0 KB | 19.01% | **Estimated memory fit; conversion, operators, tensor arena, and target build unverified** |

> **Memory boundary**: the three recurrent replicas contain 76,422 parameters and project to 74.63 KB of Int8 weights. The table uses an assumed runtime overhead and an activation lower bound; actual tensor-arena size and operator support remain unverified.

---

## 4. Latency & Energy-Per-Inference Breakdown

| Device | Clock Speed | SIMD Library | Latency (Opt / Nom / Cons) | Active Power | Energy / Inference | Decision Cost vs. Sampling |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Espressif ESP32-WROOM-32 | 240 MHz | ESP-NN (vectorized assembly kernels) | 0.626 / 1.251 / 2.502 ms | 528.0 mW | **660.53 µJ** | 4.707% of sample cost |
| Bharat Pi Node (ESP32 Silicon Core) | 240 MHz | ESP-NN via ESP-IDF / Arduino ESP32 core | 0.626 / 1.251 / 2.502 ms | 561.0 mW | **701.81 µJ** | 5.001% of sample cost |
| Raspberry Pi Pico (RP2040) | 133 MHz | None (Standard 32-bit ALU emulation) | 4.516 / 6.773 / 10.16 ms | 72.6 mW | **491.72 µJ** | 3.504% of sample cost |
| Arduino Nano 33 BLE Sense (Nordic nRF52840) | 64 MHz | Arm CMSIS-NN (vectorized DSP instructions) | 1.173 / 2.346 / 4.692 ms | 49.5 mW | **116.13 µJ** | 0.828% of sample cost |

### Key Scientific Finding: Energy Return on Investment (ROI)
- A full IoT sample action (sensing + MCU active + RF transmission) consumes **~14,030 µJ (14.03 mJ)**.
- The specification model projects **116.13 µJ to 701.81 µJ** per three-replica decision.
- These are projections, not measured energy. A target build and board-level power trace are required before claiming net-positive edge deployment.

---

## 5. Host ONNX Quantization Check

One Extended replica was dynamically quantized for ONNX Runtime on the host. Serialized size changed from 116.92 KB to 40.46 KB (2.89x), with 99.78% action agreement over 10,000 synthetic recurrent inputs. This is numerical host evidence for one replica, not a converted microcontroller binary, whole-ensemble validation, or measured device performance.
