# Module C — Hardware Profiling and Evaluation

Paste the master prompt first, then this.

## Your Role

You are responsible for determining which candidate microcontroller platform is best suited to run Module B's trained policy in the real world, entirely through simulation and published specifications, without requiring access to physical hardware.

## Goal

Given an ONNX exported MARL policy from Module B, evaluate it against multiple candidate hardware platforms and produce a ranked, justified recommendation for which device the team should physically build with in the final implementation phase.

## Phase 1, Device Specification Database

Research and document the following fields for each candidate device. Use manufacturer datasheets and published benchmarks, do not guess values.

Candidate devices to include at minimum:
- **ESP32** (the baseline reference device from the project's literature survey, has native ESP NN optimized kernels and strong TensorFlow Lite Micro support)
- **Bharat Pi** (note, this board is itself built on ESP32 silicon and uses ESP32 board libraries in the Arduino IDE, so expect it to score similarly to ESP32 on compute and memory, the comparison value here is board level differences such as cost, onboard peripherals, and local availability, not a different architecture)
- **Raspberry Pi Pico, RP2040** (dual core, different memory architecture than ESP32, no native Wi Fi on the base model, genuinely distinct comparison point)
- **Arduino Nano 33 BLE Sense** (a common TinyML benchmark board, useful third comparison point)

Fields required per device:
- SRAM available in KB
- Flash storage available in KB or MB
- Clock speed in MHz
- Number of cores
- Native int8 SIMD or ML acceleration support, yes or no, and which library supports it (e.g. ESP NN for ESP32)
- Active mode current draw in mA
- Sleep or deep sleep mode current draw in microamps or mA
- Approximate unit cost
- TensorFlow Lite Micro support maturity, rate as strong, moderate, or experimental based on documentation and community usage you find

Build this as a structured data file (JSON or a Python dict), not just a written table, so the scoring logic in later phases can read it programmatically.

## Phase 2, Dummy Model Testing First

Do not wait for Module B's real trained model. Build a dummy ONNX model matching the expected architecture shape (a small feedforward network, few hidden layers, matching the state vector size of 3 inputs and 2 output actions from the shared interface contract) and build your entire evaluation pipeline against this dummy model first. This lets you develop and debug the full harness in parallel with Module B's training work.

## Phase 3, Quantization Simulation

Use TensorFlow Lite's post training int8 quantization tools (or an ONNX equivalent such as onnxruntime quantization tools) to simulate converting the model from float32 to int8. Measure and report:

- The resulting model size in KB after quantization
- The accuracy or output difference between the float32 and int8 versions on a representative set of test inputs (you can generate synthetic test inputs matching the state vector format if you do not yet have real evaluation data from Module B)

## Phase 4, Memory Footprint Estimation

For each candidate device, calculate whether the quantized model actually fits within that device's available SRAM, accounting not just for the model weights themselves but also realistic overhead for the TensorFlow Lite Micro interpreter and buffers, which typically require additional working memory beyond the raw model size. Flag any device where the model does not fit as infeasible outright, before even considering latency or energy.

## Phase 5, Latency Estimation

Since you cannot run the model live on physical chips, estimate inference latency using published benchmark figures for each chip's compute throughput, cross referenced against the model's estimated FLOP or MAC, multiply accumulate, count. Clearly document your estimation method and its assumptions, since this number will be presented to a panel and needs to be defensible as an estimate, not misrepresented as a measured result.

## Phase 6, Energy per Inference Estimation

Combine each device's active mode current draw with your estimated latency to calculate energy cost per single inference, sample or sleep decision. Compare this against a reasonable assumed solar harvesting budget (coordinate with Module A on what harvesting rate their environment assumes, so your energy comparison is grounded in the same numbers rather than an arbitrary assumption).

## Phase 7, Weighted Scoring Rubric

Build a configurable scoring system that ranks candidate devices using weighted criteria. Default weighting, adjustable later as a team:

- **Energy fit, 40 percent** — how well estimated energy cost per inference fits within the harvesting budget
- **Memory fit, 30 percent** — whether and how comfortably the quantized model fits in available SRAM
- **Latency, 20 percent** — how close inference time is to the sub 100 millisecond real time target stated in the project's methodology
- **Cost and toolchain maturity, 10 percent** — unit cost combined with how well supported the device is for TensorFlow Lite Micro deployment, since a device that is technically superior but poorly documented adds major risk to a limited timeline student project

Any device that failed the memory fit check in Phase 4 should be automatically disqualified regardless of its score on other criteria.

## Phase 8, Final Report

Produce a ranked table of all candidate devices with their scores across each criterion, the final weighted ranking, and a clear written justification for the top recommendation, written so it can be dropped directly into a project report or presentation slide as the bridge between the simulation phase and the physical hardware implementation phase.

## Deliverables

1. Structured device specification database, in the `hardware_eval/` folder
2. Quantization simulation code and results, tested against a dummy model first, then the real model once Module B delivers it
3. Memory footprint estimator with results per device
4. Latency and energy estimator with documented assumptions
5. Weighted scoring rubric, configurable weights
6. Final ranked recommendation report with written justification

If every candidate device fails the memory or energy fit check once the real trained model is evaluated, flag this immediately rather than forcing a recommendation, since it likely means Module B's network needs to be made smaller, and that conversation needs to happen with the team early, not discovered late.
