# Module C: MARL Policy Architecture Specification

This document provides the exact model specifications for the trained EPyMARL policy network (`training/policy.onnx`). Module C should use this for all hardware profiling (memory footprint, latency, and energy estimation).

## 1. Network Topology (MLP Mode)
The policy network uses the `RNNAgent` class from EPyMARL but is configured as a standard Multi-Layer Perceptron (`use_rnn=False`). It executes independently per agent (Decentralized Execution).

**Layers:**
1. **Input Layer (`fc1`)**: Fully Connected `Linear(in_features=7, out_features=64)` -> `ReLU` activation
2. **Hidden Layer (`rnn`)**: Fully Connected `Linear(in_features=64, out_features=64)` -> `ReLU` activation
3. **Output Layer (`fc2`)**: Fully Connected `Linear(in_features=64, out_features=2)` -> No activation (raw Q-values)

## 2. Parameter Count Calculation
- **fc1 (Input to Hidden 1)**: `(7 * 64) + 64 (bias)` = 512 parameters
- **rnn (Hidden 1 to Hidden 2)**: `(64 * 64) + 64 (bias)` = 4,160 parameters
- **fc2 (Hidden 2 to Output)**: `(64 * 2) + 2 (bias)` = 130 parameters
- **TOTAL**: **4,802 parameters**

At 32-bit floating-point precision (4 bytes per parameter), the model weights require exactly **19.2 KB** of memory, making it highly suitable for microcontroller deployment without pruning.

## 3. Input & Output Specifications

### Inputs
The ONNX model expects two inputs (due to EPyMARL's default RNN interface), though `hidden_state_in` is effectively ignored during MLP execution:
1. `obs` (Shape: `[batch_size, 7]`)
   - Index 0: `residual_energy` (Normalized 0.0 to 1.0)
   - Index 1: `data_entropy` (Normalized 0.0 to 1.0)
   - Index 2: `neighbor_sampling_rate` (Normalized 0.0 to 1.0)
   - Indices 3-6: One-hot encoded Agent ID (e.g., `[1,0,0,0]` for Agent 0)
2. `hidden_state_in` (Shape: `[batch_size, 64]`) - Pass a zero-tensor for MLP execution.

### Outputs
1. `q_values` (Shape: `[batch_size, 2]`) - The raw Q-values for the two possible actions.
2. `hidden_state_out` (Shape: `[batch_size, 64]`) - Output hidden state (ignore).

**Action Selection:**
Take the `argmax` of `q_values` along axis 1.
- `0`: Sleep / Wait
- `1`: Sample Now
