# Neural Network Architecture for Hardware Evaluation

This document outlines the architecture of the trained MARL policy network (individual agent network) that will be exported to ONNX for hardware profiling.

## Network Specifications

- **Model Type**: Multi-Layer Perceptron (MLP) or small RNN (depending on final EPyMARL config, defaulting to standard MLP for simple environments).
- **Input Shape**: `[1, 3]` (Batch size of 1, State Vector of 3 floats: `residual_energy`, `data_entropy`, `neighbor_sampling_rate`)
- **Input Normalization**: All inputs are floats normalized to `[0, 1]`.
- **Output Shape**: `[1, 2]` (Batch size of 1, Q-values for 2 discrete actions: `0` (Sleep), `1` (Sample Now))
- **Output Interpretation**: Apply `argmax()` over the 2 output values to select the action.

## Parameter Count Estimate

- The target parameter count for the final exported model is **strictly < 100k parameters** to ensure compatibility with memory-constrained microcontrollers (e.g., ESP32, Arduino Nano 33 BLE Sense).
- A typical architecture of 2 hidden layers of size 64 will result in approximately `(3*64) + 64 + (64*64) + 64 + (64*2) + 2 = ~4.5k` parameters, well within our budget.

*Note: Once the final model is trained and exported via `export_onnx.py`, this document should be updated with the exact parameter count and layer shapes.*
