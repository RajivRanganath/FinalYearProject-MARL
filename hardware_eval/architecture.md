# Hardware-Evaluation Policy Architecture

The active hardware analysis profiles the validation-selected Extended QMIX
recurrent policies under `results/upgrade_models/extended/coordinated/`. It does
not use the legacy `training/policy.onnx` MLP.

## Per-replica contract

- Model: shared recurrent GRU policy (`fc1` + `GRUCell` + `fc2`).
- Inputs: `obs[batch, 5]` and `hidden_state_in[batch, 64]`.
- Outputs: `q_values[batch, 2]` and `hidden_state_out[batch, 64]`.
- Parameters: 25,474.
- Matrix MACs per recurrent step: 25,024.
- Stateful execution: each agent retains its own 64-value hidden state between
  timesteps and resets it only at an episode boundary.

The promoted v3 deployment rule evaluates three independently trained replicas
per agent decision, giving 76,422 parameters and 75,072 matrix MACs before the
voting operation. The same three weight sets are shared across the four agents;
the recurrent hidden state is per agent and per replica.

## Evidence boundary

`model_analysis.py` derives these values from the actual ONNX initializers and
graph. `quantize_model.py` has verified dynamic Int8 weight quantization for one
replica on host ONNX Runtime (see `quantization_report.json`). There is no
ONNX-to-microcontroller conversion, compiled target runtime, operator-support
proof, tensor-arena measurement, board latency, or power measurement. Therefore
the device memory/latency/energy tables remain analytical estimates, not
deployment feasibility results.
