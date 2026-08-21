"""
Model Complexity and Computational Analysis Engine
Module C — TinyML Hardware Evaluation

Programmatically analyzes exported ONNX and PyTorch MARL policy graphs:
- Exact parameter counts per layer
- Memory footprint for Float32 vs Int8 quantized representations
- Multiply-Accumulate (MAC) and Floating Point Operation (FLOP) counts per inference
- Peak activation buffer requirements for TinyML tensor arena sizing
"""

import sys
import json
import numpy as np
import onnx
from onnx import TensorProto
from pathlib import Path
from typing import Dict, Any

# Cross-platform path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config

DEFAULT_MODEL_PATH = (
    ROOT_DIR
    / "results"
    / "upgrade_models"
    / "extended"
    / "coordinated"
    / "qmix_seed101.onnx"
)
DEPLOYMENT_REPLICA_COUNT = 3
DEPLOYMENT_AGENT_COUNT = shared_config.NUM_AGENTS


def analyze_marl_policy(
    onnx_path: str | Path | None = None,
    replica_count: int = DEPLOYMENT_REPLICA_COUNT,
) -> Dict[str, Any]:
    """Derive recurrent-policy parameters and matrix MACs from the real graph.

    The returned Int8 memory is a projection only.  This function does not
    claim successful conversion, micro-runtime support, or board measurement.
    """
    if replica_count < 1:
        raise ValueError("replica_count must be positive")
    model_path = DEFAULT_MODEL_PATH if onnx_path is None else Path(onnx_path)
    if not model_path.is_absolute():
        model_path = ROOT_DIR / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    model = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(model)
    if len(model.graph.input) < 2:
        raise ValueError("Expected observation and recurrent hidden-state inputs")
    input_dim = int(model.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
    hidden_dim = int(model.graph.input[1].type.tensor_type.shape.dim[1].dim_value)
    output_dim = int(model.graph.output[0].type.tensor_type.shape.dim[1].dim_value)

    float_initializers = {
        init.name: init
        for init in model.graph.initializer
        if init.data_type == TensorProto.FLOAT
    }
    parameters_per_replica = int(sum(
        np.prod(init.dims, dtype=np.int64) for init in float_initializers.values()
    ))
    gemm_weights = {
        node.input[1]
        for node in model.graph.node
        if node.op_type in {"Gemm", "MatMul"} and len(node.input) > 1
    }
    layers = []
    macs_per_replica = 0
    for name in sorted(gemm_weights):
        initializer = float_initializers.get(name)
        if initializer is None or len(initializer.dims) != 2:
            continue
        macs = int(np.prod(initializer.dims, dtype=np.int64))
        macs_per_replica += macs
        layers.append({"name": name, "shape": list(initializer.dims), "macs": macs})

    total_parameters = parameters_per_replica * replica_count
    total_macs = macs_per_replica * replica_count
    largest_tensor_elements = max(input_dim, output_dim, hidden_dim * 3)
    external_path = model_path.with_name(model_path.name + ".data")
    serialized_bytes = model_path.stat().st_size
    if external_path.is_file():
        serialized_bytes += external_path.stat().st_size

    return {
        "status": "ANALYTICAL_GRAPH_ACCOUNTING_ONLY",
        "model_path": str(model_path.relative_to(ROOT_DIR)),
        "architecture": "shared recurrent GRU policy",
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "replica_count": replica_count,
        "agents": DEPLOYMENT_AGENT_COUNT,
        "matrix_layers_per_replica": layers,
        "parameters_per_replica": parameters_per_replica,
        "total_parameters": total_parameters,
        "macs_per_replica_inference": macs_per_replica,
        "total_macs_per_agent_decision": total_macs,
        "total_macs_per_inference": total_macs,
        "total_flops_per_agent_decision_lower_bound": total_macs * 2,
        "float32_weight_memory_kb": round(total_parameters * 4 / 1024.0, 2),
        "projected_int8_weight_memory_kb": round(total_parameters / 1024.0, 2),
        "int8_weight_memory_kb": round(total_parameters / 1024.0, 2),
        "largest_activation_tensor_fp32_bytes_lower_bound": largest_tensor_elements * 4,
        "largest_activation_tensor_int8_bytes_lower_bound": largest_tensor_elements,
        "recurrent_state_fp32_bytes": (
            DEPLOYMENT_AGENT_COUNT * replica_count * hidden_dim * 4
        ),
        "serialized_onnx_kb_per_replica": round(serialized_bytes / 1024.0, 2),
        "limitations": [
            "Int8 weight memory is a projection, not proof of successful quantization.",
            "Tensor-arena peak depends on conversion, operator kernels, and memory planning.",
            "Latency and energy require a target-runtime build and board measurement.",
        ],
    }

if __name__ == "__main__":
    report = analyze_marl_policy()
    print("=" * 60)
    print("MARL POLICY MODEL COMPLEXITY ANALYSIS")
    print("=" * 60)
    print(json.dumps(report, indent=2))
