"""
Post-Training Int8 Quantization & Numerical Concordance Engine
Module C — TinyML Hardware Evaluation

Performs fail-closed dynamic Int8 weight quantization of one exported recurrent
MARL policy replica. Evaluates recurrent-state fidelity and action agreement on
host ONNX Runtime; it does not claim microcontroller deployment validation.
"""

import sys
import hashlib
import json
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Any

# Cross-platform path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from hardware_eval.model_analysis import DEFAULT_MODEL_PATH

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    QUANT_AVAILABLE = True
except ImportError:
    QUANT_AVAILABLE = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialized_size(path: Path) -> int:
    size = path.stat().st_size
    external = path.with_name(path.name + ".data")
    if external.is_file():
        size += external.stat().st_size
    return size


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT_DIR / value


def _repair_legacy_shape_annotations(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove stale batch-1 value_info without changing nodes or tensors.

    Older exports marked only inputs as dynamic, leaving intermediate/output
    annotations at batch 1. ORT executes other batch sizes, but its quantizer's
    shape inference rejects those contradictory annotations.
    """
    del model.graph.value_info[:]
    for output in model.graph.output:
        if output.type.tensor_type.shape.dim:
            first = output.type.tensor_type.shape.dim[0]
            first.ClearField("dim_value")
            first.dim_param = "batch_size"
    onnx.checker.check_model(model)
    return model


def run_quantization_evaluation(
    float_model_path: str | Path = DEFAULT_MODEL_PATH,
    quant_model_path: str | Path = "hardware_eval/extended_seed101_int8.onnx",
    n_test_samples: int = 10_000,
    seed: int = 42,
    report_path: str | Path = "hardware_eval/quantization_report.json",
) -> Dict[str, Any]:
    """Quantize and verify without ever substituting the float input model."""
    float_path = _resolve(float_model_path)
    quant_path = _resolve(quant_model_path)
    output_path = _resolve(report_path)
    if not float_path.is_file():
        raise FileNotFoundError(f"Float ONNX model not found: {float_path}")
    if float_path.resolve() == quant_path.resolve():
        raise ValueError("Quantized output must not be the float input model")
    if not QUANT_AVAILABLE:
        raise RuntimeError("ONNX Runtime quantization support is unavailable")

    quant_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_model = _repair_legacy_shape_annotations(
        onnx.load(str(float_path), load_external_data=True)
    )
    try:
        quantize_dynamic(
            model_input=prepared_model,
            model_output=str(quant_path),
            weight_type=QuantType.QInt8,
        )
    except Exception as exc:
        raise RuntimeError(f"ONNX dynamic quantization failed: {exc}") from exc
    if not quant_path.is_file():
        raise RuntimeError("Quantizer returned without creating an output model")

    quant_graph = _repair_legacy_shape_annotations(
        onnx.load(str(quant_path), load_external_data=True)
    )
    onnx.save(quant_graph, str(quant_path))
    onnx.checker.check_model(quant_graph)
    quant_ops = sorted({node.op_type for node in quant_graph.graph.node})
    markers = {"DynamicQuantizeLinear", "MatMulInteger", "QLinearMatMul"}
    if not markers.intersection(quant_ops):
        raise RuntimeError(
            "Output graph has no recognized quantized matrix operator; refusing float fallback"
        )

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session_fp32 = ort.InferenceSession(
        prepared_model.SerializeToString(), sess_options=session_options
    )
    session_int8 = ort.InferenceSession(str(quant_path), sess_options=session_options)
    float_inputs = session_fp32.get_inputs()
    int8_inputs = session_int8.get_inputs()
    float_contract = [(item.name, item.shape) for item in float_inputs]
    int8_contract = [(item.name, item.shape) for item in int8_inputs]
    if float_contract != int8_contract:
        raise RuntimeError(f"Quantized input contract changed: {int8_contract} != {float_contract}")

    input_dim = int(float_inputs[0].shape[1])
    hidden_dim = int(float_inputs[1].shape[1])
    rng = np.random.default_rng(seed)
    sequence_count = min(100, n_test_samples)
    sequence_steps = int(np.ceil(n_test_samples / sequence_count))
    float_hidden = np.zeros((sequence_count, hidden_dim), dtype=np.float32)
    int8_hidden = np.zeros((sequence_count, hidden_dim), dtype=np.float32)
    float_q_batches = []
    int8_q_batches = []
    hidden_errors = []
    for _ in range(sequence_steps):
        observations = rng.uniform(0.0, 1.0, (sequence_count, input_dim)).astype(np.float32)
        if input_dim == shared_config.ENV_OBS_DIM:
            observations[:, shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE] = 0.0
        float_feed = {
            float_inputs[0].name: observations,
            float_inputs[1].name: float_hidden,
        }
        int8_feed = {
            int8_inputs[0].name: observations,
            int8_inputs[1].name: int8_hidden,
        }
        q_fp32, float_hidden = session_fp32.run(None, float_feed)
        q_int8, int8_hidden = session_int8.run(None, int8_feed)
        float_q_batches.append(q_fp32)
        int8_q_batches.append(q_int8)
        hidden_errors.append(np.abs(float_hidden - int8_hidden))
    q_fp32 = np.concatenate(float_q_batches, axis=0)[:n_test_samples]
    q_int8 = np.concatenate(int8_q_batches, axis=0)[:n_test_samples]
    absolute_error = np.abs(q_fp32 - q_int8)
    float_size = _serialized_size(float_path)
    int8_size = _serialized_size(quant_path)
    float_external = float_path.with_name(float_path.name + ".data")

    report = {
        "status": "ONNX_RUNTIME_DYNAMIC_QUANTIZATION_VERIFIED",
        "scope": "Host ONNX Runtime only; no microcontroller conversion or board validation",
        "fidelity_input_scope": (
            f"{sequence_count} synthetic recurrent sequences for "
            f"{sequence_steps} steps; deployed neighbor feature masked"
        ),
        "metadata_repair": (
            "Removed contradictory legacy batch-1 value_info and marked outputs dynamic; "
            "nodes and learned tensors unchanged"
        ),
        "float32_model_path": str(float_path.relative_to(ROOT_DIR)),
        "quantized_int8_model_path": str(quant_path.relative_to(ROOT_DIR)),
        "float32_sha256": _sha256(float_path),
        "float32_external_data_sha256": (
            _sha256(float_external) if float_external.is_file() else None
        ),
        "quantized_sha256": _sha256(quant_path),
        "quantized_operator_types": quant_ops,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "test_samples_evaluated": n_test_samples,
        "float32_serialized_size_kb": round(float_size / 1024.0, 2),
        "int8_serialized_size_kb": round(int8_size / 1024.0, 2),
        "compression_ratio": round(float_size / int8_size, 3),
        "mean_absolute_error_q_values": float(np.mean(absolute_error)),
        "max_q_value_error": float(np.max(absolute_error)),
        "mean_absolute_error_hidden_state": float(
            np.mean(np.concatenate(hidden_errors, axis=0))
        ),
        "max_hidden_state_error": float(np.max(np.concatenate(hidden_errors, axis=0))),
        "action_concordance_percentage": float(
            np.mean(np.argmax(q_fp32, axis=1) == np.argmax(q_int8, axis=1)) * 100.0
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report

if __name__ == "__main__":
    rep = run_quantization_evaluation()
    print("=" * 60)
    print("POST-TRAINING INT8 QUANTIZATION REPORT")
    print("=" * 60)
    print(json.dumps(rep, indent=2))
