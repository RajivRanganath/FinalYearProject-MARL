"""
ONNX Export Engine for MARL Policy
MARL Adaptive IoT Sampling & TinyML Hardware Evaluation

Exports individual agent policy networks from trained EPyMARL PyTorch checkpoints
into standard ONNX format for deployment and microcontroller hardware profiling.
"""

import os
import sys
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

# Cross-platform path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
EPYMARL_SRC = Path(__file__).resolve().parent / "epymarl" / "src"

for p in [str(ROOT_DIR), str(EPYMARL_SRC)]:
    if p not in sys.path:
        sys.path.append(p)

import shared_config
from modules.agents.rnn_agent import RNNAgent

def export_agent_to_onnx(
    checkpoint_path: str,
    output_onnx_path: str = "training/policy.onnx",
    hidden_dim: int = shared_config.HIDDEN_DIM
) -> Dict[str, Any]:
    """
    Loads agent.th weights from checkpoint and exports to ONNX.
    """
    ckpt_file = Path(checkpoint_path)
    if ckpt_file.is_dir():
        ckpt_file = ckpt_file / "agent.th"

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint agent weights not found at {ckpt_file}")

    state_dict = torch.load(str(ckpt_file), map_location="cpu", weights_only=True)
    detected_hidden_dim = state_dict["fc1.weight"].shape[0] if "fc1.weight" in state_dict else hidden_dim
    detected_use_rnn = "rnn.weight_ih" in state_dict

    args = SimpleNamespace(
        hidden_dim=detected_hidden_dim,
        rnn_hidden_dim=detected_hidden_dim,
        n_actions=shared_config.NUM_ACTIONS,
        use_rnn=detected_use_rnn
    )

    # Infer the exact training contract (full model or no-agent-ID ablation)
    # from the checkpoint instead of hard-coding a stale dimension.
    input_shape = int(state_dict["fc1.weight"].shape[1])
    if input_shape not in (shared_config.MODEL_INPUT_DIM, shared_config.ENV_OBS_DIM):
        raise ValueError(f"Unsupported checkpoint input dimension: {input_shape}")
    agent = RNNAgent(input_shape=input_shape, args=args)
    agent.load_state_dict(state_dict)
    agent.eval()

    dummy_obs = torch.zeros((1, input_shape), dtype=torch.float32)
    dummy_hidden = torch.zeros((1, detected_hidden_dim), dtype=torch.float32)

    out_file = Path(output_onnx_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting PyTorch model ({ckpt_file}) to ONNX ({out_file})...")

    torch.onnx.export(
        agent,
        (dummy_obs, dummy_hidden),
        str(out_file),
        # The legacy exporter is intentional here: PyTorch's dynamo exporter
        # currently specializes GRUCell outputs to the example batch size even
        # when dynamic axes are requested. This path preserves the declared
        # batch contract and is covered by batch-2 ONNX Runtime inference.
        dynamo=False,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['obs', 'hidden_state_in'],
        output_names=['q_values', 'hidden_state_out'],
        dynamic_axes={
            'obs': {0: 'batch_size'},
            'hidden_state_in': {0: 'batch_size'},
            'q_values': {0: 'batch_size'},
            'hidden_state_out': {0: 'batch_size'},
        }
    )

    # GRUCell's TorchScript export can leave the known feature width symbolic.
    # Pin only the non-batch output dimensions; axis 0 remains dynamic.
    onnx_model = onnx.load(str(out_file))
    known_output_widths = {
        "q_values": shared_config.NUM_ACTIONS,
        "hidden_state_out": detected_hidden_dim,
    }
    for output in onnx_model.graph.output:
        width = known_output_widths.get(output.name)
        if width is not None:
            feature_dim = output.type.tensor_type.shape.dim[1]
            feature_dim.ClearField("dim_param")
            feature_dim.dim_value = width
    onnx.save(onnx_model, str(out_file))

    # Verify exported ONNX model
    onnx.checker.check_model(onnx_model)

    # Test inference with ONNX Runtime
    session = ort.InferenceSession(str(out_file))
    test_out = session.run(None, {
        'obs': dummy_obs.numpy(),
        'hidden_state_in': dummy_hidden.numpy()
    })
    q_vals = test_out[0]

    param_count = sum(p.numel() for p in agent.parameters())
    file_size_kb = out_file.stat().st_size / 1024.0

    metadata = {
        "checkpoint_source": str(ckpt_file),
        "output_onnx_path": str(out_file),
        "input_dim": input_shape,
        "hidden_dim": detected_hidden_dim,
        "use_rnn": detected_use_rnn,
        "output_dim": shared_config.NUM_ACTIONS,
        "total_parameters": param_count,
        "file_size_kb": round(file_size_kb, 2),
        "verification_status": "PASSED"
    }

    print(f"Export Successful! Parameter count: {param_count}, File size: {file_size_kb:.2f} KB")
    return metadata

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else "training/policy.onnx"
        export_agent_to_onnx(ckpt, out)
    else:
        print("Usage: python export_onnx.py <path_to_agent.th_or_dir> [output_onnx_path]")
