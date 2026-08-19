"""
ONNX Export Verification — PyTorch vs ONNX Output Comparison

Module B Prompt (Phase 8) requires:
  "Verify the exported model produces the same outputs as the original PyTorch model
   on a handful of test inputs, to confirm nothing broke in the export process."

This script loads both the PyTorch RNNAgent and the exported ONNX model,
feeds identical random inputs to both, and asserts the outputs match within
floating-point tolerance.
"""

import torch
import numpy as np
import onnxruntime as ort
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "epymarl", "src"))
from modules.agents.rnn_agent import RNNAgent
from types import SimpleNamespace


def verify_onnx_against_pytorch(checkpoint_dir, onnx_path, n_tests=10):
    """
    Load PyTorch agent and ONNX model, compare outputs on random inputs.
    
    Args:
        checkpoint_dir: Path to EPyMARL checkpoint containing agent.th
        onnx_path: Path to the exported .onnx file
        n_tests: Number of random inputs to test
    """
    agent_path = os.path.join(checkpoint_dir, "agent.th")
    
    if not os.path.exists(agent_path):
        print(f"Error: agent.th not found at {agent_path}")
        sys.exit(1)
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found at {onnx_path}")
        sys.exit(1)
    
    # --- Load PyTorch model ---
    state = torch.load(agent_path, map_location="cpu", weights_only=True)
    hidden_dim = int(state["fc1.weight"].shape[0])
    input_shape = int(state["fc1.weight"].shape[1])
    args = SimpleNamespace(
        hidden_dim=hidden_dim,
        rnn_hidden_dim=hidden_dim,
        n_actions=int(state["fc2.weight"].shape[0]),
        use_rnn="rnn.weight_ih" in state,
    )
    
    pytorch_agent = RNNAgent(input_shape, args)
    pytorch_agent.load_state_dict(state)
    pytorch_agent.eval()
    
    # --- Load ONNX model ---
    onnx_session = ort.InferenceSession(onnx_path)
    onnx_input_dim = int(onnx_session.get_inputs()[0].shape[1])
    onnx_hidden_dim = int(onnx_session.get_inputs()[1].shape[1])
    if (onnx_input_dim, onnx_hidden_dim) != (input_shape, hidden_dim):
        raise ValueError(
            "ONNX/checkpoint contract mismatch: "
            f"ONNX={(onnx_input_dim, onnx_hidden_dim)}, "
            f"checkpoint={(input_shape, hidden_dim)}"
        )
    
    print(f"Loaded PyTorch agent from: {agent_path}")
    print(f"Loaded ONNX model from:    {onnx_path}")
    print(f"Running {n_tests} comparison tests...\n")
    
    max_abs_diff_q = 0.0
    max_abs_diff_h = 0.0
    all_passed = True
    
    test_hidden = torch.zeros(1, args.hidden_dim)
    onnx_hidden = test_hidden.numpy()
    for i in range(n_tests):
        # Generate random test inputs
        test_obs = torch.randn(1, input_shape)
        # --- PyTorch inference ---
        with torch.no_grad():
            pt_q, pt_h = pytorch_agent(test_obs, test_hidden)
        pt_q = pt_q.numpy()
        pt_h = pt_h.numpy()
        
        # --- ONNX inference ---
        onnx_outputs = onnx_session.run(None, {
            'obs': test_obs.numpy(),
            'hidden_state_in': onnx_hidden
        })
        onnx_q = onnx_outputs[0]
        onnx_h = onnx_outputs[1]
        
        # --- Compare ---
        q_diff = np.max(np.abs(pt_q - onnx_q))
        h_diff = np.max(np.abs(pt_h - onnx_h))
        max_abs_diff_q = max(max_abs_diff_q, q_diff)
        max_abs_diff_h = max(max_abs_diff_h, h_diff)
        
        q_match = q_diff < 1e-5
        h_match = h_diff < 1e-5
        
        status = "✅" if (q_match and h_match) else "❌"
        print(f"  Test {i+1:2d}: Q-diff={q_diff:.2e}, H-diff={h_diff:.2e}  {status}")
        
        if not (q_match and h_match):
            all_passed = False
            print(f"           PyTorch Q: {pt_q}")
            print(f"           ONNX    Q: {onnx_q}")
        test_hidden = torch.from_numpy(pt_h)
        onnx_hidden = onnx_h
    
    print(f"\nMax absolute Q-value difference: {max_abs_diff_q:.2e}")
    print(f"Max absolute hidden  difference: {max_abs_diff_h:.2e}")
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED — ONNX export is numerically faithful to PyTorch.")
        return {
            "passed": True,
            "input_dim": input_shape,
            "hidden_dim": hidden_dim,
            "use_rnn": args.use_rnn,
            "max_abs_diff_q": float(max_abs_diff_q),
            "max_abs_diff_hidden": float(max_abs_diff_h),
        }
    else:
        raise AssertionError("ONNX export is not numerically faithful to the checkpoint")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_onnx.py <checkpoint_dir> <onnx_path>")
        print("Example: python verify_onnx.py results/models/qmix_.../25000 training/policy.onnx")
        sys.exit(1)
    
    verify_onnx_against_pytorch(sys.argv[1], sys.argv[2])
