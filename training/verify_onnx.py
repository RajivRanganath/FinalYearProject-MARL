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
    args = SimpleNamespace(
        hidden_dim=64,
        rnn_hidden_dim=64,
        n_actions=2,
        use_rnn=False,
    )
    input_shape = 7  # 3 obs + 4 one-hot agent ID
    
    pytorch_agent = RNNAgent(input_shape, args)
    pytorch_agent.load_state_dict(torch.load(agent_path, map_location="cpu"))
    pytorch_agent.eval()
    
    # --- Load ONNX model ---
    onnx_session = ort.InferenceSession(onnx_path)
    
    print(f"Loaded PyTorch agent from: {agent_path}")
    print(f"Loaded ONNX model from:    {onnx_path}")
    print(f"Running {n_tests} comparison tests...\n")
    
    max_abs_diff_q = 0.0
    max_abs_diff_h = 0.0
    all_passed = True
    
    for i in range(n_tests):
        # Generate random test inputs
        test_obs = torch.randn(1, input_shape)
        test_hidden = torch.zeros(1, args.hidden_dim)
        
        # --- PyTorch inference ---
        with torch.no_grad():
            pt_q, pt_h = pytorch_agent(test_obs, test_hidden)
        pt_q = pt_q.numpy()
        pt_h = pt_h.numpy()
        
        # --- ONNX inference ---
        onnx_outputs = onnx_session.run(None, {
            'obs': test_obs.numpy(),
            'hidden_state_in': test_hidden.numpy()
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
    
    print(f"\nMax absolute Q-value difference: {max_abs_diff_q:.2e}")
    print(f"Max absolute hidden  difference: {max_abs_diff_h:.2e}")
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED — ONNX export is numerically faithful to PyTorch.")
    else:
        print("\n❌ SOME TESTS FAILED — ONNX export may have numerical issues.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_onnx.py <checkpoint_dir> <onnx_path>")
        print("Example: python verify_onnx.py results/models/qmix_.../25000 training/policy.onnx")
        sys.exit(1)
    
    verify_onnx_against_pytorch(sys.argv[1], sys.argv[2])
