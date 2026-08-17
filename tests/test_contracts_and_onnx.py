"""
Contract Assertion and ONNX Inference Tests
MARL Adaptive IoT Sampling Project
"""

import pytest
import numpy as np
import onnxruntime as ort
from pathlib import Path
import shared_config

def test_shared_config_contract_validation():
    """Verify validate_contracts catches dimension mismatches and out-of-bound values."""
    # Valid observation
    valid_obs = np.array([0.5, 0.2, 0.1], dtype=np.float32)
    assert shared_config.validate_contracts(obs=valid_obs) is True

    # Invalid observation (dimension mismatch)
    with pytest.raises(AssertionError):
        shared_config.validate_contracts(obs=np.array([0.5, 0.2], dtype=np.float32))

    # Invalid observation (out of bounds)
    with pytest.raises(AssertionError):
        shared_config.validate_contracts(obs=np.array([1.5, 0.2, 0.1], dtype=np.float32))

    # Valid model input (7 features: 3 obs + 4 one-hot)
    valid_model_input = np.zeros(shared_config.MODEL_INPUT_DIM, dtype=np.float32)
    assert shared_config.validate_contracts(model_input=valid_model_input) is True

def test_onnx_model_execution():
    """Verify ONNX model runs inference and produces expected Q-value dimensions."""
    onnx_path = Path("training/policy.onnx")
    if not onnx_path.exists():
        pytest.skip("training/policy.onnx not yet exported")

    session = ort.InferenceSession(str(onnx_path))
    dummy_obs = np.zeros((1, shared_config.MODEL_INPUT_DIM), dtype=np.float32)
    hidden_inps = [inp for inp in session.get_inputs() if 'hidden' in inp.name]
    h_dim = hidden_inps[0].shape[1] if hidden_inps and isinstance(hidden_inps[0].shape[1], int) else 64
    dummy_hidden = np.zeros((1, h_dim), dtype=np.float32)

    outputs = session.run(None, {
        'obs': dummy_obs,
        'hidden_state_in': dummy_hidden
    })
    q_values = outputs[0]
    assert q_values.shape == (1, shared_config.NUM_ACTIONS)
    assert not np.isnan(q_values).any()
