"""Interactive ONNX action-distribution diagnostic (not a pytest module)."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

import shared_config


def run_diagnostic(path: str = "training/policy.onnx") -> None:
    session = ort.InferenceSession(str(Path(path)))
    obs_input, hidden_input = session.get_inputs()[:2]
    input_dim = int(obs_input.shape[1])
    hidden_dim = int(hidden_input.shape[1])

    def get_q(obs_array: np.ndarray, agent_idx: int = 0) -> np.ndarray:
        agent_id = np.eye(shared_config.NUM_AGENTS, dtype=np.float32)[agent_idx]
        if input_dim == shared_config.ENV_OBS_DIM:
            model_obs = obs_array.astype(np.float32)[None]
        elif input_dim == shared_config.MODEL_INPUT_DIM:
            model_obs = np.concatenate([obs_array, agent_id]).astype(np.float32)[None]
        else:
            raise ValueError(
                f"Model expects legacy input dimension {input_dim}; current causal contracts are "
                f"{shared_config.ENV_OBS_DIM} or {shared_config.MODEL_INPUT_DIM}."
            )
        outputs = session.run(None, {
            "obs": model_obs,
            "hidden_state_in": np.zeros((1, hidden_dim), dtype=np.float32),
        })
        return outputs[0][0]

    print("=== Q-values on causal observation states ===")
    states = [
        ("high battery, strong proxy", [0.9, 0.8, 0.2, 0.0, 0.5]),
        ("high battery, weak proxy", [0.9, 0.1, 0.2, 0.0, 0.5]),
        ("low battery, strong proxy", [0.1, 0.8, 0.8, 0.0, 0.0]),
        ("high AoI, weak proxy", [0.9, 0.1, 0.9, 0.0, 0.5]),
        ("strong proxy, busy neighbors", [0.9, 0.8, 0.2, 0.75, 0.5]),
    ]
    for name, state in states:
        print(f"\n{name}:")
        for agent_idx in range(shared_config.NUM_AGENTS):
            q = get_q(np.asarray(state, dtype=np.float32), agent_idx)
            print(f"  agent {agent_idx}: sleep={q[0]:+.4f}, sample={q[1]:+.4f}")


if __name__ == "__main__":
    run_diagnostic()
