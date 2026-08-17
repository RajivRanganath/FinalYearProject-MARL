"""Stateful PyTorch and ONNX policy runtimes for evaluation and diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict

import numpy as np
import onnxruntime as ort
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
EPYMARL_SRC = ROOT_DIR / "training" / "epymarl" / "src"
for path in (ROOT_DIR, EPYMARL_SRC):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import shared_config
from modules.agents.rnn_agent import RNNAgent


def model_input(agent_id: str, obs: np.ndarray, include_agent_id: bool = True) -> np.ndarray:
    if not include_agent_id:
        return np.asarray(obs, dtype=np.float32)
    idx = int(agent_id.split("_")[1])
    return np.concatenate([
        np.asarray(obs, dtype=np.float32),
        np.eye(shared_config.NUM_AGENTS, dtype=np.float32)[idx],
    ])


class TorchCheckpointPolicy:
    def __init__(self, checkpoint: Path, include_agent_id: bool = True):
        checkpoint = Path(checkpoint)
        agent_file = checkpoint / "agent.th" if checkpoint.is_dir() else checkpoint
        state = torch.load(agent_file, map_location="cpu", weights_only=True)
        hidden_dim = int(state["fc1.weight"].shape[0])
        input_dim = int(state["fc1.weight"].shape[1])
        use_rnn = "rnn.weight_ih" in state
        expected = shared_config.MODEL_INPUT_DIM if include_agent_id else shared_config.ENV_OBS_DIM
        if input_dim != expected:
            raise ValueError(f"Checkpoint input dimension {input_dim} does not match current contract {expected}")
        args = SimpleNamespace(hidden_dim=hidden_dim, n_actions=shared_config.NUM_ACTIONS, use_rnn=use_rnn)
        self.agent = RNNAgent(input_dim, args)
        self.agent.load_state_dict(state)
        self.agent.eval()
        self.hidden_dim = hidden_dim
        self.use_rnn = use_rnn
        self.include_agent_id = include_agent_id
        self.hidden: Dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.hidden = {
            f"agent_{i}": torch.zeros((1, self.hidden_dim), dtype=torch.float32)
            for i in range(shared_config.NUM_AGENTS)
        }
        self.last_q: Dict[str, np.ndarray] = {}

    def q_values(self, agent_id: str, obs: np.ndarray) -> np.ndarray:
        if agent_id not in self.hidden:
            self.reset()
        x = torch.from_numpy(model_input(agent_id, obs, self.include_agent_id)[None])
        with torch.no_grad():
            q, next_hidden = self.agent(x, self.hidden[agent_id])
        self.hidden[agent_id] = next_hidden
        values = q.numpy()[0]
        self.last_q[agent_id] = values.copy()
        return values

    def select_action(self, agent_id: str, obs: np.ndarray, info: dict) -> int:
        q = self.q_values(agent_id, obs)
        mask = np.asarray(info.get("action_mask", [1, 1]))
        q = q.copy()
        q[mask == 0] = -np.inf
        return int(np.argmax(q))


class ONNXPolicy:
    def __init__(self, path: Path, include_agent_id: bool = True):
        self.path = Path(path)
        self.session = ort.InferenceSession(str(self.path))
        self.input_dim = int(self.session.get_inputs()[0].shape[1])
        self.hidden_dim = int(self.session.get_inputs()[1].shape[1])
        self.include_agent_id = include_agent_id
        expected = shared_config.MODEL_INPUT_DIM if include_agent_id else shared_config.ENV_OBS_DIM
        if self.input_dim != expected:
            raise ValueError(f"ONNX input dimension {self.input_dim} does not match current contract {expected}")
        self.hidden: Dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self.hidden = {
            f"agent_{i}": np.zeros((1, self.hidden_dim), dtype=np.float32)
            for i in range(shared_config.NUM_AGENTS)
        }
        self.last_q: Dict[str, np.ndarray] = {}

    def q_values(self, agent_id: str, obs: np.ndarray) -> np.ndarray:
        if agent_id not in self.hidden:
            self.reset()
        x = model_input(agent_id, obs, self.include_agent_id).astype(np.float32)[None]
        q, next_hidden = self.session.run(None, {
            "obs": x,
            "hidden_state_in": self.hidden[agent_id],
        })
        self.hidden[agent_id] = next_hidden
        values = q[0]
        self.last_q[agent_id] = values.copy()
        return values

    def select_action(self, agent_id: str, obs: np.ndarray, info: dict) -> int:
        q = self.q_values(agent_id, obs).copy()
        q[np.asarray(info.get("action_mask", [1, 1])) == 0] = -np.inf
        return int(np.argmax(q))
