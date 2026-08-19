"""
EPyMARL MultiAgentEnv Wrapper for IoTSensorEnv
MARL Adaptive IoT Sampling Project

Standardizes the PettingZoo IoTSensorEnv into EPyMARL's MultiAgentEnv interface:
- Strict Action Feasibility Masking: Disables ACTION_SAMPLE when battery is below sampling cost
- Canonical Reward Propagation: Optimizes the true environment objective without divergent surrogate hacks
- Global State Construction for CTDE: Formulates 12-dimensional centralized state vector
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Cross-platform path setup
_THIS_DIR = Path(__file__).resolve().parent
_EPYMARL_SRC = _THIS_DIR / "epymarl" / "src"
_PROJECT_ROOT = _THIS_DIR.parent

for p in [str(_THIS_DIR), str(_EPYMARL_SRC), str(_PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.append(p)

import shared_config
from environment.pettingzoo_env import IoTSensorEnv
from envs.multiagentenv import MultiAgentEnv

class IoTSensorEnvWrapper(MultiAgentEnv):
    """
    EPyMARL MultiAgentEnv implementation wrapping IoTSensorEnv.
    """

    def __init__(self, **kwargs):
        scenario = kwargs.get("scenario", "stable")
        regime = kwargs.get("regime", "independent")
        ablation = kwargs.get("ablation", "full")
        seed = kwargs.get("seed", shared_config.SEED)
        self.mask_neighbor_signal = bool(kwargs.get("mask_neighbor_signal", False))
        
        self.env = IoTSensorEnv(scenario=scenario, regime=regime, ablation=ablation, seed=seed)
        self.scenario = scenario
        self.regime = regime
        self.ablation = ablation
        self.n_agents = shared_config.NUM_AGENTS
        self.n_actions = shared_config.NUM_ACTIONS
        self.episode_limit = shared_config.EPISODE_LENGTH_TIMESTEPS
        self._obs: Dict[str, np.ndarray] = {}
        self._avail_actions: Dict[str, np.ndarray] = {}
        self._reset_episode_stats()
        self.reset(seed=seed)

    def _reset_episode_stats(self) -> None:
        self._episode_steps = 0
        self._episode_events = 0
        self._episode_sample_requests = 0
        self._episode_samples = 0
        self._episode_channel_blocks = 0
        self._episode_coverage = 0.0
        self._component_totals: Dict[str, float] = {}

    def step(self, actions: List[int]) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Executes one joint step across all agents.
        Args:
            actions: List of integer actions of length n_agents.
        Returns:
            team_reward, terminated, env_info
        """
        pz_actions = {f"agent_{i}": int(actions[i]) for i in range(self.n_agents)}
        obs_dict, reward_dict, terms, truncs, info_dict = self.env.step(pz_actions)
        self._obs = obs_dict
        self._avail_actions = self.env.get_avail_actions()
        team_reward = float(sum(reward_dict.values()))
        terminated = all(terms.values()) or all(truncs.values())
        self._episode_steps += 1
        self._episode_sample_requests += sum(int(a) == shared_config.ACTION_SAMPLE for a in actions)
        self._episode_events += sum(int(info["is_high_entropy"]) for info in info_dict.values())
        self._episode_samples += sum(int(info["sample_delivered"]) for info in info_dict.values())
        self._episode_channel_blocks += sum(int(info.get("channel_blocked", False)) for info in info_dict.values())
        self._episode_coverage += float(np.mean([info.get("network_coverage", 1.0) for info in info_dict.values()]))
        for info in info_dict.values():
            for name, value in info.get("reward_components", {}).items():
                self._component_totals[name] = self._component_totals.get(name, 0.0) + float(value)

        total_slots = max(1, self._episode_steps * self.n_agents)
        env_info = {
            "episode_limit": bool(terminated),
            "event_fraction": self._episode_events / total_slots,
            "sample_request_fraction": self._episode_sample_requests / total_slots,
            "sample_fraction": self._episode_samples / total_slots,
            "channel_block_fraction": self._episode_channel_blocks / total_slots,
            "network_coverage": self._episode_coverage / max(1, self._episode_steps),
        }
        env_info.update({f"reward_{name}": value for name, value in self._component_totals.items()})

        # Returns: None (for obs), team_reward, terminated, truncated (False), env_info
        return None, team_reward, terminated, False, env_info

    def get_obs(self) -> List[np.ndarray]:
        """Returns causal local observations for each agent."""
        return [self._policy_obs(self._obs[f"agent_{i}"]) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        """Returns 3D local observation for specified agent index."""
        return self._policy_obs(self._obs[f"agent_{agent_id}"])

    def _policy_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply a declared training-only input mask without changing physics."""
        if not self.mask_neighbor_signal:
            return obs
        masked = np.asarray(obs, dtype=np.float32).copy()
        masked[shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE] = 0.0
        return masked

    def get_obs_size(self) -> int:
        """Returns the configured local observation size."""
        return shared_config.ENV_OBS_DIM

    def get_state(self) -> np.ndarray:
        """Returns the causally valid centralized state vector."""
        return np.concatenate(self.get_obs())

    def get_state_size(self) -> int:
        """Returns the configured centralized state size."""
        return shared_config.GLOBAL_STATE_DIM

    def get_avail_actions(self) -> List[List[int]]:
        """Returns action availability mask list for all agents."""
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id: int) -> List[int]:
        """
        Returns action mask [can_sleep, can_sample] for agent_id.
        Enforces energy feasibility: sample is 0 when battery < cost.
        """
        aid = f"agent_{agent_id}"
        if aid in self._avail_actions:
            return list(self._avail_actions[aid])
        return [1, 1]

    def get_total_actions(self) -> int:
        """Returns total discrete actions (2)."""
        return self.n_actions

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """Resets environment and returns initial observations and global state."""
        obs_dict, infos = self.env.reset(seed=seed)
        self._reset_episode_stats()
        self._obs = obs_dict
        self._avail_actions = self.env.get_avail_actions()
        return self.get_obs(), self.get_state()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed: Optional[int] = None):
        self.env.reset(seed=seed)
