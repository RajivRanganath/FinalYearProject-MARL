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
        seed = kwargs.get("seed", shared_config.SEED)
        
        self.env = IoTSensorEnv(scenario=scenario, seed=seed)
        self.n_agents = shared_config.NUM_AGENTS
        self.n_actions = shared_config.NUM_ACTIONS
        self.episode_limit = shared_config.EPISODE_LENGTH_TIMESTEPS
        self._obs: Dict[str, np.ndarray] = {}
        self._avail_actions: Dict[str, np.ndarray] = {}
        self.reset(seed=seed)

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
        env_info = {"episode_limit": self.episode_limit}

        # Returns: None (for obs), team_reward, terminated, truncated (False), env_info
        return None, team_reward, terminated, False, env_info

    def get_obs(self) -> List[np.ndarray]:
        """Returns list of 3D local observations for each agent."""
        return [self._obs[f"agent_{i}"] for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        """Returns 3D local observation for specified agent index."""
        return self._obs[f"agent_{agent_id}"]

    def get_obs_size(self) -> int:
        """Returns size of local observation vector (3)."""
        return shared_config.ENV_OBS_DIM

    def get_state(self) -> np.ndarray:
        """Returns global centralized state vector (12 floats)."""
        return np.concatenate(self.get_obs())

    def get_state_size(self) -> int:
        """Returns size of global centralized state vector (12)."""
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
        self._obs = obs_dict
        self._avail_actions = self.env.get_avail_actions()
        return self.get_obs(), self.get_state()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed: Optional[int] = None):
        self.env.reset(seed=seed)
