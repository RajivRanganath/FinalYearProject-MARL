"""
PettingZoo ParallelEnv Standard Integration
MARL Adaptive IoT Sampling Project

This module wraps MultiAgentSensorEnv into PettingZoo's standard ParallelEnv API,
providing native compatibility with EPyMARL, SuperSuit, and multi-agent RL libraries.
"""

import sys
import functools
import random
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from environment.multi_agent_env import MultiAgentSensorEnv

class IoTSensorEnv(ParallelEnv):
    """
    Standard PettingZoo ParallelEnv environment for multi-agent IoT sensor coordination.
    """

    metadata = {"render_modes": ["human"], "name": "iot_sensor_env_v1"}

    def __init__(
        self,
        scenario: str = "stable",
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        topology: str = "ring",
        **kwargs
    ):
        super().__init__()
        self.scenario = scenario
        self.render_mode = render_mode
        self.possible_agents = [f"agent_{i}" for i in range(shared_config.NUM_AGENTS)]
        self.agents = self.possible_agents[:]
        self.timestep = 0

        self.underlying_env = MultiAgentSensorEnv(
            num_agents=shared_config.NUM_AGENTS,
            scenario=scenario,
            seed=seed,
            topology=topology,
            **kwargs
        )

        self.action_spaces = {
            agent: Discrete(shared_config.NUM_ACTIONS)
            for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: Box(
                low=0.0,
                high=1.0,
                shape=(shared_config.ENV_OBS_DIM,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Box:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Discrete:
        return self.action_spaces[agent]

    def action_mask(self, agent: str) -> np.ndarray:
        """Returns action availability mask for specified agent."""
        return self.underlying_env.agents[agent].get_action_mask()

    def get_avail_actions(self) -> Dict[str, np.ndarray]:
        """Returns action availability masks for all agents."""
        return self.underlying_env.get_avail_actions()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Resets environment and returns initial observation and info dictionaries.
        """
        if seed is not None:
            self.underlying_env.seed(seed)
        elif hasattr(shared_config, "SEED"):
            self.underlying_env.seed(shared_config.SEED)

        self.agents = self.possible_agents[:]
        self.timestep = 0

        observations, infos = self.underlying_env.reset(seed=seed, options=options)
        return observations, infos

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Executes one parallel step for all active agents.
        """
        self.timestep += 1
        obs, rewards, terms, truncs, infos = self.underlying_env.step(actions)

        # Remove terminated agents if any
        if all(terms.values()) or all(truncs.values()):
            self.agents = []

        return obs, rewards, terms, truncs, infos

    def render(self):
        """Minimal text render of current agent battery and entropy states."""
        if self.render_mode == "human":
            states = [
                f"{aid}: Bat={self.underlying_env.agents[aid].battery:.2f}, Ent={self.underlying_env.agents[aid].current_entropy:.2f}, AoI={self.underlying_env.agents[aid].aoi}"
                for aid in self.possible_agents
            ]
            print(f"[Step {self.timestep}] | " + " | ".join(states))

    def close(self):
        pass

    def state(self) -> np.ndarray:
        """Global state vector for Centralized Training with Decentralized Execution (CTDE)."""
        obs_list = [
            np.array([
                self.underlying_env.agents[aid].battery,
                self.underlying_env.agents[aid].current_entropy,
                self.underlying_env._get_neighbor_sampling_rate(aid)
            ], dtype=np.float32)
            for aid in self.possible_agents
        ]
        return np.concatenate(obs_list)
