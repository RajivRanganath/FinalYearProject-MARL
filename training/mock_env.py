import functools
import random
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
import sys
import os

# Add parent directory to path to import shared_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shared_config

class MockIoTSensorEnv(ParallelEnv):
    """
    Mock PettingZoo Parallel Environment for MARL IoT Sampling Project.
    Used for Module B to develop training pipeline before Module A finishes the real environment.
    """
    metadata = {"render_modes": ["human"], "name": "mock_iot_env_v0"}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.possible_agents = [f"agent_{i}" for i in range(shared_config.NUM_AGENTS)]
        self.agents = self.possible_agents[:]
        self.timestep = 0
        
        # Action space: 0 (Sleep), 1 (Sample Now)
        self.action_spaces = {agent: Discrete(shared_config.NUM_ACTIONS) for agent in self.possible_agents}
        
        # State space: [residual_energy, data_entropy, neighbor_sampling_rate]
        # all normalized 0 to 1
        self.observation_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(shared_config.STATE_DIM,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        elif hasattr(shared_config, "SEED"):
            random.seed(shared_config.SEED)
            np.random.seed(shared_config.SEED)
            
        self.agents = self.possible_agents[:]
        self.timestep = 0
        
        observations = {
            agent: np.random.uniform(0, 1, size=(shared_config.STATE_DIM,)).astype(np.float32)
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        self.timestep += 1
        
        # Generate random observations
        observations = {
            agent: np.random.uniform(0, 1, size=(shared_config.STATE_DIM,)).astype(np.float32)
            for agent in self.agents
        }
        
        # Placeholder reward function: 
        # small positive reward for sampling when random entropy > 0.5
        # small negative reward otherwise
        rewards = {}
        for agent in self.agents:
            entropy = observations[agent][shared_config.STATE_INDEX_DATA_ENTROPY]
            action = actions[agent]
            
            if action == shared_config.ACTION_SAMPLE:
                rewards[agent] = 0.1 if entropy > 0.5 else -0.1
            else:
                # Penalty for missing high entropy events
                rewards[agent] = -0.1 if entropy > 0.5 else 0.05
                
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.timestep >= shared_config.EPISODE_LENGTH_TIMESTEPS for agent in self.agents}
        
        if any(truncations.values()):
            self.agents = []
            
        infos = {agent: {} for agent in self.possible_agents}
        
        return observations, rewards, terminations, truncations, infos
