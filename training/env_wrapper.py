import sys
import os
import numpy as np

# Make sure we can import from epymarl and training dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "epymarl", "src"))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mock_env import MockIoTSensorEnv
from envs.multiagentenv import MultiAgentEnv
import shared_config

class IoTSensorEnvWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.env = MockIoTSensorEnv()
        self.n_agents = shared_config.NUM_AGENTS
        self.episode_limit = shared_config.EPISODE_LENGTH_TIMESTEPS
        self.n_actions = shared_config.NUM_ACTIONS
        self._obs = None
        self.reset()

    def step(self, actions):
        """
        actions: list or array of size n_agents
        Returns reward, terminated, info
        """
        pz_actions = {f"agent_{i}": int(actions[i]) for i in range(self.n_agents)}
        obs_dict, reward_dict, terminations_dict, truncations_dict, info_dict = self.env.step(pz_actions)
        
        self._obs = obs_dict
        
        # Centralized reward (sum of individual rewards) for cooperative MARL
        team_reward = sum(reward_dict.values())
        
        terminated = all(terminations_dict.values()) or any(truncations_dict.values())
        info = {"episode_limit": self.episode_limit}
        
        return team_reward, terminated, info

    def get_obs(self):
        return [self._obs[f"agent_{i}"] for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        return self._obs[f"agent_{agent_id}"]

    def get_obs_size(self):
        return shared_config.STATE_DIM

    def get_state(self):
        # Global state. For CTDE, we concatenate local observations.
        return np.concatenate(self.get_obs())

    def get_state_size(self):
        return shared_config.STATE_DIM * self.n_agents

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        # All actions available
        return [1] * self.n_actions

    def get_total_actions(self):
        return self.n_actions

    def reset(self, seed=None, options=None):
        obs_dict, _ = self.env.reset(seed=seed)
        self._obs = obs_dict
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.reset(seed=seed)

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
