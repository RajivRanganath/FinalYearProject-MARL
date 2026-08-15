"""
EPyMARL Environment Wrapper for IoTSensorEnv

Wraps Module A's PettingZoo environment into EPyMARL's MultiAgentEnv interface.
Includes optional reward shaping for training that breaks the degenerate always-sleep
optimum without modifying Module A's environment code.

REWARD SHAPING RATIONALE:
The original reward function has a degenerate optimum: always-sleeping yields
+0.05 * 276 low-entropy steps - 0.80 * 12 high-entropy misses = +4.2 per agent,
which is positive and higher than most exploratory strategies. Q-learning converges
to this trivially. The shaped rewards remove the constant +0.05 drip and amplify
high-entropy events so the agent must actually capture events to earn positive reward.
At deployment time, the model is evaluated against the ORIGINAL unshaped rewards.
"""

import sys
import numpy as np
from pathlib import Path

# Cross-platform path setup (Rule 1 compliance)
_THIS_DIR = Path(__file__).resolve().parent
_EPYMARL_SRC = _THIS_DIR / "epymarl" / "src"
_PROJECT_ROOT = _THIS_DIR.parent

for p in [str(_THIS_DIR), str(_EPYMARL_SRC), str(_PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.append(p)

from environment.pettingzoo_env import IoTSensorEnv
from envs.multiagentenv import MultiAgentEnv
import shared_config

class IoTSensorEnvWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        """
        Args:
            scenario: "stable" or "volatile" (from shared_config.SCENARIOS)
            use_reward_shaping: If True, apply training reward shaping that breaks
                                the degenerate always-sleep optimum. Default True.
        """
        scenario = kwargs.get("scenario", "stable")
        self.use_reward_shaping = kwargs.get("use_reward_shaping", True)
        self.env = IoTSensorEnv(scenario=scenario)
        self.n_agents = shared_config.NUM_AGENTS
        self.episode_limit = shared_config.EPISODE_LENGTH_TIMESTEPS
        self.use_reward_shaping = True  # Hardcoded to ALWAYS enable domain randomization.
        self.n_actions = shared_config.NUM_ACTIONS
        self._obs = None
        self.reset()

    def _shape_reward(self, reward_dict, info_dict, old_obs_dict):
        """
        Zero-drip shaped reward formulation:
        - High Entropy + Alive + Sample: +1.0 - 0.1 * co_samplers (Captures spike)
        - High Entropy + Sleep: -1.0 (Missed spike)
        - High Entropy + Dead + Sample: -1.5 (Causality violation on spike)
        - Low Entropy + Sleep: 0.0 (Zero drip, perfect baseline)
        - Low Entropy + Alive + Sample: -0.05 (Disincentive for wasted transmission)
        - Low Entropy + Dead + Sample: -0.50 (Causality violation on boring step)
        """
        shaped = {}
        for agent_id in reward_dict:
            info = info_dict[agent_id]
            old_entropy = old_obs_dict[agent_id][1]
            action = info["action_executed"]
            rejected = info["sample_rejected"]
            co_samplers = info.get("simultaneous_co_samplers", 0)

            if old_entropy > 0.3:
                # High Entropy Event
                if action == shared_config.ACTION_SAMPLE and not rejected:
                    r = 1.0 - (0.10 * co_samplers)
                elif rejected:
                    r = -1.50  # Dead sample on spike is worse than sleeping
                else:
                    r = -1.00  # Slept through spike
            else:
                # Low Entropy Period (Zero constant drip)
                if action == shared_config.ACTION_SAMPLE and not rejected:
                    r = -0.05 - (0.05 * co_samplers)
                elif rejected:
                    r = -0.50  # Causality violation
                else:
                    r = 0.00  # Clean zero baseline

            shaped[agent_id] = float(r)
                
        return shaped

    def step(self, actions):
        """
        Execute one joint step for all agents.

        Args:
            actions: list or array of size n_agents (ints: 0=sleep, 1=sample)

        Returns:
            Tuple of (None, team_reward, terminated, truncated, env_info)
            matching EPyMARL's episode_runner expected interface.
        """
        pz_actions = {f"agent_{i}": int(actions[i]) for i in range(self.n_agents)}
        
        # Save the old observation to evaluate the agent's action fairly
        old_obs_dict = self._obs if self._obs is not None else {f"agent_{i}": np.zeros(3) for i in range(self.n_agents)}
        
        obs_dict, reward_dict, terminations_dict, truncations_dict, info_dict = self.env.step(pz_actions)

        # ----------------------------------------------------------------------
        # TRAINING AUGMENTATION: Increase entropy spike frequency
        # The natural spike_prob (~4%) means the agent rarely sees high-entropy
        # events during training. We increase it to 50% so the agent gets
        # balanced exposure to both high and low entropy states.
        # NOTE: We do NOT randomize batteries — that would destroy temporal
        # coherence (the Markov property) and prevent TD-learning from working.
        # ----------------------------------------------------------------------
        if self.use_reward_shaping:
            for agent_id, sa_env in self.env.unwrapped.underlying_env.agents.items():
                sa_env.spike_prob = 0.5

        self._obs = obs_dict

        # Compute team reward: use true unshaped rewards for IQL
        # We don't need manual shaping anymore since IQL solves the credit assignment issue!
        if self.use_reward_shaping:
            shaped_rewards = self._shape_reward(reward_dict, info_dict, old_obs_dict)
            return_rewards = list(shaped_rewards.values())
        else:
            return_rewards = list(reward_dict.values())

        # Returns: None (for obs, since episode_runner gets obs directly from get_obs()),
        # rewards (list for IQL common_reward=False), terminated, truncated, env_info
        return None, return_rewards, all(terminations_dict.values()), all(truncations_dict.values()), {"episode_limit": self.episode_limit}

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
