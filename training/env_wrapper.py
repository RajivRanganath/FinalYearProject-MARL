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
        self.n_actions = shared_config.NUM_ACTIONS
        self._obs = None
        self.reset()

    def _shape_reward(self, reward_dict, info_dict, old_obs_dict):
        """
        Training-only reward shaping.

        CRITICAL FIX: Evaluate action against OLD entropy (the state the agent actually saw).
        Previously, the shaping evaluated the action against the NEW entropy (which includes
        random unpredictable spikes), meaning the agent was randomly punished/rewarded
        regardless of its actual decision.
        """
        shaped = {}
        for agent_id in reward_dict:
            info = info_dict[agent_id]
            # Use the entropy the agent OBSERVED when choosing the action!
            old_entropy = old_obs_dict[agent_id][1]
            action = info["action_executed"]
            rejected = info["sample_rejected"]

            if old_entropy > 0.3:
                # High Entropy
                if action == shared_config.ACTION_SAMPLE and not rejected:
                    shaped[agent_id] = 1.0
                else:
                    shaped[agent_id] = -1.0
            else:
                # Low Entropy
                if action == shared_config.ACTION_SAMPLE:
                    shaped[agent_id] = -0.01
                else:
                    shaped[agent_id] = 0.0 # ZERO to stop gradient domination!
                
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
        # CRITICAL FIX 3: THE EXPLORATION BATTERY STARVATION HACK
        # Under random exploration (epsilon-greedy), the agent samples randomly 
        # and drains its battery within 20 steps. For the rest of the 288-step 
        # episode, it has no battery, so all 'Sample' attempts are rejected.
        # This physically prevents it from exploring 'Sample' when the rare 
        # high-entropy spikes finally occur.
        # SOLUTION: We use domain randomization to artificially restore battery 
        # during training so exploration is never starved, while still letting 
        # the network see a variety of battery states so it generalizes to deployment.
        # ----------------------------------------------------------------------
        if self.use_reward_shaping:
            import numpy as np
            # Unwrap PettingZoo wrappers to get to the base MultiAgentEnv
            for agent_id, sa_env in self.env.unwrapped.underlying_env.agents.items():
                # Force balanced data distribution (50% high entropy spikes)
                # This solves the MLP gradient starvation issue on the 4% minority class
                sa_env.spike_prob = 0.5
                
                # Give a random battery level across the entire spectrum
                new_bat = np.random.uniform(0.0, 1.0)
                sa_env.battery = float(new_bat)
                # Update the observation so the agent sees the new battery next step
                obs_dict[agent_id][0] = new_bat

        self._obs = obs_dict

        # Compute team reward: use true unshaped rewards for IQL
        # We don't need manual shaping anymore since IQL solves the credit assignment issue!
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
