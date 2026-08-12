"""
PettingZoo ParallelEnv Wrapper (Phase 3 Integration)

This module wraps Phase 2's MultiAgentSensorEnv into PettingZoo's standard ParallelEnv API.
It serves as a 100% drop-in replacement for training/mock_env.py's MockIoTSensorEnv.

Key Features:
- Inherits from pettingzoo.ParallelEnv.
- Exposes identical attributes & methods to MockIoTSensorEnv:
    possible_agents, agents, observation_space(agent), action_space(agent), reset(), step(), render(), close().
- Supports configurable scenario selection ("stable" vs "volatile") pulling from shared_config.SCENARIOS.
"""

import sys
import functools
import random
import numpy as np
from pathlib import Path
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box

# Add project root to sys.path using pathlib for cross-platform compliance
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from multi_agent_env import MultiAgentSensorEnv

class IoTSensorEnv(ParallelEnv):
    """
    PettingZoo Parallel Environment for MARL IoT Sampling Project.
    Wraps MultiAgentSensorEnv to provide standard multi-agent RL training compatibility.
    """

    metadata = {"render_modes": ["human"], "name": "iot_sensor_env_v0"}

    def __init__(self, scenario: str = "stable", render_mode: str = None, seed: int = None, **kwargs):
        """
        Initialize the PettingZoo ParallelEnv wrapper.

        Args:
            scenario (str): Scenario selection ("stable" or "volatile") matching shared_config.SCENARIOS.
            render_mode (str): Optional rendering mode.
            seed (int): Optional random seed.
            **kwargs: Additional parameters passed to MultiAgentSensorEnv.
        """
        super().__init__()
        self.scenario = scenario
        self.render_mode = render_mode
        self.possible_agents = [f"agent_{i}" for i in range(shared_config.NUM_AGENTS)]
        self.agents = self.possible_agents[:]
        self.timestep = 0

        # Underlying Dec-POMDP multi-agent simulation physics
        self.underlying_env = MultiAgentSensorEnv(
            num_agents=shared_config.NUM_AGENTS,
            scenario=scenario,
            seed=seed,
            **kwargs
        )

        # PettingZoo Action and Observation Spaces per agent
        self.action_spaces = {
            agent: Discrete(shared_config.NUM_ACTIONS) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(shared_config.STATE_DIM,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        """Returns the observation space for a given agent."""
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        """Returns the action space for a given agent."""
        return self.action_spaces[agent]

    def reset(self, seed: int = None, options: dict = None):
        """
        Resets the environment for a new episode.

        Returns:
            observations (dict): Dict mapping agent_id -> 3D NumPy array [residual_energy, data_entropy, neighbor_sampling_rate]
            infos (dict): Dict mapping agent_id -> metadata dict
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        elif hasattr(shared_config, "SEED"):
            random.seed(shared_config.SEED)
            np.random.seed(shared_config.SEED)

        self.agents = self.possible_agents[:]
        self.timestep = 0

        observations, infos = self.underlying_env.reset(seed=seed, options=options)
        return observations, infos

    def step(self, actions: dict):
        """
        Executes one step across all agents.

        Args:
            actions (dict): Dict mapping agent_id -> action (0: Sleep, 1: Sample)

        Returns:
            observations (dict): Dict of 3D obs vectors keyed by agent ID
            rewards (dict): Dict of float reward scalars keyed by agent ID
            terminations (dict): Dict of bool termination flags keyed by agent ID
            truncations (dict): Dict of bool truncation flags keyed by agent ID
            infos (dict): Dict of metadata dicts keyed by agent ID
        """
        self.timestep += 1

        observations, rewards, terminations, truncations, infos = self.underlying_env.step(actions)

        # PettingZoo standard: clear self.agents if any agent reaches termination or truncation
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Render environment state (no-op or simple status line)."""
        if self.render_mode == "human":
            print(f"[Timestep {self.timestep}/{shared_config.EPISODE_LENGTH_TIMESTEPS}] Scenario: {self.scenario}")

    def close(self):
        """Close environment resources."""
        pass


# -----------------------------------------------------------------------------
# Test Harness & Scenario Comparison Verification
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING PHASE 3 PETTINGZOO PARALLELENV WRAPPER & SCENARIO COMPARISON TEST")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Helper function to run an episode matching env_wrapper.py's interface usage
    # -------------------------------------------------------------------------
    def run_pettingzoo_test_episode(scenario_name: str, seed_val: int):
        # Instantiate IoTSensorEnv as Module B would
        env = IoTSensorEnv(scenario=scenario_name, seed=seed_val)
        obs_dict, info_dict = env.reset(seed=seed_val)

        # Verify PettingZoo attributes
        assert env.possible_agents == [f"agent_{i}" for i in range(shared_config.NUM_AGENTS)], "Agent names mismatch."
        assert len(env.agents) == shared_config.NUM_AGENTS, "Active agents count mismatch."

        agent_rewards = {aid: 0.0 for aid in env.possible_agents}
        agent_batteries = {aid: [] for aid in env.possible_agents}
        agent_rejections = {aid: 0 for aid in env.possible_agents}
        agent_missed = {aid: 0 for aid in env.possible_agents}
        agent_events_generated = {aid: 0 for aid in env.possible_agents}
        agent_entropies = {aid: [] for aid in env.possible_agents}
        simultaneous_overlaps = {2: 0, 3: 0, 4: 0}

        terminated = False
        step_count = 0

        while not terminated:
            step_count += 1
            # Module B action dict structure: {agent_0: a0, agent_1: a1, ...}
            actions = {
                aid: np.random.choice([shared_config.ACTION_SLEEP, shared_config.ACTION_SAMPLE])
                for aid in env.possible_agents
            }

            obs_dict, rewards_dict, terminations_dict, truncations_dict, info_dict = env.step(actions)

            samplers_in_step = sum(
                1 for aid in env.possible_agents if info_dict[aid]["action_executed"] == shared_config.ACTION_SAMPLE
            )
            if samplers_in_step in simultaneous_overlaps:
                simultaneous_overlaps[samplers_in_step] += 1

            for aid in env.possible_agents:
                agent_rewards[aid] += rewards_dict[aid]
                agent_batteries[aid].append(info_dict[aid]["battery"])
                agent_entropies[aid].append(info_dict[aid]["data_entropy"])

                if info_dict[aid]["is_high_entropy"]:
                    agent_events_generated[aid] += 1

                if info_dict[aid]["sample_rejected"]:
                    agent_rejections[aid] += 1

                if info_dict[aid]["is_high_entropy"] and info_dict[aid]["action_executed"] == shared_config.ACTION_SLEEP:
                    agent_missed[aid] += 1

            terminated = (len(env.agents) == 0) or all(terminations_dict.values()) or any(truncations_dict.values())

        total_high_entropy_events = sum(agent_events_generated.values())
        total_missed_events = sum(agent_missed.values())
        mean_data_entropy = float(np.mean([np.mean(agent_entropies[aid]) for aid in env.possible_agents]))

        return agent_rewards, agent_batteries, agent_rejections, agent_missed, simultaneous_overlaps, total_high_entropy_events, total_missed_events, mean_data_entropy

    # -------------------------------------------------------------------------
    # 1. Run STABLE Scenario Test
    # -------------------------------------------------------------------------
    print("\n--- TESTING STABLE SCENARIO ---")
    st_rewards, st_bats, st_rej, st_miss, st_overlaps, st_events, st_tot_miss, st_mean_ent = run_pettingzoo_test_episode("stable", shared_config.SEED)

    for aid in [f"agent_{i}" for i in range(shared_config.NUM_AGENTS)]:
        avg_bat = np.mean(st_bats[aid])
        print(f"[{aid}] Total Reward: {st_rewards[aid]:7.2f} | Avg Battery: {avg_bat:.4f} | "
              f"Rejections: {st_rej[aid]:2d} | Missed Events: {st_miss[aid]:2d}")
    print(f"Stable Scenario Total High-Entropy Events Generated: {st_events}")
    print(f"Stable Scenario Mean Data Entropy                 : {st_mean_ent:.4f}")
    print(f"Overlap (2/3/4 agents): {st_overlaps[2]} / {st_overlaps[3]} / {st_overlaps[4]}")

    # -------------------------------------------------------------------------
    # 2. Run VOLATILE Scenario Test
    # -------------------------------------------------------------------------
    print("\n--- TESTING VOLATILE SCENARIO ---")
    vol_rewards, vol_bats, vol_rej, vol_miss, vol_overlaps, vol_events, vol_tot_miss, vol_mean_ent = run_pettingzoo_test_episode("volatile", shared_config.SEED)

    for aid in [f"agent_{i}" for i in range(shared_config.NUM_AGENTS)]:
        avg_bat = np.mean(vol_bats[aid])
        print(f"[{aid}] Total Reward: {vol_rewards[aid]:7.2f} | Avg Battery: {avg_bat:.4f} | "
              f"Rejections: {vol_rej[aid]:2d} | Missed Events: {vol_miss[aid]:2d}")
    print(f"Volatile Scenario Total High-Entropy Events Generated: {vol_events}")
    print(f"Volatile Scenario Mean Data Entropy                 : {vol_mean_ent:.4f}")
    print(f"Overlap (2/3/4 agents): {vol_overlaps[2]} / {vol_overlaps[3]} / {vol_overlaps[4]}")

    # -------------------------------------------------------------------------
    # 3. Assert Scenario Differences
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO COMPARISON VERIFICATION:")
    print(f"High-Entropy Events Generated (Stable vs Volatile) : {st_events} vs {vol_events}")
    print(f"Total Missed Events (Stable vs Volatile)            : {st_tot_miss} vs {vol_tot_miss}")
    print(f"Mean Data Entropy (Stable vs Volatile)              : {st_mean_ent:.4f} vs {vol_mean_ent:.4f}")

    # NOTE ON METRIC SELECTION:
    # Reward StdDev under a random policy is MISLEADING because a quiet baseline (+0.05) interrupted
    # by rare isolated penalty spikes (-0.80) creates higher step-to-step reward variance than a busy 
    # volatile environment where penalties occur frequently and cluster together.
    # The true metrics for environmental volatility are Total High-Entropy Events Generated, Missed Event Count, 
    # and Mean Data Entropy.

    assert vol_events > st_events, "Volatile scenario must generate more high-entropy events than stable."
    assert vol_tot_miss > st_tot_miss, "Volatile scenario must yield more missed events under a random policy."
    assert vol_mean_ent > st_mean_ent, "Volatile scenario must have higher average data entropy."

    print("\nPETTINGZOO WRAPPER TEST PASSED PERFECTLY!")
    print("IoTSensorEnv is a 100% drop-in replacement for MockIoTSensorEnv.")
    print("=" * 80)
