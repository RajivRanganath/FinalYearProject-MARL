"""
Multi-Agent Sensor Environment (Phase 2 Extension)

This module extends single_agent_env.py to a 4-agent Decentralized POMDP (Dec-POMDP) environment.

Key Features:
- 4 Independent Sensor Agents ("agent_0" through "agent_3") matching NUM_AGENTS in shared_config.py.
- Independent solar harvesting & data entropy per agent (uncorrelated cloud noise & distinct event spikes).
- Partial Observability Boundary strictly enforced:
    Each agent observes EXACTLY 3 values: [residual_energy, data_entropy, neighbor_sampling_rate].
    No agent ever observes neighbor battery levels or neighbor data entropy.
- Neighbor Sampling Rate: Rolling window average (default 12 steps) of executed samples across the OTHER 3 agents.
- Coordination Reward: Individual reward + modest overlap penalty for redundant simultaneous sampling.
- Dictionary-based interface keyed by agent ID ("agent_0" .. "agent_3") ready for PettingZoo wrapping in Phase 3.
"""

import sys
import random
import numpy as np
from pathlib import Path
from collections import deque

# Add project root to sys.path using pathlib for cross-platform compliance
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from environment.single_agent_env import SingleAgentSensorEnv

class MultiAgentSensorEnv:
    """
    4-Agent IoT Sensor Environment simulating decentralized sampling coordination
    under energy harvesting and partial observability constraints.
    """

    def __init__(
        self,
        num_agents: int = shared_config.NUM_AGENTS,
        window_size: int = 12,
        coordination_penalty_weight: float = 0.10,
        scenario: str = "stable",
        seed: int = None
    ):
        """
        Initialize multi-agent environment parameters.

        Args:
            num_agents: Number of IoT sensor nodes (defaults to shared_config.NUM_AGENTS = 4).
            window_size: Rolling window size (in timesteps) for calculating neighbor sampling rate.
            coordination_penalty_weight: Penalty scalar applied per co-sampling neighbor to discourage overlap.
            scenario: Scenario configuration key ("stable" or "volatile").
            seed: Optional random seed for reproducible multi-agent runs.
        """
        self.num_agents = num_agents
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        self.window_size = window_size
        self.coordination_penalty_weight = coordination_penalty_weight
        self.scenario = scenario
        self.episode_length = shared_config.EPISODE_LENGTH_TIMESTEPS

        # Instantiate independent SingleAgentSensorEnv instances for each node
        # Each agent gets its own distinct seed offset to ensure non-identical cloud noise and entropy spikes
        base_seed = seed if seed is not None else shared_config.SEED
        self.agents = {}
        for i, agent_id in enumerate(self.agent_ids):
            agent_seed = base_seed + (i * 100) if base_seed is not None else None
            self.agents[agent_id] = SingleAgentSensorEnv(
                scenario=scenario,
                seed=agent_seed
            )

        # Rolling history of executed sample actions per agent: 1 if executed sample, 0 otherwise
        self.action_history = {agent_id: deque(maxlen=self.window_size) for agent_id in self.agent_ids}
        self.timestep = 0

        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int = None):
        """Set random seeds across all internal single-agent instances."""
        if seed is None:
            seed = shared_config.SEED
        random.seed(seed)
        np.random.seed(seed)
        for i, agent_id in enumerate(self.agent_ids):
            self.agents[agent_id].seed(seed + (i * 100))

    def _get_neighbor_sampling_rate(self, target_agent_id: str) -> float:
        """
        Calculates the rolling average sampling rate of the OTHER (N-1) agents over the window.

        PARTIAL OBSERVABILITY RULE:
        This function calculates a single scalar summary of neighbor ACTIVITY over time.
        It does NOT expose neighbor battery levels or neighbor entropy values.

        Args:
            target_agent_id (str): The agent ID requesting its neighbor observation.

        Returns:
            float: Normalized sampling rate in [0.0, 1.0] across all neighbors in window.
        """
        other_agents = [aid for aid in self.agent_ids if aid != target_agent_id]
        total_executed_samples = 0
        total_possible_slots = len(other_agents) * self.window_size

        for aid in other_agents:
            total_executed_samples += sum(self.action_history[aid])

        if total_possible_slots == 0:
            return 0.0

        neighbor_rate = total_executed_samples / total_possible_slots
        return float(np.clip(neighbor_rate, 0.0, 1.0))

    def reset(self, seed: int = None, options: dict = None):
        """
        Resets environment state for all 4 agents.

        Returns:
            observations (dict): Dict of np.ndarray [residual_energy, data_entropy, neighbor_sampling_rate]
                                 keyed by agent ID ("agent_0" .. "agent_3").
            infos (dict): Initial metadata dicts keyed by agent ID.
        """
        if seed is not None:
            self.seed(seed)

        self.timestep = 0
        observations = {}
        infos = {}

        # Reset rolling action history to 0s for all agents
        for agent_id in self.agent_ids:
            self.action_history[agent_id].clear()
            for _ in range(self.window_size):
                self.action_history[agent_id].append(0)

        # Reset each underlying single-agent environment
        for agent_id in self.agent_ids:
            # Re-seed sub-agents deterministically if seed provided
            sub_seed = (seed + int(agent_id.split('_')[1]) * 100) if seed is not None else None
            sa_obs, sa_info = self.agents[agent_id].reset(seed=sub_seed)

            # sa_obs is [battery, entropy]
            # Now compute neighbor sampling rate (initially 0.0)
            neighbor_rate = self._get_neighbor_sampling_rate(agent_id)

            # Combine into 3D state vector: [residual_energy, data_entropy, neighbor_sampling_rate]
            obs_3d = np.array([sa_obs[0], sa_obs[1], neighbor_rate], dtype=np.float32)
            observations[agent_id] = obs_3d
            infos[agent_id] = sa_info

        return observations, infos

    def step(self, actions: dict):
        """
        Executes one joint environment step for all 4 agents.

        Args:
            actions (dict): Dict mapping agent_id -> action (0: Sleep, 1: Sample)

        Returns:
            observations (dict): Dict of 3D obs vectors keyed by agent ID
            rewards (dict): Dict of float reward scalars keyed by agent ID
            terminations (dict): Dict of bool termination flags keyed by agent ID
            truncations (dict): Dict of bool truncation flags (False) keyed by agent ID
            infos (dict): Dict of metadata dicts keyed by agent ID
        """
        assert isinstance(actions, dict), "Actions must be a dictionary keyed by agent ID."
        for agent_id in self.agent_ids:
            assert agent_id in actions, f"Missing action for {agent_id} in step()."

        self.timestep += 1

        # Phase 1: Step each single agent independently to apply physics, harvesting & causality checks
        sa_results = {}
        executed_samples_count = 0

        for agent_id in self.agent_ids:
            action = actions[agent_id]
            obs_2d, ind_reward, term, trunc, info = self.agents[agent_id].step(action)
            
            # Record whether sample was successfully executed (action requested was SAMPLE and NOT rejected)
            executed_sample = 1 if (info["action_executed"] == shared_config.ACTION_SAMPLE and not info["sample_rejected"]) else 0
            self.action_history[agent_id].append(executed_sample)
            
            if executed_sample == 1:
                executed_samples_count += 1

            sa_results[agent_id] = {
                "obs_2d": obs_2d,
                "ind_reward": ind_reward,
                "term": term,
                "trunc": trunc,
                "info": info,
                "executed_sample": executed_sample
            }

        # Phase 2: Compute joint coordination rewards & 3D state observations
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent_id in self.agent_ids:
            res = sa_results[agent_id]

            # Coordination Component:
            # If this agent successfully sampled, count how many OTHER agents also sampled simultaneously
            co_samplers = executed_samples_count - res["executed_sample"]
            
            if res["executed_sample"] == 1 and co_samplers > 0:
                # Modest penalty for redundant simultaneous sampling
                # Weight = 0.10 per co-sampling neighbor
                coordination_penalty = self.coordination_penalty_weight * co_samplers
            else:
                coordination_penalty = 0.0

            total_reward = res["ind_reward"] - coordination_penalty

            # Compute updated neighbor sampling rate over rolling window
            neighbor_rate = self._get_neighbor_sampling_rate(agent_id)

            # Strictly enforce 3D State Vector: [residual_energy, data_entropy, neighbor_sampling_rate]
            obs_3d = np.array([res["obs_2d"][0], res["obs_2d"][1], neighbor_rate], dtype=np.float32)

            observations[agent_id] = obs_3d
            rewards[agent_id] = float(total_reward)
            terminations[agent_id] = res["term"]
            truncations[agent_id] = res["trunc"]

            # Merge metadata
            info = res["info"]
            info["coordination_penalty"] = coordination_penalty
            info["simultaneous_co_samplers"] = co_samplers if res["executed_sample"] == 1 else 0
            info["neighbor_sampling_rate"] = neighbor_rate
            infos[agent_id] = info

        return observations, rewards, terminations, truncations, infos


# -----------------------------------------------------------------------------
# Test Harness & Partial Observability Boundary Verification
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 75)
    print("RUNNING PHASE 2 MULTI-AGENT ENVIRONMENT SANITY & PARTIAL OBSERVABILITY TEST")
    print("=" * 75)

    env = MultiAgentSensorEnv(scenario="stable", seed=shared_config.SEED)
    obs_dict, info_dict = env.reset(seed=shared_config.SEED)

    # Verification 1: Check initial observations match NUM_AGENTS and STATE_DIM
    assert len(obs_dict) == shared_config.NUM_AGENTS, f"Expected {shared_config.NUM_AGENTS} agents."
    for agent_id, obs in obs_dict.items():
        assert len(obs) == shared_config.STATE_DIM, \
            f"Observation length for {agent_id} is {len(obs)}, expected exactly {shared_config.STATE_DIM}."
        assert obs.shape == (shared_config.STATE_DIM,), f"Observation shape mismatch: {obs.shape}"

    # Track metrics across a full 288-step episode with a random policy
    agent_rewards = {aid: 0.0 for aid in env.agent_ids}
    agent_batteries = {aid: [] for aid in env.agent_ids}
    agent_rejections = {aid: 0 for aid in env.agent_ids}
    agent_missed = {aid: 0 for aid in env.agent_ids}
    simultaneous_overlaps = {2: 0, 3: 0, 4: 0}  # Overlap counter for 2, 3, or 4 agents sampling together

    terminated = False
    step_count = 0

    while not terminated:
        step_count += 1
        # Generate random joint actions for all 4 agents
        joint_actions = {
            aid: np.random.choice([shared_config.ACTION_SLEEP, shared_config.ACTION_SAMPLE])
            for aid in env.agent_ids
        }

        obs_dict, rewards_dict, terminations_dict, truncations_dict, info_dict = env.step(joint_actions)

        # Count simultaneous executed samples
        samplers_in_step = sum(1 for aid in env.agent_ids if info_dict[aid]["action_executed"] == shared_config.ACTION_SAMPLE)
        if samplers_in_step in simultaneous_overlaps:
            simultaneous_overlaps[samplers_in_step] += 1

        for aid in env.agent_ids:
            agent_obs = obs_dict[aid]
            agent_info = info_dict[aid]

            # -----------------------------------------------------------------
            # CRITICAL PARTIAL OBSERVABILITY ASSERTIONS:
            # 1. Observation array must be strictly length 3 (STATE_DIM).
            # 2. Index 0 is OWN residual_energy.
            # 3. Index 1 is OWN data_entropy.
            # 4. Index 2 is scalar neighbor_sampling_rate.
            # 5. NO neighbor battery levels or neighbor entropy values exist in the observation array.
            # -----------------------------------------------------------------
            assert len(agent_obs) == 3, f"Partial Observability Violation: Obs size is {len(agent_obs)}, expected 3."
            assert agent_obs[0] == agent_info["battery"], "Obs[0] must match agent's own battery."
            assert agent_obs[1] == agent_info["data_entropy"], "Obs[1] must match agent's own entropy."
            assert 0.0 <= agent_obs[2] <= 1.0, "Obs[2] neighbor rate must be normalized between 0 and 1."

            # Assert battery stays strictly within [0.0, 1.0] for all agents at all steps
            assert 0.0 <= agent_info["battery"] <= 1.0, f"Battery out of bounds for {aid}: {agent_info['battery']}"

            # Accumulate test stats
            agent_rewards[aid] += rewards_dict[aid]
            agent_batteries[aid].append(agent_info["battery"])
            if agent_info["sample_rejected"]:
                agent_rejections[aid] += 1
            if agent_info["is_high_entropy"] and agent_info["action_executed"] == shared_config.ACTION_SLEEP:
                agent_missed[aid] += 1

        terminated = all(terminations_dict.values())

    print("\n--- MULTI-AGENT RANDOM POLICY TEST RESULTS (288 Timesteps) ---")
    for aid in env.agent_ids:
        avg_bat = np.mean(agent_batteries[aid])
        print(f"[{aid}] Total Reward: {agent_rewards[aid]:7.2f} | Avg Battery: {avg_bat:.4f} | "
              f"Rejections: {agent_rejections[aid]:2d} | Missed Events: {agent_missed[aid]:2d}")

    print("\n--- SIMULTANEOUS SAMPLING OVERLAP FREQUENCY ---")
    print(f"2 Agents Sampled Together: {simultaneous_overlaps[2]} timesteps")
    print(f"3 Agents Sampled Together: {simultaneous_overlaps[3]} timesteps")
    print(f"4 Agents Sampled Together: {simultaneous_overlaps[4]} timesteps")

    print("\n" + "=" * 75)
    print("SANITY & PARTIAL OBSERVABILITY TEST PASSED PERFECTLY!")
    print("All agent observations strictly length 3. No neighbor state leak detected.")
    print("=" * 75)
