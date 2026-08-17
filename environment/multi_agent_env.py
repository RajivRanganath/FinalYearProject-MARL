"""
Multi-Agent Dec-POMDP Sensor Environment (4 Agents with Spatial Topology)
MARL Adaptive IoT Sampling Project

Key Features & Scientific Enhancements:
1. Dec-POMDP Partial Observability:
   Each agent observes strictly [residual_energy, data_entropy, neighbor_sampling_rate].
   Raw neighbor battery levels and neighbor entropy are strictly unobservable.
2. Spatial Topology & Localized Neighborhoods:
   4 nodes arranged in a spatial network with defined adjacency (Ring / 2x2 Grid).
   Neighbor sampling rate is calculated across immediate neighbors over a rolling window.
3. Coordinated Canonical Reward Function:
   Incentivizes capturing high-entropy events, penalizes redundant overlapping samples
   between neighbors, penalizes data staleness (AoI), and enforces battery causality.
4. Action Availability Masking:
   Generates per-agent action availability masks [1, can_sample].
"""

import sys
import random
import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from environment.single_agent_env import SingleAgentSensorEnv
from environment.energy_model import EnergyModel

class MultiAgentSensorEnv:
    """
    Decentralized Partially Observable Markov Decision Process (Dec-POMDP)
    environment simulating 4 cooperating IoT sensor nodes.
    """

    def __init__(
        self,
        num_agents: int = shared_config.NUM_AGENTS,
        scenario: str = "stable",
        seed: Optional[int] = None,
        rolling_window_size: int = 12,
        topology: str = "ring"
    ):
        self.num_agents = num_agents
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self.scenario = scenario
        self.rolling_window_size = rolling_window_size
        self.energy_model = EnergyModel()
        self.timestep = 0

        # Spatial Sensor Topology Adjacency Matrix
        # Ring topology: Agent 0 connected to 1 & 3; Agent 1 to 0 & 2; Agent 2 to 1 & 3; Agent 3 to 2 & 0
        if topology == "ring":
            self.adjacency: Dict[str, list] = {
                "agent_0": ["agent_1", "agent_3"],
                "agent_1": ["agent_0", "agent_2"],
                "agent_2": ["agent_1", "agent_3"],
                "agent_3": ["agent_2", "agent_0"]
            }
        else:
            # Fully connected neighborhood fallback
            self.adjacency = {
                aid: [other for other in self.agent_ids if other != aid]
                for aid in self.agent_ids
            }

        # Initialize underlying physical simulation per agent
        self.agents: Dict[str, SingleAgentSensorEnv] = {
            agent_id: SingleAgentSensorEnv(
                scenario=scenario,
                energy_model=self.energy_model
            )
            for agent_id in self.agent_ids
        }

        # Rolling history of executed samples per agent for neighbor rate tracking
        self.action_history: Dict[str, deque] = {
            agent_id: deque(maxlen=self.rolling_window_size)
            for agent_id in self.agent_ids
        }

        if seed is not None:
            self.seed(seed)

    def seed(self, seed: Optional[int] = None):
        """Seeds all agents deterministically."""
        if seed is None:
            seed = shared_config.SEED
        random.seed(seed)
        np.random.seed(seed)
        for i, agent_id in enumerate(self.agent_ids):
            self.agents[agent_id].seed(seed + i * 100)

    def get_avail_actions(self) -> Dict[str, np.ndarray]:
        """Returns action availability masks for all agents."""
        return {
            agent_id: self.agents[agent_id].get_action_mask()
            for agent_id in self.agent_ids
        }

    def _get_neighbor_sampling_rate(self, agent_id: str) -> float:
        """
        Calculates recent sampling frequency of adjacent spatial neighbors.
        Strictly preserves partial observability: does NOT expose neighbor battery or entropy.
        """
        neighbors = self.adjacency.get(agent_id, [])
        if not neighbors:
            return 0.0

        total_neighbor_samples = 0
        total_slots = 0
        for n_id in neighbors:
            hist = self.action_history[n_id]
            total_neighbor_samples += sum(hist)
            total_slots += len(hist)

        if total_slots == 0:
            return 0.0

        return float(np.clip(total_neighbor_samples / total_slots, 0.0, 1.0))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Resets multi-agent environment and returns initial 3D observations.
        """
        if seed is not None:
            self.seed(seed)

        self.timestep = 0
        for agent_id in self.agent_ids:
            self.action_history[agent_id].clear()

        observations = {}
        infos = {}

        for i, agent_id in enumerate(self.agent_ids):
            sub_seed = (seed + i * 100) if seed is not None else None
            sa_obs, sa_info = self.agents[agent_id].reset(seed=sub_seed)

            neighbor_rate = 0.0
            obs_3d = np.array([sa_obs[0], sa_obs[1], neighbor_rate], dtype=np.float32)
            shared_config.validate_contracts(obs=obs_3d)

            observations[agent_id] = obs_3d
            sa_info["neighbor_sampling_rate"] = neighbor_rate
            sa_info["action_mask"] = self.agents[agent_id].get_action_mask()
            infos[agent_id] = sa_info

        return observations, infos

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Executes one joint step across all 4 agents.
        """
        self.timestep += 1

        # Step 1: Execute single-agent physics and compute independent rewards
        sa_results = {}
        executed_samples: Dict[str, int] = {}

        for agent_id in self.agent_ids:
            action = actions[agent_id]
            next_obs_2d, ind_reward, term, trunc, info = self.agents[agent_id].step(action)
            is_sample_executed = 1 if info["sample_executed"] else 0

            self.action_history[agent_id].append(is_sample_executed)
            executed_samples[agent_id] = is_sample_executed

            sa_results[agent_id] = {
                "next_obs_2d": next_obs_2d,
                "ind_reward": ind_reward,
                "term": term,
                "trunc": trunc,
                "info": info,
                "sample_executed": is_sample_executed
            }

        # Step 2: Compute joint spatial coordination & redundancy penalties
        w_red = shared_config.REWARD_WEIGHTS["w_redundancy"]
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent_id in self.agent_ids:
            res = sa_results[agent_id]
            neighbors = self.adjacency.get(agent_id, [])

            # Count how many immediate spatial neighbors also sampled simultaneously
            co_samplers = sum(executed_samples[n_id] for n_id in neighbors)

            if res["sample_executed"] == 1 and co_samplers > 0:
                redundancy_penalty = w_red * co_samplers
            else:
                redundancy_penalty = 0.0

            total_reward = res["ind_reward"] - redundancy_penalty
            neighbor_rate = self._get_neighbor_sampling_rate(agent_id)

            obs_3d = np.array([res["next_obs_2d"][0], res["next_obs_2d"][1], neighbor_rate], dtype=np.float32)
            shared_config.validate_contracts(obs=obs_3d)

            observations[agent_id] = obs_3d
            rewards[agent_id] = float(total_reward)
            terminations[agent_id] = res["term"]
            truncations[agent_id] = res["trunc"]

            info = res["info"]
            info["redundancy_penalty"] = redundancy_penalty
            info["neighbor_co_samplers"] = co_samplers if res["sample_executed"] == 1 else 0
            info["neighbor_sampling_rate"] = neighbor_rate
            info["action_mask"] = self.agents[agent_id].get_action_mask()
            infos[agent_id] = info

        return observations, rewards, terminations, truncations, infos
