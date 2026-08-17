"""Multi-agent sensor network with independent and coordinated regimes."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from environment.energy_model import EnergyModel
from environment.single_agent_env import SingleAgentSensorEnv


class MultiAgentSensorEnv:
    """A Dec-POMDP with explicit, configurable sources of agent coupling."""

    def __init__(
        self,
        num_agents: int = shared_config.NUM_AGENTS,
        scenario: str = "stable",
        regime: str = "independent",
        seed: Optional[int] = None,
        rolling_window_size: int = 12,
        topology: str = "ring",
        ablation: str = "full",
    ):
        if regime not in shared_config.REGIMES:
            raise ValueError(f"Unknown regime {regime!r}")
        self.num_agents = num_agents
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self.scenario = scenario
        self.regime = regime
        self.regime_cfg = dict(shared_config.REGIMES[regime])
        self.ablation = ablation
        self.rolling_window_size = rolling_window_size
        self.energy_model = EnergyModel()
        self.timestep = 0
        self.np_random = np.random.RandomState(shared_config.SEED)

        if topology == "ring":
            self.adjacency = {
                aid: [self.agent_ids[(i - 1) % num_agents], self.agent_ids[(i + 1) % num_agents]]
                for i, aid in enumerate(self.agent_ids)
            }
        else:
            self.adjacency = {
                aid: [other for other in self.agent_ids if other != aid]
                for aid in self.agent_ids
            }

        reward_overrides: Dict[str, float] = {}
        if ablation == "no_aoi":
            reward_overrides["w_aoi"] = 0.0
        if ablation == "no_energy":
            reward_overrides["w_energy"] = 0.0

        external_events = regime == "coordinated"
        self.agents: Dict[str, SingleAgentSensorEnv] = {
            aid: SingleAgentSensorEnv(
                scenario=scenario,
                energy_model=self.energy_model,
                external_event_control=external_events,
                reward_weights=reward_overrides,
            )
            for aid in self.agent_ids
        }
        self.action_history = {
            aid: deque(maxlen=rolling_window_size) for aid in self.agent_ids
        }
        self.regional_event_strength = 0.0
        if seed is not None:
            self.seed(seed)

    @property
    def coordination_enabled(self) -> bool:
        return self.regime == "coordinated" and self.ablation != "no_coordination_constraint"

    @property
    def channel_capacity(self) -> int:
        if not self.coordination_enabled:
            return self.num_agents
        return int(self.regime_cfg["channel_capacity"])

    def seed(self, seed: Optional[int] = None) -> None:
        seed = shared_config.SEED if seed is None else seed
        self.np_random = np.random.RandomState(seed)
        for i, aid in enumerate(self.agent_ids):
            self.agents[aid].seed(seed + i * 100)

    def get_avail_actions(self) -> Dict[str, np.ndarray]:
        return {aid: self.agents[aid].get_action_mask() for aid in self.agent_ids}

    def _get_neighbor_sampling_rate(self, agent_id: str) -> float:
        if self.ablation == "no_neighbor_signal":
            return 0.0
        neighbors = self.adjacency.get(agent_id, [])
        total = sum(sum(self.action_history[n]) for n in neighbors)
        slots = sum(len(self.action_history[n]) for n in neighbors)
        return float(total / slots) if slots else 0.0

    def _compose_obs(self, agent_id: str) -> np.ndarray:
        local = self.agents[agent_id]._local_obs()
        obs = np.array([
            local[0],
            local[1],
            local[2],
            self._get_neighbor_sampling_rate(agent_id),
            local[3],
        ], dtype=np.float32)
        shared_config.validate_contracts(obs=obs)
        return obs

    def _advance_coordinated_events(self) -> None:
        """Generate a persistent regional phenomenon plus weaker local events."""
        persistence = float(self.regime_cfg.get("regional_event_persistence", 0.85))
        arrival = float(self.regime_cfg.get("regional_event_probability", 0.06))
        if self.regional_event_strength >= shared_config.BASELINE_RULE_ENTROPY_THRESHOLD:
            if self.np_random.rand() < persistence:
                self.regional_event_strength = max(0.62, self.regional_event_strength * 0.92)
            else:
                self.regional_event_strength = 0.0
        elif self.np_random.rand() < arrival:
            self.regional_event_strength = float(self.np_random.uniform(0.78, 1.0))

        correlation = float(self.regime_cfg.get("spatial_event_correlation", 0.75))
        base_local_p = shared_config.SCENARIOS[self.scenario]["event_frequency_lambda"]
        for aid in self.agent_ids:
            baseline = float(self.np_random.uniform(0.05, 0.18))
            # Correlation controls how often a node shares the regional event,
            # not the event amplitude. Scaling the amplitude by correlation
            # previously pushed genuine regional events below the threshold.
            regional = self.regional_event_strength if (
                self.regional_event_strength > 0.0 and self.np_random.rand() < correlation
            ) else 0.0
            local_event = float(self.np_random.uniform(0.72, 1.0)) if (
                self.np_random.rand() < base_local_p * (1.0 - correlation)
            ) else 0.0
            entropy = max(baseline, regional + self.np_random.normal(0.0, 0.06), local_event)
            self.agents[aid].set_external_event_entropy(float(np.clip(entropy, 0.0, 1.0)))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        self.timestep = 0
        self.regional_event_strength = 0.0
        for aid in self.agent_ids:
            self.action_history[aid].clear()
            sub_seed = (seed + int(aid.split("_")[1]) * 100) if seed is not None else None
            self.agents[aid].reset(seed=sub_seed)
        if self.regime == "coordinated":
            self._advance_coordinated_events()

        observations = {aid: self._compose_obs(aid) for aid in self.agent_ids}
        infos = {
            aid: {
                "timestep": 0,
                "battery": self.agents[aid].battery,
                "event_proxy": self.agents[aid].event_proxy,
                "aoi": self.agents[aid].aoi,
                "harvest_forecast": self.agents[aid].get_harvest_forecast(),
                "measured_entropy": None,
                "neighbor_sampling_rate": 0.0,
                "action_mask": self.agents[aid].get_action_mask(),
                "regime": self.regime,
                "ablation": self.ablation,
                "causal_observation": True,
            }
            for aid in self.agent_ids
        }
        return observations, infos

    def _channel_winners(self, actions: Dict[str, int]) -> Tuple[set[str], set[str]]:
        feasible = [
            aid for aid in self.agent_ids
            if actions[aid] == shared_config.ACTION_SAMPLE
            and self.agents[aid].get_action_mask()[1] == 1
        ]
        if len(feasible) <= self.channel_capacity:
            return set(feasible), set()
        # Fair deterministic round-robin MAC priority. This is a network
        # mechanism, not a reward manipulation, and avoids permanent ID bias.
        start = self.timestep % self.num_agents
        order = self.agent_ids[start:] + self.agent_ids[:start]
        winners = {aid for aid in order if aid in feasible}
        winners = set(list(aid for aid in order if aid in winners)[: self.channel_capacity])
        return winners, set(feasible) - winners

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        if set(actions) != set(self.agent_ids):
            raise ValueError("A joint action is required for every agent")
        self.timestep += 1
        winners, channel_blocked = self._channel_winners(actions)

        results: Dict[str, Dict[str, Any]] = {}
        executed: Dict[str, int] = {}
        event_agents: set[str] = set()
        for aid in self.agent_ids:
            requested = int(actions[aid])
            if requested == shared_config.ACTION_SAMPLE and aid in channel_blocked:
                effective_action = shared_config.ACTION_SLEEP
            else:
                effective_action = requested
            _, reward, term, trunc, info = self.agents[aid].step(effective_action)
            if info["is_high_entropy"]:
                event_agents.add(aid)
            is_delivered = int(info["sample_executed"] and aid not in channel_blocked)
            executed[aid] = is_delivered
            self.action_history[aid].append(is_delivered)
            info["action_requested"] = requested
            info["sample_requested"] = requested == shared_config.ACTION_SAMPLE
            info["sample_delivered"] = bool(is_delivered)
            info["channel_blocked"] = aid in channel_blocked
            results[aid] = {"reward": reward, "term": term, "trunc": trunc, "info": info}

        delivered_agents = {aid for aid, value in executed.items() if value}
        covered_event_agents: set[str] = set()
        for sampler in delivered_agents:
            footprint = {sampler, *self.adjacency.get(sampler, [])}
            covered_event_agents.update(footprint & event_agents)
        coverage = len(covered_event_agents) / max(1, len(event_agents)) if event_agents else 1.0

        if self.regime == "coordinated":
            self._advance_coordinated_events()

        observations: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, Any] = {}
        w = shared_config.REWARD_WEIGHTS

        for aid in self.agent_ids:
            info = results[aid]["info"]
            neighbors = self.adjacency.get(aid, [])
            co_samplers = sum(executed[n] for n in neighbors) if executed[aid] else 0
            redundancy_penalty = 0.0
            if self.coordination_enabled and self.ablation != "no_redundancy" and co_samplers:
                redundancy_penalty = w["w_redundancy"] * co_samplers
            contention_penalty = w["w_channel_contention"] if aid in channel_blocked else 0.0
            coverage_bonus = (
                w["w_coverage"] * coverage / self.num_agents
                if self.coordination_enabled and event_agents
                else 0.0
            )

            components = dict(info["reward_components"])
            components.update({
                "redundancy": -redundancy_penalty,
                "channel_contention": -contention_penalty,
                "coverage": coverage_bonus,
            })
            total_reward = results[aid]["reward"] - redundancy_penalty - contention_penalty + coverage_bonus

            info.update({
                "reward_components": components,
                "redundancy_penalty": redundancy_penalty,
                "neighbor_co_samplers": co_samplers,
                "neighbor_sampling_rate": self._get_neighbor_sampling_rate(aid),
                "network_coverage": coverage,
                "regional_event_active": self.regional_event_strength >= shared_config.BASELINE_RULE_ENTROPY_THRESHOLD,
                "channel_capacity": self.channel_capacity,
                "action_mask": self.agents[aid].get_action_mask(),
                "regime": self.regime,
                "ablation": self.ablation,
            })
            observations[aid] = self._compose_obs(aid)
            rewards[aid] = float(total_reward)
            terminations[aid] = results[aid]["term"]
            truncations[aid] = results[aid]["trunc"]
            infos[aid] = info

        return observations, rewards, terminations, truncations, infos
