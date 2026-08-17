"""
Single-Agent Sensor Environment (Physically Grounded & Mathematically Sound)
MARL Adaptive IoT Sampling Project

Key Features & Scientific Enhancements:
1. Strict Markov Temporal Alignment:
   Actions a_t are evaluated against state s_t = (battery_t, entropy_t, aoi_t), computing
   reward R(s_t, a_t) BEFORE transitioning to s_{t+1}.
2. Isolated Deterministic Random Stream:
   Uses dedicated numpy.random.RandomState per instance for true reproducibility across instances.
3. Physically Grounded Energy Model:
   Directly parameterized via EnergyModel (Joules, Watts, mAh, ms).
4. Autoregressive AR(1) Cloud Weather Dynamics:
   Eliminates white-noise flickering; models realistic overcast weather fronts.
5. Multi-Step Persistent Poisson/Markov Event Process:
   Generates realistic multi-timestep anomaly bursts with decay persistence.
6. First-Class Age of Information (AoI) State Tracking:
   Tracks AoI step-by-step; resets to 0 on successful sample transmissions.
7. Action Availability Masking:
   Generates action masks [1, 1] (when battery >= cost) and [1, 0] (when battery < cost).
"""

import sys
import math
import random
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from environment.energy_model import EnergyModel, PhysicalEnergyProfile

class SingleAgentSensorEnv:
    """
    Simulates an individual solar-harvesting IoT sensor node.
    """

    def __init__(
        self,
        scenario: str = "stable",
        high_entropy_threshold: float = shared_config.BASELINE_RULE_ENTROPY_THRESHOLD,
        seed: Optional[int] = None,
        energy_model: Optional[EnergyModel] = None
    ):
        self.scenario_name = scenario
        self.scenario_cfg = shared_config.SCENARIOS.get(scenario, shared_config.SCENARIOS["stable"])
        self.energy_model = energy_model or EnergyModel()
        self.high_entropy_threshold = high_entropy_threshold
        self.episode_length = shared_config.EPISODE_LENGTH_TIMESTEPS

        # Isolated RNG stream
        self.np_random = np.random.RandomState(seed if seed is not None else shared_config.SEED)
        self.py_random = random.Random(seed if seed is not None else shared_config.SEED)

        # Solar Day/Night Timesteps (288 steps = 24h)
        self.sunrise_step = int(self.episode_length * 0.25)
        self.sunset_step = int(self.episode_length * 0.75)

        # Weather AR(1) Parameters
        self.cloud_ar1 = self.scenario_cfg.get("cloud_correlation_ar1", 0.70)
        self.cloud_noise_std = self.scenario_cfg.get("cloud_noise_std", 0.015)
        self.cloud_factor = 1.0  # Clear sky baseline

        # Event Generator Parameters
        self.event_lambda = self.scenario_cfg.get("event_frequency_lambda", 0.03)
        self.event_decay = self.scenario_cfg.get("event_persistence_decay", 0.50)

        # Dynamic State Variables
        self.battery = 1.0
        self.current_entropy = 0.10
        self.aoi = 0
        self.timestep = 0

        # Diagnostics & Metrics
        self.total_samples_requested = 0
        self.total_samples_executed = 0
        self.total_rejections = 0
        self.total_missed_events = 0
        self.total_harvested_energy = 0.0
        self.total_consumed_energy = 0.0

        if seed is not None:
            self.seed(seed)

    def seed(self, seed: Optional[int] = None):
        """Sets isolated deterministic random seed."""
        if seed is None:
            seed = shared_config.SEED
        self.np_random = np.random.RandomState(seed)
        self.py_random = random.Random(seed)

    def get_action_mask(self) -> np.ndarray:
        """
        Returns binary action availability mask: [can_sleep, can_sample].
        Sample is unavailable if battery is below sample energy cost.
        """
        can_sample = 1 if self.battery >= self.energy_model.sample_energy_cost else 0
        return np.array([1, can_sample], dtype=np.int8)

    def _update_weather(self) -> float:
        """
        Updates cloud attenuation using a mean-reverting AR(1) stochastic process:
        c_t = alpha * c_{t-1} + (1 - alpha) * 1.0 + N(0, sigma^2)
        """
        noise = self.np_random.normal(0.0, self.cloud_noise_std)
        self.cloud_factor = (
            self.cloud_ar1 * self.cloud_factor
            + (1.0 - self.cloud_ar1) * 1.0
            + noise
        )
        self.cloud_factor = float(np.clip(self.cloud_factor, 0.05, 1.0))
        return self.cloud_factor

    def _calculate_solar_harvest(self) -> float:
        """Calculates solar energy harvested during current timestep."""
        t = self.timestep % self.episode_length
        if self.sunrise_step <= t <= self.sunset_step:
            progress = (t - self.sunrise_step) / (self.sunset_step - self.sunrise_step)
            cloud_factor = self._update_weather()
            harvest = self.energy_model.calculate_harvested_energy(progress, cloud_factor)
        else:
            harvest = 0.0
        return harvest

    def _update_event_entropy(self) -> float:
        """
        Updates data entropy signal with multi-step persistent Poisson event bursts.
        """
        # Baseline background environmental volatility
        baseline_noise = float(self.np_random.uniform(0.05, 0.18))

        # Check for new high-entropy anomaly event arrival (Poisson process)
        if self.np_random.rand() < self.event_lambda:
            # Trigger new peak burst
            new_spike = float(self.np_random.uniform(0.75, 1.00))
            self.current_entropy = max(self.current_entropy, new_spike)
        else:
            # Decay active event toward baseline
            self.current_entropy = max(baseline_noise, self.current_entropy * self.event_decay)

        return float(np.clip(self.current_entropy, 0.0, 1.0))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets environment to initial state.
        """
        if seed is not None:
            self.seed(seed)

        self.timestep = 0
        self.aoi = 0
        self.cloud_factor = 1.0

        # Initial battery distribution
        if self.scenario_cfg.get("initial_battery_low", False):
            self.battery = float(self.np_random.uniform(0.10, 0.25))
        else:
            self.battery = float(self.np_random.uniform(0.50, 0.90))

        self.current_entropy = float(self.np_random.uniform(0.05, 0.20))

        self.total_samples_requested = 0
        self.total_samples_executed = 0
        self.total_rejections = 0
        self.total_missed_events = 0
        self.total_harvested_energy = 0.0
        self.total_consumed_energy = 0.0

        obs = np.array([self.battery, self.current_entropy], dtype=np.float32)
        info = {
            "timestep": self.timestep,
            "battery": self.battery,
            "battery_joules": self.energy_model.normalized_to_joules(self.battery),
            "data_entropy": self.current_entropy,
            "aoi": self.aoi,
            "action_mask": self.get_action_mask()
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one environment step with strict Markov temporal alignment:
        1. Action a_t evaluated on state s_t = (battery_t, entropy_t, aoi_t).
        2. Reward R(s_t, a_t) computed.
        3. Physical state transitions to s_{t+1}.
        """
        shared_config.validate_contracts(action=action)
        self.timestep += 1

        # Snapshot current state s_t on which the agent made its decision
        current_battery = self.battery
        current_entropy = self.current_entropy
        current_aoi = self.aoi
        is_event = (current_entropy >= self.high_entropy_threshold)

        # 1. Action Feasibility & Energy Causality Check
        sample_requested = (action == shared_config.ACTION_SAMPLE)
        sample_executed = False
        sample_rejected = False
        rejection_penalty = 0.0

        if sample_requested:
            self.total_samples_requested += 1
            if current_battery >= self.energy_model.sample_energy_cost:
                sample_executed = True
                self.total_samples_executed += 1
                energy_deducted = self.energy_model.sample_energy_cost
            else:
                # Energy Causality: Reject sample; treat as sleep; do NOT allow energy debt
                sample_rejected = True
                self.total_rejections += 1
                energy_deducted = 0.0
                rejection_penalty = shared_config.REWARD_WEIGHTS["w_rejection"]
        else:
            energy_deducted = 0.0

        # 2. Canonical Reward Calculation on (s_t, a_t)
        w = shared_config.REWARD_WEIGHTS
        normalized_aoi = min(1.0, current_aoi / 50.0)

        if sample_executed:
            if is_event:
                # Captured critical high-entropy event
                reward = w["w_info"] - w["w_energy"]
            else:
                # Wasted sample on low-value data
                reward = -w["w_energy"]
        else:
            if is_event:
                # Slept through / failed to capture high-entropy event
                reward = -w["w_miss"]
                self.total_missed_events += 1
            else:
                # Disciplined sleep during low-value period (zero constant drip)
                reward = 0.0

        # Subtract linear AoI freshness penalty and rejection penalty
        reward -= (w["w_aoi"] * normalized_aoi + rejection_penalty)

        # 3. Physical State Transitions to s_{t+1}
        # Solar harvesting
        harvested = self._calculate_solar_harvest()
        self.total_harvested_energy += harvested

        # Battery update: E_{t+1} = E_t - E_action - E_sleep + E_harvest
        consumed = energy_deducted + self.energy_model.sleep_energy_cost
        self.total_consumed_energy += consumed
        new_battery = current_battery - consumed + harvested
        self.battery = float(np.clip(new_battery, 0.0, 1.0))

        # AoI update: Reset to 0 on successful sample; increment otherwise
        if sample_executed:
            self.aoi = 0
        else:
            self.aoi += 1

        # Entropy update for next timestep t+1
        next_entropy = self._update_event_entropy()

        # Termination
        terminated = (self.timestep >= self.episode_length)
        truncated = False

        next_obs = np.array([self.battery, next_entropy], dtype=np.float32)
        info = {
            "timestep": self.timestep,
            "action_requested": action,
            "action_executed": shared_config.ACTION_SAMPLE if sample_executed else shared_config.ACTION_SLEEP,
            "sample_rejected": sample_rejected,
            "sample_executed": sample_executed,
            "harvested_energy": harvested,
            "consumed_energy": consumed,
            "battery": self.battery,
            "battery_joules": self.energy_model.normalized_to_joules(self.battery),
            "data_entropy_at_decision": current_entropy,
            "data_entropy_next": next_entropy,
            "is_high_entropy": is_event,
            "aoi": self.aoi,
            "action_mask": self.get_action_mask()
        }

        return next_obs, float(reward), terminated, truncated, info
