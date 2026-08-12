"""
Single-Agent Sensor Environment (Phase 1 Foundation)

This module implements Phase 1 of Module A: a single IoT sensor node operating under solar 
energy harvesting constraints and data entropy volatility.

Key Features:
- Battery state initialized randomly and updated via a solar harvesting bell curve with cloud noise.
- Hard energy causality constraint: sampling when battery < energy cost is rejected as a no-op (Sleep).
- Data entropy signal with low baseline background noise and sparse, decaying high-entropy event spikes.
- Gymnasium-style (obs, reward, terminated, truncated, info) interface.

NOTE ON PHASE 1 STATE SPACE:
The observation vector returned in single-agent mode is 2-dimensional:
    [residual_energy, data_entropy]
The 3rd state element defined in shared_config.py (neighbor_sampling_rate) is exclusive to 
multi-agent coordination and will be added in Phase 2.
"""

import sys
import math
import random
import numpy as np
from pathlib import Path

# Add project root to path using pathlib for cross-platform compliance
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config

class SingleAgentSensorEnv:
    """
    Single IoT Sensor Node Environment simulating solar harvesting, battery dynamics,
    and volatile data entropy sampling decisions.
    """

    def __init__(
        self,
        sample_energy_cost: float = 0.05,
        max_harvest_rate: float = 0.08,
        cloud_noise_std: float = 0.015,
        spike_prob: float = 0.04,
        decay_rate: float = 0.6,
        high_entropy_threshold: float = 0.6,
        scenario: str = "stable",
        seed: int = None
    ):
        """
        Initialize single-agent environment parameters.

        Args:
            sample_energy_cost: Energy deducted when a sample is successfully taken (0.0 to 1.0 scale).
            max_harvest_rate: Peak energy harvested per timestep at solar midday under clear skies.
            cloud_noise_std: Standard deviation of Gaussian noise added to solar harvesting (cloud cover).
            spike_prob: Probability per timestep of a high-entropy data event spike occurring.
            decay_rate: Rate at which a data entropy spike decays back to baseline per timestep.
            high_entropy_threshold: Cutoff above which data entropy is deemed high/urgent.
            scenario: Scenario key ("stable" or "volatile") to adjust environmental variance.
            seed: Optional random seed for reproducible runs.
        """
        self.sample_energy_cost = sample_energy_cost
        self.max_harvest_rate = max_harvest_rate
        self.high_entropy_threshold = high_entropy_threshold
        self.decay_rate = decay_rate
        
        # Adjust variance according to shared_config scenario specs
        if scenario == "volatile" or (scenario in shared_config.SCENARIOS and shared_config.SCENARIOS[scenario]["entropy_volatility"] == "high"):
            self.cloud_noise_std = cloud_noise_std * 2.0
            self.spike_prob = spike_prob * 2.0
        else:
            self.cloud_noise_std = cloud_noise_std
            self.spike_prob = spike_prob

        self.scenario = scenario
        self.episode_length = shared_config.EPISODE_LENGTH_TIMESTEPS

        # Solar day cycle timing: 288 timesteps per 24 hours
        # Daylight is active between timestep 72 (06:00) and 216 (18:00), peaking at 144 (12:00)
        self.sunrise_step = int(self.episode_length * 0.25)  # Timestep 72
        self.sunset_step = int(self.episode_length * 0.75)   # Timestep 216
        self.midday_step = int(self.episode_length * 0.50)   # Timestep 144

        # Dynamic state variables
        self.battery = 1.0
        self.current_entropy = 0.1
        self.timestep = 0

        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int = None):
        """Set random seed for reproducibility."""
        if seed is None:
            seed = shared_config.SEED
        random.seed(seed)
        np.random.seed(seed)

    def _get_solar_harvest(self) -> float:
        """
        Calculates solar energy harvested at current timestep using a bell curve (clipped sine)
        with additive Gaussian noise to model stochastic cloud cover.
        """
        t = self.timestep % self.episode_length
        if self.sunrise_step <= t <= self.sunset_step:
            # Solar angle ranges from 0 to pi across daylight hours
            daylight_progress = (t - self.sunrise_step) / (self.sunset_step - self.sunrise_step)
            ideal_solar = math.sin(math.pi * daylight_progress) * self.max_harvest_rate
            # Additive cloud noise
            cloud_noise = np.random.normal(0.0, self.cloud_noise_std)
            harvest = max(0.0, ideal_solar + cloud_noise)
        else:
            # Night time: zero solar harvesting
            harvest = 0.0
        return float(harvest)

    def _update_data_entropy(self) -> float:
        """
        Generates data entropy signal: stays low (background noise 0.05-0.2) with sparse,
        decaying high-entropy event spikes (0.7-1.0).
        """
        # Baseline background noise representing low-activity monitoring
        baseline = float(np.random.uniform(0.05, 0.18))

        # Check if a new event spike occurs
        if np.random.rand() < self.spike_prob:
            # Trigger sharp spike
            self.current_entropy = float(np.random.uniform(0.75, 1.0))
        else:
            # Decay active spike or maintain baseline
            self.current_entropy = max(baseline, self.current_entropy * self.decay_rate)

        return float(np.clip(self.current_entropy, 0.0, 1.0))

    def reset(self, seed: int = None, options: dict = None):
        """
        Resets environment state for a new episode.

        Returns:
            observation (np.ndarray): Array [residual_energy, data_entropy]
            info (dict): Initial environment metadata
        """
        if seed is not None:
            self.seed(seed)

        self.timestep = 0
        # Initialize battery to random value between 0.5 and 1.0
        self.battery = float(np.random.uniform(0.5, 1.0))
        self.current_entropy = float(np.random.uniform(0.05, 0.20))

        # Phase 1 observation: 2 elements [residual_energy, data_entropy]
        obs = np.array([self.battery, self.current_entropy], dtype=np.float32)
        info = {
            "timestep": self.timestep,
            "battery": self.battery,
            "data_entropy": self.current_entropy
        }
        return obs, info

    def step(self, action: int):
        """
        Executes one environment step.

        Args:
            action (int): 0 (ACTION_SLEEP) or 1 (ACTION_SAMPLE)

        Returns:
            observation (np.ndarray): Array [residual_energy, data_entropy]
            reward (float): Reward scalar
            terminated (bool): True if episode reached max timesteps
            truncated (bool): Always False in standard execution
            info (dict): Diagnostic metadata including rejection status
        """
        assert action in (shared_config.ACTION_SLEEP, shared_config.ACTION_SAMPLE), \
            f"Invalid action {action}. Must be ACTION_SLEEP (0) or ACTION_SAMPLE (1)."

        self.timestep += 1

        # 1. Harvesting step
        harvested = self._get_solar_harvest()
        self.battery = min(1.0, self.battery + harvested)

        # 2. Hard Energy Causality Constraint Check
        sample_rejected = False
        action_executed = action

        if action == shared_config.ACTION_SAMPLE:
            if self.battery < self.sample_energy_cost:
                # Reject sample: treat as Sleep (no-op), do NOT let battery drop below 0
                sample_rejected = True
                action_executed = shared_config.ACTION_SLEEP
            else:
                # Execute sample & deduct energy
                self.battery -= self.sample_energy_cost
                action_executed = shared_config.ACTION_SAMPLE

        # Guarantee strict battery bounds [0.0, 1.0]
        self.battery = float(np.clip(self.battery, 0.0, 1.0))

        # 3. Update data entropy signal
        entropy_val = self._update_data_entropy()
        is_high_entropy = (entropy_val >= self.high_entropy_threshold)

        # 4. Calculate Reward Structure
        # Tradeoff: Positive reward for capturing high entropy; penalty for wasted sample or missed event.
        if action_executed == shared_config.ACTION_SAMPLE and not sample_rejected:
            if is_high_entropy:
                reward = 1.0   # Positive reward: captured important event
            else:
                reward = -0.2  # Negative reward: wasted energy on low-value data
        else:
            # Action executed was Sleep (or action requested was Sample but REJECTED due to low battery)
            if is_high_entropy:
                reward = -0.8  # Negative penalty: slept through a high-entropy event
            else:
                reward = 0.05  # Small positive reward: correct restraint during boring period

        # 5. Construct Phase 1 observation & Gymnasium-style return
        obs = np.array([self.battery, entropy_val], dtype=np.float32)
        terminated = (self.timestep >= self.episode_length)
        truncated = False

        info = {
            "timestep": self.timestep,
            "action_requested": action,
            "action_executed": action_executed,
            "sample_rejected": sample_rejected,
            "harvested_energy": harvested,
            "battery": self.battery,
            "data_entropy": entropy_val,
            "is_high_entropy": is_high_entropy
        }

        return obs, reward, terminated, truncated, info


# -----------------------------------------------------------------------------
# Test Harness & Sanity Verification
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PHASE 1 SINGLE-AGENT ENVIRONMENT SANITY TEST")
    print("=" * 70)

    env = SingleAgentSensorEnv(scenario="stable", seed=shared_config.SEED)

    # -------------------------------------------------------------------------
    # Test 1: Random Policy Run
    # -------------------------------------------------------------------------
    obs, info = env.reset(seed=shared_config.SEED)
    total_reward_random = 0.0
    battery_levels_random = []
    rejected_count_random = 0
    missed_events_random = 0

    terminated = False
    while not terminated:
        # Random action choice: 0 (Sleep) or 1 (Sample)
        action = np.random.choice([shared_config.ACTION_SLEEP, shared_config.ACTION_SAMPLE])
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward_random += reward
        battery_levels_random.append(info["battery"])
        if info["sample_rejected"]:
            rejected_count_random += 1
        if info["is_high_entropy"] and info["action_executed"] == shared_config.ACTION_SLEEP:
            missed_events_random += 1

        # Assert battery constraints hold at all times
        assert 0.0 <= info["battery"] <= 1.0, f"Battery out of bounds: {info['battery']}"

    avg_battery_random = np.mean(battery_levels_random)

    print("\n--- RANDOM POLICY RESULTS ---")
    print(f"Total Reward           : {total_reward_random:.2f}")
    print(f"Average Battery Level  : {avg_battery_random:.4f}")
    print(f"Rejected Samples       : {rejected_count_random}")
    print(f"Missed High-Entropy Ev : {missed_events_random}")

    # -------------------------------------------------------------------------
    # Test 2: Fixed Interval Policy Run (Sample every N steps, N=BASELINE_FIXED_INTERVAL_N)
    # -------------------------------------------------------------------------
    obs, info = env.reset(seed=shared_config.SEED + 1)
    total_reward_fixed = 0.0
    battery_levels_fixed = []
    rejected_count_fixed = 0
    missed_events_fixed = 0
    N = shared_config.BASELINE_FIXED_INTERVAL_N

    step_idx = 0
    terminated = False
    while not terminated:
        step_idx += 1
        action = shared_config.ACTION_SAMPLE if (step_idx % N == 0) else shared_config.ACTION_SLEEP
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward_fixed += reward
        battery_levels_fixed.append(info["battery"])
        if info["sample_rejected"]:
            rejected_count_fixed += 1
        if info["is_high_entropy"] and info["action_executed"] == shared_config.ACTION_SLEEP:
            missed_events_fixed += 1

        # Assert battery constraints hold at all times
        assert 0.0 <= info["battery"] <= 1.0, f"Battery out of bounds: {info['battery']}"

    avg_battery_fixed = np.mean(battery_levels_fixed)

    print("\n--- FIXED-INTERVAL POLICY RESULTS (N = {}) ---".format(N))
    print(f"Total Reward           : {total_reward_fixed:.2f}")
    print(f"Average Battery Level  : {avg_battery_fixed:.4f}")
    print(f"Rejected Samples       : {rejected_count_fixed}")
    print(f"Missed High-Entropy Ev : {missed_events_fixed}")

    print("\n" + "=" * 70)
    print("SANITY TEST PASSED: All battery bounds [0.0, 1.0] verified.")
    print("=" * 70)
