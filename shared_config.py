"""
Shared Configuration and Contract Specification
MARL Adaptive IoT Sampling & TinyML Hardware Evaluation

This file is the single source of truth for:
- Agent count, episode duration, and timestep resolution
- State vector and neural network observation dimensions
- Action space definition
- Scenario definitions (Stable, Volatile, Unseen Stress)
- Baseline parameters
- Reward formulation weights
- Contract validation assertions

DO NOT hardcode divergent parameters in individual modules.
"""

from typing import Dict, Any
import numpy as np

# -----------------------------------------------------------------------------
# 1. Core Multi-Agent Environment Parameters
# -----------------------------------------------------------------------------
NUM_AGENTS: int = 4
TIMESTEP_DURATION_SECONDS: float = 300.0  # 5 minutes per timestep
EPISODE_LENGTH_TIMESTEPS: int = 288       # 288 timesteps * 5 min = 24 hours (1 diurnal cycle)

# -----------------------------------------------------------------------------
# 2. State & Model Input Vector Contract
# -----------------------------------------------------------------------------
# Environment Observation Vector (per agent): 3 elements, all float32 in [0.0, 1.0]
# 0: residual_energy (normalized battery charge)
# 1: data_entropy (local event volatility)
# 2: neighbor_sampling_rate (recent transmission rate of neighborhood)
ENV_OBS_DIM: int = 3
STATE_DIM: int = ENV_OBS_DIM  # Backward compatibility alias

STATE_INDEX_RESIDUAL_ENERGY: int = 0
STATE_INDEX_DATA_ENTROPY: int = 1
STATE_INDEX_NEIGHBOR_SAMPLING_RATE: int = 2

# Global State Vector (Centralized Training for CTDE):
# Concatenation of all 4 agents' observations = 4 * 3 = 12
GLOBAL_STATE_DIM: int = NUM_AGENTS * ENV_OBS_DIM

# Decentralized Neural Network Input:
# When using parameter sharing with one-hot agent ID: 3 (obs) + 4 (one-hot) = 7
MODEL_INPUT_DIM: int = ENV_OBS_DIM + NUM_AGENTS  # 7
DECENTRALIZED_INPUT_DIM: int = ENV_OBS_DIM        # 3 (without agent ID)

# Network Architecture Standard
HIDDEN_DIM: int = 128
TINYML_PRUNED_HIDDEN_DIM: int = 64

# -----------------------------------------------------------------------------
# 3. Action Space Contract
# -----------------------------------------------------------------------------
# Discrete action space with 2 values
ACTION_SLEEP: int = 0   # Sleep / Harvest / Retain Energy
ACTION_SAMPLE: int = 1  # Sense, Process & Transmit Reading

NUM_ACTIONS: int = 2

# -----------------------------------------------------------------------------
# 4. Canonical Reward Weights (Unified Objective)
# R_i = w_info * I_event * a_i - w_energy * a_i - w_aoi * norm_AoI - w_red * Overlap - w_miss * I_event * (1 - a_i)
# -----------------------------------------------------------------------------
REWARD_WEIGHTS: Dict[str, float] = {
    "w_info": 5.0,         # Value of successfully capturing a high-entropy event
    "w_energy": 0.05,      # Cost of energy expenditure during sampling
    "w_aoi": 0.01,         # Penalty per step for accumulated Age of Information staleness
    "w_redundancy": 0.10,  # Penalty per simultaneous neighbor co-sampling collision
    "w_miss": 3.0,         # Penalty for failing to sample during a critical event
    "w_rejection": 0.50,   # Penalty for attempting sample with insufficient battery (causality violation)
}

# -----------------------------------------------------------------------------
# 5. Baseline Parameters
# -----------------------------------------------------------------------------
# Fixed Interval Baseline: sample once every N timesteps (e.g. N=12 is once per hour)
BASELINE_FIXED_INTERVAL_N: int = 12

# Rule-based Battery Threshold: if battery < 20%, sleep; otherwise sample
BASELINE_RULE_BATTERY_THRESHOLD: float = 0.20

# Rule-based Entropy Threshold: if entropy > 0.60, sample; otherwise sleep
BASELINE_RULE_ENTROPY_THRESHOLD: float = 0.60

# -----------------------------------------------------------------------------
# 6. Scenario Configurations
# -----------------------------------------------------------------------------
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "stable": {
        "description": "Stable scenario: Predictable diurnal solar curve, low cloud variance, sparse isolated events.",
        "solar_variance": "low",
        "cloud_correlation_ar1": 0.70,
        "cloud_noise_std": 0.015,
        "event_frequency_lambda": 0.03,  # Poisson event arrival rate per step
        "event_persistence_decay": 0.50, # Rapid decay
        "entropy_volatility": "low"
    },
    "volatile": {
        "description": "Volatile scenario: High cloud attenuation with weather fronts, frequent multi-step event bursts.",
        "solar_variance": "high",
        "cloud_correlation_ar1": 0.88,  # Persistent cloud cover fronts
        "cloud_noise_std": 0.040,
        "event_frequency_lambda": 0.08,  # Higher event frequency
        "event_persistence_decay": 0.80, # Persistent multi-step bursts
        "entropy_volatility": "high"
    },
    "stress": {
        "description": "Unseen Stress scenario: Severe overcast storm fronts, low initial charge, dense correlated events.",
        "solar_variance": "severe",
        "cloud_correlation_ar1": 0.95,
        "cloud_noise_std": 0.070,
        "event_frequency_lambda": 0.12,
        "event_persistence_decay": 0.88,
        "initial_battery_low": True,
        "entropy_volatility": "extreme"
    }
}

# -----------------------------------------------------------------------------
# 7. Device, Random Seeds & Split Protocol
# -----------------------------------------------------------------------------
DEVICE: str = "cpu"
SEED: int = 42

# Train / Validation / Test Seed Partitioning
TRAIN_SEEDS = [101, 102, 103, 104, 105]
VAL_SEEDS = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210]
TEST_SEEDS = list(range(1001, 1031))  # 30 deterministic held-out test seeds

# -----------------------------------------------------------------------------
# 8. Contract Validation Assertion Helper
# -----------------------------------------------------------------------------
def validate_contracts(
    obs: np.ndarray = None,
    action: int = None,
    avail_actions: list = None,
    model_input: np.ndarray = None
) -> bool:
    """
    Validates runtime data against the shared contract.
    Raises AssertionError if any contract constraint is violated.
    """
    if obs is not None:
        assert isinstance(obs, np.ndarray), f"Obs must be numpy ndarray, got {type(obs)}"
        assert obs.shape == (ENV_OBS_DIM,), f"Obs shape must be ({ENV_OBS_DIM},), got {obs.shape}"
        assert np.all(obs >= -1e-5) and np.all(obs <= 1.0 + 1e-5), f"Obs values out of [0, 1] range: {obs}"

    if action is not None:
        assert action in (ACTION_SLEEP, ACTION_SAMPLE), f"Invalid action {action}. Must be 0 or 1."

    if avail_actions is not None:
        assert len(avail_actions) == NUM_ACTIONS, f"Avail actions length must be {NUM_ACTIONS}, got {len(avail_actions)}"
        assert avail_actions[0] == 1, "ACTION_SLEEP (action 0) must always be available"
        assert all(a in (0, 1) for a in avail_actions), f"Avail actions must be binary, got {avail_actions}"

    if model_input is not None:
        assert model_input.shape[-1] in (MODEL_INPUT_DIM, DECENTRALIZED_INPUT_DIM), \
            f"Model input dim must be {MODEL_INPUT_DIM} or {DECENTRALIZED_INPUT_DIM}, got {model_input.shape}"

    return True
