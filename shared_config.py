"""
Shared Configuration for MARL Adaptive IoT Sampling Project

This file is the single source of truth for:
- Agent count
- Episode length
- State vector definition
- Action space definition
- Baseline parameters
- Scenario configurations

DO NOT hardcode these values in individual modules.
"""

# -----------------------------------------------------------------------------
# Core Environment Parameters
# -----------------------------------------------------------------------------
NUM_AGENTS = 4
EPISODE_LENGTH_TIMESTEPS = 288  # Represents one simulated day (~5 min intervals)

# -----------------------------------------------------------------------------
# State Vector Definition
# -----------------------------------------------------------------------------
# State vector per agent, in this exact order. All values float, normalized 0 to 1.
STATE_INDEX_RESIDUAL_ENERGY = 0
STATE_INDEX_DATA_ENTROPY = 1
STATE_INDEX_NEIGHBOR_SAMPLING_RATE = 2

STATE_DIM = 3

# -----------------------------------------------------------------------------
# Action Space Definition
# -----------------------------------------------------------------------------
# Discrete action space with 2 values
ACTION_SLEEP = 0   # Sleep or Wait
ACTION_SAMPLE = 1  # Sample Now

NUM_ACTIONS = 2

# -----------------------------------------------------------------------------
# Baseline Parameters
# -----------------------------------------------------------------------------
# Fixed Interval Baseline: sample every N timesteps regardless of state
BASELINE_FIXED_INTERVAL_N = 12  # e.g., sample once per hour

# Rule-based Baseline: if battery is below this threshold, sleep; otherwise sample
BASELINE_RULE_BATTERY_THRESHOLD = 0.20

# -----------------------------------------------------------------------------
# Scenario Configurations
# -----------------------------------------------------------------------------
SCENARIOS = {
    "stable": {
        "description": "Stable low volatility scenario. Predictable solar harvesting and consistent data entropy.",
        "solar_variance": "low",
        "entropy_volatility": "low"
    },
    "volatile": {
        "description": "Volatile high entropy spike scenario. Unpredictable weather and sudden data events.",
        "solar_variance": "high",
        "entropy_volatility": "high"
    }
}

# -----------------------------------------------------------------------------
# Training and Execution Parameters
# -----------------------------------------------------------------------------
# Explicitly default to CPU for consistency across OS and ease of deployment.
DEVICE = "cpu"

# Seed for reproducibility in all environments and splits.
SEED = 42

