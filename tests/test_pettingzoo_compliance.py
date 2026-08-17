"""
PettingZoo ParallelEnv API Compliance Tests
MARL Adaptive IoT Sampling Project
"""

import pytest
from pettingzoo.test import parallel_api_test
import shared_config
from environment.pettingzoo_env import IoTSensorEnv

def test_pettingzoo_parallel_api():
    """Runs the official PettingZoo parallel_api_test suite."""
    env = IoTSensorEnv(scenario="stable", seed=42)
    # Run official PettingZoo test harness
    parallel_api_test(env, num_cycles=50)

def test_pettingzoo_reset_and_step_shapes():
    """Verify observations, rewards, terminations, and truncations dictionary structure."""
    env = IoTSensorEnv(scenario="volatile", seed=101)
    obs, infos = env.reset(seed=101)

    assert set(obs.keys()) == set(env.possible_agents)
    assert set(infos.keys()) == set(env.possible_agents)

    actions = {aid: shared_config.ACTION_SLEEP for aid in env.possible_agents}
    next_obs, rewards, terms, truncs, infos = env.step(actions)

    assert len(next_obs) == shared_config.NUM_AGENTS
    assert len(rewards) == shared_config.NUM_AGENTS
    assert len(terms) == shared_config.NUM_AGENTS
    assert len(truncs) == shared_config.NUM_AGENTS
    assert len(infos) == shared_config.NUM_AGENTS

    # State vector check
    state_vector = env.state()
    assert state_vector.shape == (shared_config.GLOBAL_STATE_DIM,)
