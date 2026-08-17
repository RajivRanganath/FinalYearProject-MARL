"""
Unit & Integration Tests: Single and Multi-Agent Environment
MARL Adaptive IoT Sampling Project
"""

import pytest
import numpy as np
import shared_config
from environment.single_agent_env import SingleAgentSensorEnv
from environment.multi_agent_env import MultiAgentSensorEnv
from environment.pettingzoo_env import IoTSensorEnv

def test_observation_shapes_and_bounds():
    """Verify single and multi-agent observations strictly conform to [0, 1] 3D contract."""
    env = MultiAgentSensorEnv(scenario="stable", seed=42)
    obs_dict, info_dict = env.reset(seed=42)

    assert len(obs_dict) == shared_config.NUM_AGENTS
    for agent_id, obs in obs_dict.items():
        assert obs.shape == (shared_config.ENV_OBS_DIM,)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        assert 0.0 <= obs[shared_config.STATE_INDEX_RESIDUAL_ENERGY] <= 1.0
        assert 0.0 <= obs[shared_config.STATE_INDEX_DATA_ENTROPY] <= 1.0
        assert 0.0 <= obs[shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE] <= 1.0

def test_battery_causality_and_non_negativity():
    """Verify that battery never drops below 0 and invalid samples are rejected."""
    env = SingleAgentSensorEnv(scenario="stable", seed=42)
    obs, info = env.reset(seed=42)

    # Force battery to near-zero
    env.battery = 0.02  # Less than sample cost (0.05)
    mask = env.get_action_mask()
    assert mask[1] == 0, "Action mask must disallow SAMPLE when battery < cost"

    # Attempt sample anyway
    next_obs, reward, term, trunc, info = env.step(shared_config.ACTION_SAMPLE)
    assert info["sample_rejected"] is True
    assert info["sample_executed"] is False
    assert env.battery >= 0.0, "Battery must never drop below 0"

def test_markov_temporal_reward_alignment():
    """
    Verify that reward R(s_t, a_t) corresponds to the entropy observed at decision time t,
    NOT the newly generated entropy at t+1.
    """
    env = SingleAgentSensorEnv(scenario="stable", seed=100)
    obs, info = env.reset(seed=100)

    # Manually set high-entropy state at decision time t
    env.battery = 0.90
    env.current_entropy = 0.85  # Above threshold (0.60)
    decision_entropy = env.current_entropy

    # Take SAMPLE action
    next_obs, reward, term, trunc, info = env.step(shared_config.ACTION_SAMPLE)

    assert info["data_entropy_at_decision"] == decision_entropy
    assert info["is_high_entropy"] is True
    assert info["sample_executed"] is True
    # Reward should be positive (capture reward: w_info - w_energy)
    assert reward > 0.0, "Sampling a known high-entropy spike must yield positive net reward"

def test_deterministic_reproducibility():
    """Verify that identical seeds produce identical state trajectories."""
    env1 = MultiAgentSensorEnv(scenario="volatile", seed=1234)
    obs1, _ = env1.reset(seed=1234)

    env2 = MultiAgentSensorEnv(scenario="volatile", seed=1234)
    obs2, _ = env2.reset(seed=1234)

    for aid in env1.agent_ids:
        np.testing.assert_allclose(obs1[aid], obs2[aid], err_msg="Reset observations must match exactly")

    actions = {aid: shared_config.ACTION_SAMPLE if i % 2 == 0 else shared_config.ACTION_SLEEP for i, aid in enumerate(env1.agent_ids)}

    for _ in range(20):
        next_obs1, r1, _, _, _ = env1.step(actions)
        next_obs2, r2, _, _, _ = env2.step(actions)

        for aid in env1.agent_ids:
            np.testing.assert_allclose(next_obs1[aid], next_obs2[aid], err_msg="Trajectory observations must match")
            assert r1[aid] == pytest.approx(r2[aid], abs=1e-6), "Rewards must match exactly"

def test_partial_observability_neighbor_rate():
    """Verify neighbor sampling rate accurately reflects neighbor transmissions without state leakage."""
    env = MultiAgentSensorEnv(scenario="stable", seed=42)
    env.reset(seed=42)

    # Agent 1 samples, other agents sleep
    actions = {
        "agent_0": shared_config.ACTION_SLEEP,
        "agent_1": shared_config.ACTION_SAMPLE,
        "agent_2": shared_config.ACTION_SLEEP,
        "agent_3": shared_config.ACTION_SLEEP,
    }
    obs, _, _, _, infos = env.step(actions)

    # Agent 0 is neighbor to Agent 1 and Agent 3
    # Agent 0's observed neighbor rate should increase
    assert obs["agent_0"][shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE] > 0.0
    # Neighbor rate must remain strictly within [0.0, 1.0]
    for aid in env.agent_ids:
        assert 0.0 <= obs[aid][shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE] <= 1.0
