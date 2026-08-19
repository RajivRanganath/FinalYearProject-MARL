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
    """Verify causal multi-agent observations conform to the shared contract."""
    env = MultiAgentSensorEnv(scenario="stable", seed=42)
    obs_dict, info_dict = env.reset(seed=42)

    assert len(obs_dict) == shared_config.NUM_AGENTS
    for agent_id, obs in obs_dict.items():
        assert obs.shape == (shared_config.ENV_OBS_DIM,)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        assert 0.0 <= obs[shared_config.STATE_INDEX_RESIDUAL_ENERGY] <= 1.0
        assert 0.0 <= obs[shared_config.STATE_INDEX_EVENT_PROXY] <= 1.0
        assert 0.0 <= obs[shared_config.STATE_INDEX_AOI] <= 1.0
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


def test_sample_mask_accounts_for_all_same_step_energy_costs():
    env = SingleAgentSensorEnv(scenario="stable", seed=123)
    env.timestep = 0  # night: no harvest can mask an energy-causality defect
    full_sample_cost = (
        env.energy_model.sample_energy_cost
        + env.energy_model.sleep_energy_cost
        + env.energy_model.proxy_monitor_energy_cost
    )
    env.battery = full_sample_cost - 1e-8
    assert env.get_action_mask().tolist() == [1, 0]
    _, _, _, _, rejected = env.step(shared_config.ACTION_SAMPLE)
    assert rejected["sample_executed"] is False
    assert rejected["sample_rejected"] is True

    env.battery = full_sample_cost
    assert env.get_action_mask().tolist() == [1, 1]
    _, _, _, _, executed = env.step(shared_config.ACTION_SAMPLE)
    assert executed["sample_executed"] is True
    assert executed["consumed_energy"] <= full_sample_cost

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

def test_entropy_is_not_available_before_sampling():
    """Latent full-sensor entropy must not leak into the decision observation."""
    env = SingleAgentSensorEnv(scenario="stable", seed=12)
    env.reset(seed=12)
    env.event_proxy = 0.25
    env.current_entropy = 0.90
    before = env._local_obs().copy()
    assert before[1] == pytest.approx(0.25)

    _, _, _, _, sleep_info = env.step(shared_config.ACTION_SLEEP)
    assert sleep_info["measured_entropy"] is None

    env.battery = 0.90
    env.current_entropy = 0.90
    env.event_proxy = 0.25
    _, _, _, _, sample_info = env.step(shared_config.ACTION_SAMPLE)
    assert sample_info["measured_entropy"] == pytest.approx(0.90)

def test_scientific_regime_channel_contract():
    """Only the coordinated regime imposes the declared two-packet channel."""
    actions = {f"agent_{i}": shared_config.ACTION_SAMPLE for i in range(shared_config.NUM_AGENTS)}
    independent = MultiAgentSensorEnv(scenario="volatile", regime="independent", seed=9)
    independent.reset(seed=9)
    _, _, _, _, ind_info = independent.step(actions)
    assert sum(info["sample_delivered"] for info in ind_info.values()) == shared_config.NUM_AGENTS
    assert not any(info["channel_blocked"] for info in ind_info.values())

    coordinated = MultiAgentSensorEnv(scenario="volatile", regime="coordinated", seed=9)
    coordinated.reset(seed=9)
    _, _, _, _, coord_info = coordinated.step(actions)
    assert sum(info["sample_delivered"] for info in coord_info.values()) == 2
    assert sum(info["channel_blocked"] for info in coord_info.values()) == 2

def test_unseeded_reset_does_not_replay_identical_day():
    """Training reset(None) must continue RNG streams instead of reseeding to 42."""
    env = IoTSensorEnv(scenario="volatile", regime="independent", seed=42)
    first, _ = env.reset(seed=42)
    second, _ = env.reset()
    assert any(not np.array_equal(first[aid], second[aid]) for aid in env.possible_agents)

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


def test_ablation_removes_only_declared_signal_or_reward_term():
    """Core ablation switches must alter the intended training component."""
    neighbor = MultiAgentSensorEnv(
        scenario="volatile", regime="coordinated", ablation="no_neighbor_signal", seed=7
    )
    neighbor.reset(seed=7)
    actions = {aid: shared_config.ACTION_SLEEP for aid in neighbor.agent_ids}
    actions["agent_1"] = shared_config.ACTION_SAMPLE
    obs, _, _, _, _ = neighbor.step(actions)
    assert all(
        item[shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE] == pytest.approx(0.0)
        for item in obs.values()
    )

    no_aoi = MultiAgentSensorEnv(
        scenario="volatile", regime="coordinated", ablation="no_aoi", seed=7
    )
    no_aoi.reset(seed=7)
    for agent in no_aoi.agents.values():
        agent.aoi = 50
    _, _, _, _, infos = no_aoi.step({aid: shared_config.ACTION_SLEEP for aid in no_aoi.agent_ids})
    assert all(info["reward_components"]["aoi"] == pytest.approx(0.0) for info in infos.values())

    no_energy = MultiAgentSensorEnv(
        scenario="volatile", regime="coordinated", ablation="no_energy", seed=7
    )
    no_energy.reset(seed=7)
    _, _, _, _, infos = no_energy.step({aid: shared_config.ACTION_SAMPLE for aid in no_energy.agent_ids})
    assert all(info["reward_components"]["sample_energy"] == pytest.approx(0.0) for info in infos.values())

    no_redundancy = MultiAgentSensorEnv(
        scenario="volatile", regime="coordinated", ablation="no_redundancy", seed=7
    )
    no_redundancy.reset(seed=7)
    _, _, _, _, infos = no_redundancy.step(
        {aid: shared_config.ACTION_SAMPLE for aid in no_redundancy.agent_ids}
    )
    assert all(info["reward_components"]["redundancy"] == pytest.approx(0.0) for info in infos.values())


def test_no_coordination_constraint_removes_capacity_and_joint_terms():
    """The named macro-ablation disables the coordinated channel/objective."""
    env = MultiAgentSensorEnv(
        scenario="volatile", regime="coordinated",
        ablation="no_coordination_constraint", seed=11,
    )
    env.reset(seed=11)
    _, _, _, _, infos = env.step({aid: shared_config.ACTION_SAMPLE for aid in env.agent_ids})
    assert sum(info["sample_delivered"] for info in infos.values()) == shared_config.NUM_AGENTS
    assert not any(info["channel_blocked"] for info in infos.values())
    assert all(info["reward_components"]["redundancy"] == pytest.approx(0.0) for info in infos.values())
    assert all(info["reward_components"]["coverage"] == pytest.approx(0.0) for info in infos.values())
