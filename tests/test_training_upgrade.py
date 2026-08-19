from __future__ import annotations

import numpy as np
import pytest

import shared_config
from training.env_wrapper import IoTSensorEnvWrapper
from training.policy_runtime import ObservationMaskPolicy
from training.train_all import _merge_training_summaries
from training.training_profiles import get_training_profile


def test_improved_profile_is_explicit_and_preserves_reward_contract():
    baseline = get_training_profile("baseline")
    improved = get_training_profile("improved")
    assert baseline.include_agent_id is True
    assert improved.include_agent_id is False
    assert improved.mask_neighbor_signal is True
    assert improved.updates_per_episode > baseline.updates_per_episode
    assert improved.checkpoint_count > baseline.checkpoint_count
    with pytest.raises(ValueError):
        get_training_profile("unknown")


def test_wrapper_masks_neighbor_signal_in_observation_and_state():
    raw = IoTSensorEnvWrapper(
        scenario="volatile", regime="coordinated", seed=71, mask_neighbor_signal=False
    )
    masked = IoTSensorEnvWrapper(
        scenario="volatile", regime="coordinated", seed=71, mask_neighbor_signal=True
    )
    actions = [shared_config.ACTION_SAMPLE, shared_config.ACTION_SLEEP] * 2
    raw.step(actions)
    masked.step(actions)

    raw_obs = raw.get_obs()
    masked_obs = masked.get_obs()
    index = shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE
    assert any(obs[index] > 0.0 for obs in raw_obs)
    assert all(obs[index] == 0.0 for obs in masked_obs)
    assert masked.get_state().shape == (shared_config.GLOBAL_STATE_DIM,)
    assert np.all(masked.get_state().reshape(shared_config.NUM_AGENTS, -1)[:, index] == 0.0)


def test_observation_mask_policy_masks_only_declared_feature():
    class Recorder:
        train_seed = 101

        def reset(self):
            self.last_q = {}

        def select_action(self, agent_id, obs, info):
            self.obs = obs.copy()
            return 0

    recorder = Recorder()
    policy = ObservationMaskPolicy(
        recorder, [shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE]
    )
    obs = np.arange(shared_config.ENV_OBS_DIM, dtype=np.float32)
    assert policy.select_action("agent_0", obs, {}) == 0
    assert recorder.obs[shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE] == 0.0
    np.testing.assert_array_equal(recorder.obs[:3], obs[:3])


def test_upgrade_manifest_merge_replaces_matching_seed_only():
    old = [
        {"profile": "improved", "ablation": "full", "regime": "coordinated", "algorithm": "qmix", "seed": 101, "value": "old"},
        {"profile": "improved", "ablation": "full", "regime": "coordinated", "algorithm": "qmix", "seed": 102, "value": "keep"},
    ]
    new = [
        {"profile": "improved", "ablation": "full", "regime": "coordinated", "algorithm": "qmix", "seed": 101, "value": "new"},
    ]
    merged = _merge_training_summaries(old, new)
    assert len(merged) == 2
    assert next(item for item in merged if item["seed"] == 101)["value"] == "new"
    assert next(item for item in merged if item["seed"] == 102)["value"] == "keep"
