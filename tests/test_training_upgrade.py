from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import shared_config
from training.env_wrapper import IoTSensorEnvWrapper
from training.policy_runtime import ObservationMaskPolicy
from training.train_all import (
    ROOT_DIR,
    _export_directory,
    _merge_training_summaries,
    _optimizer_learning_rates,
    _resume_checkpoint_root,
)
from training.training_profiles import get_training_profile
from training.refined_protocol import REFINED_FINAL_SEEDS, REFINED_SELECTION_SEEDS
from learners.q_learner import QLearner
from deployment.evaluate_training_v2 import (
    _holm_adjust,
    _selection_decision as _v2_selection_decision,
)
from deployment.split_lock import FinalSplitLock, SplitAlreadyConsumed, atomic_dataframe_csv
from deployment.evaluate_training_v3 import (
    PRIMARY_METRICS,
    V3_FINAL_SEEDS,
    V3_SELECTION_SEEDS,
    VotingEnsemblePolicy,
    _promotion_decision,
)


def test_improved_profile_is_explicit_and_preserves_reward_contract():
    baseline = get_training_profile("baseline")
    improved = get_training_profile("improved")
    improved_v2 = get_training_profile("improved_v2")
    extended = get_training_profile("extended")
    refined = get_training_profile("refined")
    assert baseline.include_agent_id is True
    assert improved.include_agent_id is False
    assert improved.mask_neighbor_signal is True
    assert improved.updates_per_episode > baseline.updates_per_episode
    assert improved.checkpoint_count > baseline.checkpoint_count
    assert improved_v2.include_agent_id is True
    assert improved_v2.mask_neighbor_signal is improved.mask_neighbor_signal
    assert improved_v2.recommended_t_max == improved.recommended_t_max
    assert improved_v2.learning_rate == improved.learning_rate
    assert improved_v2.gamma == improved.gamma
    assert improved_v2.batch_size == improved.batch_size
    assert improved_v2.buffer_size == improved.buffer_size
    assert improved_v2.updates_per_episode == improved.updates_per_episode
    assert extended.recommended_t_max == 2 * improved.recommended_t_max
    assert extended.include_agent_id == improved.include_agent_id
    assert extended.mask_neighbor_signal == improved.mask_neighbor_signal
    assert int(extended.recommended_t_max * extended.epsilon_anneal_fraction) == int(
        improved.recommended_t_max * improved.epsilon_anneal_fraction
    )
    assert refined.recommended_t_max == 540_000
    assert refined.learning_rate < extended.learning_rate
    assert refined.include_agent_id == extended.include_agent_id
    assert refined.mask_neighbor_signal == extended.mask_neighbor_signal
    assert refined.hidden_dim == extended.hidden_dim
    assert int(refined.recommended_t_max * refined.epsilon_anneal_fraction) == int(
        improved.recommended_t_max * improved.epsilon_anneal_fraction
    )
    with pytest.raises(ValueError):
        get_training_profile("unknown")


def test_v2_selection_and_test_splits_are_disjoint():
    splits = [
        set(shared_config.TRAIN_SEEDS),
        set(shared_config.VAL_SEEDS),
        set(shared_config.TEST_SEEDS),
        set(shared_config.UPGRADE_TEST_SEEDS),
        set(shared_config.V2_SELECTION_SEEDS),
        set(shared_config.V2_TEST_SEEDS),
        set(V3_SELECTION_SEEDS),
        set(V3_FINAL_SEEDS),
        set(REFINED_SELECTION_SEEDS),
        set(REFINED_FINAL_SEEDS),
    ]
    for index, first in enumerate(splits):
        for second in splits[index + 1 :]:
            assert first.isdisjoint(second)


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


def test_manifest_merge_keeps_distinct_scenarios():
    common = {
        "profile": "extended",
        "ablation": "full",
        "regime": "coordinated",
        "algorithm": "qmix",
        "seed": 101,
        "config_digest": "same",
    }
    merged = _merge_training_summaries(
        [{**common, "scenario": "stable"}],
        [{**common, "scenario": "volatile"}],
    )
    assert {item["scenario"] for item in merged} == {"stable", "volatile"}


def test_noncanonical_export_paths_cannot_overwrite_volatile_full_model():
    canonical = _export_directory("extended", "volatile", "coordinated", "full")
    stable = _export_directory("extended", "stable", "coordinated", "full")
    ablation = _export_directory("extended", "volatile", "coordinated", "no_energy")
    assert canonical == ROOT_DIR / "results/upgrade_models/extended/coordinated"
    assert stable != canonical
    assert ablation != canonical


def test_resume_resolves_exact_validation_selected_step():
    root, step = _resume_checkpoint_root("extended", "coordinated", "qmix", 103)
    assert step == 289_440
    assert (root / str(step) / "agent.th").exists()


def test_qlearner_resume_synchronises_target_mixer(tmp_path):
    class StateRecorder:
        def __init__(self):
            self.state = {"weight": torch.tensor([-1.0])}

        def load_state_dict(self, state):
            self.state = {name: value.clone() for name, value in state.items()}

        def state_dict(self):
            return self.state

    class MacRecorder:
        def load_models(self, path):
            self.loaded = path

    class OptimiserRecorder:
        def load_state_dict(self, state):
            self.state = state
            self.param_groups = [dict(group) for group in state["param_groups"]]

    torch.save({"weight": torch.tensor([3.0])}, tmp_path / "mixer.th")
    torch.save(
        {"state": {}, "param_groups": [{"lr": 3e-4}]},
        tmp_path / "opt.th",
    )
    learner = object.__new__(QLearner)
    learner.mac = MacRecorder()
    learner.target_mac = MacRecorder()
    learner.mixer = StateRecorder()
    learner.target_mixer = StateRecorder()
    learner.optimiser = OptimiserRecorder()
    learner.args = SimpleNamespace(lr=1e-4)
    learner.load_models(tmp_path)
    assert learner.mixer.state["weight"].item() == 3.0
    assert learner.target_mixer.state["weight"].item() == 3.0
    assert learner.optimiser.param_groups[0]["lr"] == 1e-4
    assert _optimizer_learning_rates(tmp_path) == [3e-4]


def test_final_split_lock_is_irreversible_and_outputs_are_atomic(tmp_path):
    lock = FinalSplitLock(tmp_path / "final", {"seeds": [9001]})
    lock.acquire()
    assert lock.started.exists()
    with pytest.raises(SplitAlreadyConsumed):
        FinalSplitLock(tmp_path / "final", {"seeds": [9001]}).acquire()
    atomic_dataframe_csv(pd.DataFrame([{"seed": 9001, "reward": 1.0}]), tmp_path / "final/raw.csv")
    assert (tmp_path / "final/raw.csv").exists()
    assert not (tmp_path / "final/raw.csv.tmp").exists()
    lock.mark_complete({"raw_rows": 1})
    assert lock.complete.exists()


def test_holm_adjustment_preserves_order_and_controls_family():
    adjusted = _holm_adjust([0.01, 0.04, 0.001])
    np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.003])
    assert np.all(adjusted >= np.array([0.01, 0.04, 0.001]))


def test_v2_selection_decision_freezes_scenario_and_regime():
    raw = pd.DataFrame([
        {"policy": policy, "training_seed": seed, "raw_episode_reward": reward}
        for seed in shared_config.TRAIN_SEEDS[:3]
        for policy, reward in (("Improved QMIX", 1.0), ("Extended QMIX", 2.0))
    ])
    decision = _v2_selection_decision(
        raw, {"model": "hash"}, {"analysis": "hash"}, "stable", "independent"
    )
    assert decision["scenario"] == "stable"
    assert decision["regime"] == "independent"
    assert decision["promote_extended"] is True


@pytest.mark.parametrize("minimum_votes,expected", [(2, 1), (3, 0)])
def test_voting_ensemble_uses_scale_independent_votes(minimum_votes, expected):
    class FakePolicy:
        def __init__(self, q_values):
            self.values = np.asarray(q_values, dtype=np.float32)
            self.calls = 0

        def reset(self):
            self.calls = 0

        def q_values(self, agent_id, obs):
            self.calls += 1
            return self.values

    policy = object.__new__(VotingEnsemblePolicy)
    policy.minimum_sample_votes = minimum_votes
    policy.policies = [
        FakePolicy([1000.0, -1000.0]),
        FakePolicy([0.0, 1.0]),
        FakePolicy([-2.0, 3.0]),
    ]
    action = policy.select_action(
        "agent_0", np.ones(shared_config.ENV_OBS_DIM), {"action_mask": [1, 1]}
    )
    assert action == expected
    assert [replica.calls for replica in policy.policies] == [1, 1, 1]


def test_v3_promotion_requires_joint_reward_and_engineering_gates():
    rows = []
    for candidate, reward_low, recall, energy, redundancy in [
        ("QMIX Majority Ensemble", 1.0, 0.0, 0.0, 0.0),
        ("QMIX Unanimous Ensemble", 2.0, -0.01, 1.0, 3.0),
    ]:
        values = {
            "raw_episode_reward": (4.0, reward_low),
            "event_recall": (recall, recall - 0.001),
            "total_energy_consumption": (energy, energy - 0.001),
            "mean_aoi": (0.0, -0.1),
            "redundant_sampling": (redundancy, redundancy - 0.1),
            "network_coverage": (0.0, -0.1),
        }
        for metric, (mean, low) in values.items():
            rows.append({
                "candidate": candidate,
                "comparator": "Extended QMIX Replica Mean",
                "metric": metric,
                "engineering_advantage_mean": mean,
                "ci95_low": low,
                "reward_p_holm": 0.01 if metric == "raw_episode_reward" else np.nan,
            })
    decision = _promotion_decision(pd.DataFrame(rows))
    assert decision["promoted_candidate"] == "QMIX Majority Ensemble"
    assert decision["candidate_evidence"]["QMIX Majority Ensemble"]["passed"] is True
    assert decision["candidate_evidence"]["QMIX Unanimous Ensemble"]["passed"] is False


def test_v3_promotion_requires_holm_control_across_screened_candidates():
    rows = []
    for candidate, reward_p_holm in [
        ("QMIX Majority Ensemble", 0.051),
        ("QMIX Unanimous Ensemble", 0.01),
    ]:
        for metric in PRIMARY_METRICS:
            rows.append({
                "candidate": candidate,
                "comparator": "Extended QMIX Replica Mean",
                "metric": metric,
                "engineering_advantage_mean": 1.0,
                "ci95_low": 0.5,
                "reward_p_holm": reward_p_holm if metric == "raw_episode_reward" else np.nan,
            })
    decision = _promotion_decision(pd.DataFrame(rows))
    assert decision["candidate_evidence"]["QMIX Majority Ensemble"]["passed"] is False
    assert decision["candidate_evidence"]["QMIX Unanimous Ensemble"]["passed"] is True
    assert decision["promoted_candidate"] == "QMIX Unanimous Ensemble"
