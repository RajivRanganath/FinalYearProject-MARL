from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

EPYMARL_SRC = Path(__file__).resolve().parents[1] / "training" / "epymarl" / "src"
if str(EPYMARL_SRC) not in sys.path:
    sys.path.insert(0, str(EPYMARL_SRC))

from deployment.audit_final_artifacts import verify_provenance
from hardware_eval.model_analysis import analyze_marl_policy
from hardware_eval.quantize_model import run_quantization_evaluation
from modules.agents.rnn_agent import RNNAgent
from training.export_onnx import export_agent_to_onnx


def test_hardware_analysis_reads_recurrent_ensemble_graph():
    report = analyze_marl_policy()
    assert report["architecture"] == "shared recurrent GRU policy"
    assert report["input_dim"] == 5
    assert report["hidden_dim"] == 64
    assert report["parameters_per_replica"] == 25_474
    assert report["total_parameters"] == 3 * 25_474
    assert report["macs_per_replica_inference"] == 25_024
    assert report["total_macs_per_agent_decision"] == 3 * 25_024
    assert report["status"] == "ANALYTICAL_GRAPH_ACCOUNTING_ONLY"


def test_quantizer_refuses_float_model_as_its_output():
    report = analyze_marl_policy(replica_count=1)
    model_path = report["model_path"]
    with pytest.raises(ValueError, match="must not be the float input"):
        run_quantization_evaluation(model_path, model_path, n_test_samples=1)


def test_export_marks_recurrent_outputs_with_dynamic_batch(tmp_path):
    args = SimpleNamespace(hidden_dim=64, n_actions=2, use_rnn=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save(RNNAgent(5, args).state_dict(), checkpoint / "agent.th")
    output = tmp_path / "policy.onnx"
    metadata = export_agent_to_onnx(checkpoint, output)
    graph = onnx.load(output)
    output_shapes = {
        value.name: [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]
        for value in graph.graph.output
    }
    assert output_shapes == {
        "q_values": ["batch_size", 2],
        "hidden_state_out": ["batch_size", 64],
    }
    session = ort.InferenceSession(str(output))
    values = session.run(None, {
        "obs": np.zeros((2, 5), dtype=np.float32),
        "hidden_state_in": np.zeros((2, 64), dtype=np.float32),
    })
    assert [value.shape for value in values] == [(2, 2), (2, 64)]
    assert metadata["total_parameters"] == 25_474


def test_provenance_verifier_detects_content_changes(tmp_path, monkeypatch):
    import deployment.audit_final_artifacts as audit_module

    source = tmp_path / "source.py"
    source.write_text("before\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    archive = tmp_path / "provenance.json"
    archive.write_text(json.dumps({
        "source_sha256": {"source.py": digest},
        "primary_artifact_sha256": {},
    }))
    monkeypatch.setattr(audit_module, "ROOT_DIR", tmp_path)
    assert verify_provenance(archive)["passed"] is True
    source.write_text("after\n")
    result = verify_provenance(archive)
    assert result["passed"] is False
    assert "source.py" in result["mismatched"]


def test_invalidated_v4_cannot_consume_untouched_final_split(tmp_path, monkeypatch):
    import deployment.evaluate_training_v4 as evaluator

    root = tmp_path / "results" / "training_v4"
    root.mkdir(parents=True)
    (root / "INVALIDATED.json").write_text(json.dumps({
        "status": "INVALIDATED_PROTOCOL",
    }))
    monkeypatch.setattr(evaluator, "ROOT_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="untouched final split must not be consumed"):
        evaluator.evaluate("final")


def test_config_digest_tracks_environment_source_changes(tmp_path, monkeypatch):
    """The gate digest must move when environment behavior moves.

    Before this closure the digest covered only config values, so the
    sample-feasibility repair changed the environment underneath a full set of
    published results without invalidating a single gate artifact.
    """
    import shutil

    import training.sanity_checks as sanity_module

    environment = tmp_path / "environment"
    environment.mkdir()
    for name in (
        "energy_model.py",
        "single_agent_env.py",
        "multi_agent_env.py",
        "pettingzoo_env.py",
    ):
        shutil.copy(Path(sanity_module.ROOT_DIR) / "environment" / name, environment / name)

    monkeypatch.setattr(sanity_module, "ROOT_DIR", tmp_path)
    before = sanity_module.config_digest("volatile", "coordinated")

    target = environment / "single_agent_env.py"
    target.write_text(target.read_text() + "\n# behavior-changing edit\n")
    after = sanity_module.config_digest("volatile", "coordinated")

    assert before != after


def test_recorded_sanity_digest_matches_current_environment_source():
    """Guards against results whose gate artifact predates a source repair."""
    from training.sanity_checks import config_digest

    root = Path(__file__).resolve().parents[1]
    for regime in ("independent", "coordinated"):
        recorded = json.loads((root / "results" / "sanity" / f"{regime}_volatile.json").read_text())
        assert recorded["config_digest"] == config_digest("volatile", regime), (
            f"{regime} gate artifact was produced by a different environment source"
        )


def test_every_manifest_digest_is_either_current_or_declared_drift():
    """Pre-repair digests are allowed only while the drift record names them."""
    from training.sanity_checks import config_digest

    root = Path(__file__).resolve().parents[1]
    current = {config_digest("volatile", regime) for regime in ("independent", "coordinated")}
    drift = json.loads((root / "results" / "environment_drift.json").read_text())
    declared = set(drift["config_digest_drift"]["legacy"].values())
    declared |= set(drift["config_digest_drift"]["current"].values())

    seen = set()
    for manifest in sorted((root / "results" / "upgrade_experiments").glob("training_manifest_*.json")):
        for item in json.loads(manifest.read_text()):
            if item.get("config_digest"):
                seen.add(item["config_digest"])

    assert seen, "no upgrade manifests found"
    assert not (seen - current - declared)


def test_training_manifest_fingerprints_the_learner_source():
    """A dirty-worktree git SHA cannot identify the code that actually ran."""
    from training.train_all import TRAINING_SOURCE_FILES, _training_source_digest

    assert "training/epymarl/src/learners/q_learner.py" in TRAINING_SOURCE_FILES
    digest = _training_source_digest()
    assert digest["training/epymarl/src/learners/q_learner.py"]
    assert len(set(digest.values())) == len(digest)


def test_provenance_covers_the_consumed_split_locked_holdouts():
    """The once-only holdouts are exactly the files that need tamper evidence."""
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "results" / "provenance.json").read_text())
    archived = set(payload["primary_artifact_sha256"])
    for required in (
        "results/training_v2/final/raw.csv",
        "results/training_v3/final/raw.csv",
        "results/training_v4/INVALIDATED.json",
        "results/environment_drift.json",
        "results/upgrade_experiments/training_manifest_extended_full.json",
    ):
        assert required in archived, f"{required} is not hashed in the provenance archive"
