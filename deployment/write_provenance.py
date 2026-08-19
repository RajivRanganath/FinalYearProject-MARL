"""Write source and primary-artifact hashes for the final experiment archive."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT_DIR = Path(__file__).resolve().parent.parent


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, check=False
    )
    return result.stdout


def write_provenance() -> Path:
    source_patterns = [
        "shared_config.py",
        "environment/*.py",
        "training/*.py",
        "training/epymarl/src/config/algs/{iql,vdn,qmix}.yaml",
        "training/epymarl/src/config/envs/iot.yaml",
        "training/epymarl/src/learners/q_learner.py",
        "deployment/*.py",
        "tests/*.py",
    ]
    source_paths = set()
    for pattern in source_patterns:
        if "{" in pattern:
            prefix, variants, suffix = pattern.partition("{")
            options, _, tail = suffix.partition("}")
            source_paths.update(ROOT_DIR / f"{prefix}{name}{tail}" for name in options.split(","))
        else:
            source_paths.update(ROOT_DIR.glob(pattern))

    artifact_patterns = [
        "results/sanity/*.json",
        "results/experiments/training_manifest_*.json",
        "results/learned_models/**/*.onnx",
        "results/learned_models/**/*.onnx.data",
        "results/ablation_models/**/*.onnx",
        "results/ablation_models/**/*.onnx.data",
        "results/upgrade_models/**/*.onnx",
        "results/upgrade_models/**/*.onnx.data",
        "results/final/**/*.csv",
        "results/ablations/*",
        "results/figures/*",
        # The split-locked holdouts are the artifacts that most need tamper
        # evidence: they were evaluated exactly once and can never be
        # regenerated, so an unnoticed edit would be unrecoverable.
        "results/training_v2/**/*.csv",
        "results/training_v2/**/*.json",
        "results/training_v2/**/*.md",
        "results/training_v3/**/*.csv",
        "results/training_v3/**/*.json",
        "results/training_v3/**/*.md",
        "results/training_v4/**/*.csv",
        "results/training_v4/**/*.json",
        "results/training_v4/**/*.md",
        "results/upgrade_experiments/training_manifest_*.json",
        "results/environment_drift.json",
        "results/environment_drift_impact.json",
        "FINAL_RESEARCH_REPORT.md",
    ]
    artifact_paths = set()
    for pattern in artifact_patterns:
        artifact_paths.update(path for path in ROOT_DIR.glob(pattern) if path.is_file())

    status = _git("status", "--porcelain")
    diff = _git("diff", "--binary")
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_base_sha": _git("rev-parse", "HEAD").strip() or "unknown",
        "git_worktree_dirty": bool(status.strip()),
        "git_status_porcelain": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "note": (
            "This record fingerprints the source and artifacts as they stand when it is written; "
            "it is not a claim that the base SHA is the source that produced the artifacts. Training "
            "ran earlier, in a dirty worktree, and two source repairs postdate every result here. "
            "See results/environment_drift.json for what changed and results/environment_drift_impact.json "
            "for the measured effect. A clean git_worktree_dirty flag below means only that the tree "
            "was committed before hashing. External ONNX data files are hashed separately because "
            "they contain model tensors."
        ),
        "source_sha256": {
            str(path.relative_to(ROOT_DIR)): _sha(path)
            for path in sorted(source_paths)
            if path.is_file()
        },
        "primary_artifact_sha256": {
            str(path.relative_to(ROOT_DIR)): _sha(path)
            for path in sorted(artifact_paths)
        },
    }
    output = ROOT_DIR / "results" / "provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return output


if __name__ == "__main__":
    write_provenance()
