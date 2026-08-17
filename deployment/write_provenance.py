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
        "results/ablation_models/**/*.onnx",
        "results/final/**/*.csv",
        "results/ablations/*",
        "results/figures/*",
        "results/final_audit.json",
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
            "Full training ran in this disclosed dirty worktree. Later report/plotting and "
            "provenance-only edits do not retroactively make the base SHA a clean source snapshot."
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
