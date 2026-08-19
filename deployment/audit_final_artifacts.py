"""Fail-fast integrity audit for the final research artifact set."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
from typing import Any, Dict, List

import onnxruntime as ort
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_provenance(path: Path) -> Dict[str, Any]:
    """Re-hash every declared file instead of trusting archive existence."""
    payload = json.loads(path.read_text())
    missing: List[str] = []
    mismatched: Dict[str, Dict[str, str]] = {}
    checked = 0
    for section in ("source_sha256", "primary_artifact_sha256"):
        for relative, expected in payload.get(section, {}).items():
            # The audit writes this file itself, so including it would create a
            # self-invalidating provenance cycle in historical archives.
            if relative == "results/final_audit.json":
                continue
            target = ROOT_DIR / relative
            if not target.is_file():
                missing.append(relative)
                continue
            actual = _sha256(target)
            checked += 1
            if actual != expected:
                mismatched[relative] = {"expected": expected, "actual": actual}
    return {
        "passed": not missing and not mismatched,
        "checked_files": checked,
        "missing": sorted(missing),
        "mismatched": mismatched,
    }


def audit() -> Path:
    checks: List[Dict[str, Any]] = []

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(condition), "evidence": evidence})

    check(
        "seed_partitions_disjoint",
        not (
            set(shared_config.TRAIN_SEEDS) & set(shared_config.VAL_SEEDS)
            or set(shared_config.TRAIN_SEEDS) & set(shared_config.TEST_SEEDS)
            or set(shared_config.VAL_SEEDS) & set(shared_config.TEST_SEEDS)
        ),
        {
            "train": shared_config.TRAIN_SEEDS[:3],
            "validation": shared_config.VAL_SEEDS,
            "test": shared_config.TEST_SEEDS,
        },
    )

    sanity = [
        json.loads((ROOT_DIR / "results" / "sanity" / f"{regime}_volatile.json").read_text())
        for regime in ("independent", "coordinated")
    ]
    check("sanity_gates", all(item["all_passed"] for item in sanity), [item["config_digest"] for item in sanity])

    # A passing gate artifact proves nothing if the environment source moved
    # after the gate ran, so recompute the digest instead of trusting the
    # stored one.  This is the check whose absence let an environment repair
    # land silently underneath a complete set of published results.
    from training.sanity_checks import config_digest

    recomputed = {
        regime: config_digest("volatile", regime)
        for regime in ("independent", "coordinated")
    }
    recorded = {item["regime"]: item["config_digest"] for item in sanity}
    check(
        "sanity_digest_matches_current_source",
        recomputed == recorded,
        {"recorded": recorded, "recomputed": recomputed},
    )

    # Historical manifests legitimately carry pre-repair digests.  That is
    # allowed only while it stays declared, so the audit fails if drift appears
    # that the disclosure record does not already name.
    drift_path = ROOT_DIR / "results" / "environment_drift.json"
    manifest_digests = set()
    for manifest_path in sorted(
        (ROOT_DIR / "results" / "upgrade_experiments").glob("training_manifest_*.json")
    ):
        for item in json.loads(manifest_path.read_text()):
            if item.get("config_digest"):
                manifest_digests.add(item["config_digest"])
    if drift_path.is_file():
        drift = json.loads(drift_path.read_text())
        declared = set(drift.get("config_digest_drift", {}).get("legacy", {}).values())
        declared |= set(drift.get("config_digest_drift", {}).get("current", {}).values())
    else:
        drift = None
        declared = set()
    undeclared = sorted(manifest_digests - set(recomputed.values()) - declared)
    check(
        "environment_drift_disclosed",
        drift is not None and not undeclared,
        {
            "disclosure_record": str(drift_path.relative_to(ROOT_DIR)) if drift is not None else None,
            "manifest_digests": sorted(manifest_digests),
            "undeclared_digests": undeclared,
        },
    )

    full = json.loads((ROOT_DIR / "results" / "experiments" / "training_manifest_full.json").read_text())
    check(
        "full_training_matrix",
        len(full) == 18 and all(item["status"] == "SUCCESS" for item in full),
        {"runs": len(full), "successes": sum(item["status"] == "SUCCESS" for item in full)},
    )
    check(
        "selected_policies_use_both_actions",
        all(item["selected_checkpoint"]["chooses_both_actions"] for item in full),
        sorted({item["selected_checkpoint"]["chooses_both_actions"] for item in full}),
    )

    expected_policies = {
        "Always Sleep", "Always Sample", "Random Feasible", "Fixed Interval",
        "Entropy Threshold", "Battery + Entropy", "Greedy", "IQL", "VDN", "QMIX",
    }
    for regime in ("independent", "coordinated"):
        root = ROOT_DIR / "results" / "final" / regime
        raw = pd.read_csv(root / "benchmark_raw.csv")
        per_env = pd.read_csv(root / "benchmark_per_environment_seed.csv")
        summary = pd.read_csv(root / "benchmark_summary.csv")
        paired = pd.read_csv(root / "paired_comparisons.csv")
        counts = raw.groupby("policy").size().to_dict()
        count_ok = all(
            counts.get(policy) == (90 if policy in {"IQL", "VDN", "QMIX"} else 30)
            for policy in expected_policies
        )
        check(
            f"{regime}_benchmark_balance",
            set(raw.policy) == expected_policies
            and count_ok
            and len(per_env) == 300
            and set(per_env.environment_seed) == set(shared_config.TEST_SEEDS),
            counts,
        )
        check(
            f"{regime}_summary_and_statistics",
            len(summary) == 10
            and (summary.n_environment_seeds == 30).all()
            and len(paired) == 24
            and set(paired.comparator) == {"Battery + Entropy", "Entropy Threshold"},
            {"summary_rows": len(summary), "paired_rows": len(paired)},
        )

    variants = [
        "no_neighbor_signal", "no_redundancy", "no_aoi", "no_energy",
        "no_agent_id", "no_coordination_constraint",
    ]
    manifests = {
        variant: json.loads(
            (ROOT_DIR / "results" / "experiments" / f"training_manifest_{variant}.json").read_text()
        ) for variant in variants
    }
    check(
        "ablation_training_matrix",
        all(len(items) == 3 and all(item["status"] == "SUCCESS" for item in items) for items in manifests.values()),
        {variant: len(items) for variant, items in manifests.items()},
    )

    ablation_raw = pd.read_csv(ROOT_DIR / "results" / "ablations" / "ablation_raw.csv")
    ablation_summary = pd.read_csv(ROOT_DIR / "results" / "ablations" / "ablation_summary.csv")
    ablation_counts = ablation_raw.groupby("ablation").size().to_dict()
    check(
        "ablation_evaluation_balance",
        set(ablation_counts) == {"full", *variants}
        and all(value == 90 for value in ablation_counts.values())
        and len(ablation_summary) == 7,
        ablation_counts,
    )

    input_dims = {}
    for variant in variants:
        path = ROOT_DIR / "results" / "ablation_models" / variant / "coordinated" / "qmix_seed101.onnx"
        input_dims[variant] = int(ort.InferenceSession(str(path)).get_inputs()[0].shape[1])
    check(
        "ablation_model_contracts",
        input_dims["no_agent_id"] == shared_config.ENV_OBS_DIM
        and all(
            dim == shared_config.MODEL_INPUT_DIM
            for variant, dim in input_dims.items()
            if variant != "no_agent_id"
        ),
        input_dims,
    )

    figures = sorted((ROOT_DIR / "results" / "figures").glob("[0-9][0-9]_*.png"))
    check("thirteen_final_figures", len(figures) == 13, [path.name for path in figures])
    check("final_report", (ROOT_DIR / "FINAL_RESEARCH_REPORT.md").exists(), "FINAL_RESEARCH_REPORT.md")
    provenance_path = ROOT_DIR / "results" / "provenance.json"
    if provenance_path.is_file():
        provenance = verify_provenance(provenance_path)
        check("provenance_hashes", provenance["passed"], provenance)
    else:
        check("provenance_hashes", False, {"missing": ["results/provenance.json"]})

    report = {"all_passed": all(item["passed"] for item in checks), "checks": checks}
    output = ROOT_DIR / "results" / "final_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_passed"]:
        failed = [item["name"] for item in checks if not item["passed"]]
        raise RuntimeError(f"Final artifact audit failed: {failed}")
    print(output)
    return output


if __name__ == "__main__":
    audit()
