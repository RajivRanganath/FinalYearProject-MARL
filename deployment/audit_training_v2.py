"""Audit the frozen split-locked QMIX continuation artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
EXPECTED_POLICIES = {
    "Always Sleep",
    "Always Sample",
    "Random Feasible",
    "Fixed Interval",
    "Entropy Threshold",
    "Battery + Entropy",
    "Greedy",
    "Published IQL",
    "Published VDN",
    "Published QMIX",
    "Improved QMIX",
    "Extended QMIX",
}
STATIC_POLICIES = {
    "Always Sleep",
    "Always Sample",
    "Random Feasible",
    "Fixed Interval",
    "Entropy Threshold",
    "Battery + Entropy",
    "Greedy",
}
LEARNED_POLICIES = EXPECTED_POLICIES - STATIC_POLICIES
EXPECTED_TRAINING_SEEDS = {101, 102, 103}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _check(condition: bool, name: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "evidence": evidence}


def audit() -> dict[str, Any]:
    root = ROOT_DIR / "results" / "training_v2"
    selection = _load_json(root / "selection" / "selection_decision.json")
    final_manifest = _load_json(root / "final" / "evaluation_manifest.json")
    extended_manifest = _load_json(
        ROOT_DIR / "results" / "upgrade_experiments" / "training_manifest_extended_full.json"
    )
    with (root / "final" / "raw.csv").open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    with (root / "final" / "summary.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    with (root / "final" / "paired_comparisons.csv").open(newline="") as handle:
        comparisons = list(csv.DictReader(handle))

    selection_seeds = {int(seed) for seed in selection["selection_seeds"]}
    final_seeds = {int(seed) for seed in final_manifest["environment_seeds"]}
    validation_seeds = {
        int(seed) for run in extended_manifest for seed in run["validation_seeds"]
    }
    raw_final_seeds = {int(row["environment_seed"]) for row in raw}
    raw_policies = {row["policy"] for row in raw}

    model_hashes = {
        relative: _sha256(ROOT_DIR / relative)
        for relative in final_manifest["model_fingerprints"]
    }
    analysis_hashes = {
        relative: _sha256(ROOT_DIR / relative)
        for relative in final_manifest["analysis_fingerprints"]
    }

    learned_seed_map = {
        policy: {
            int(float(row["training_seed"]))
            for row in raw
            if row["policy"] == policy and row["training_seed"]
        }
        for policy in LEARNED_POLICIES
    }
    static_training_seeds = {
        row["training_seed"]
        for row in raw
        if row["policy"] in STATIC_POLICIES
    }
    unique_keys = {
        (row["policy"], row["training_seed"], row["environment_seed"])
        for row in raw
    }
    comparison_keys = {
        (row["comparator"], row["metric"])
        for row in comparisons
    }
    expected_comparison_keys = {
        (comparator, metric)
        for comparator in final_manifest["primary_comparators"]
        for metric in final_manifest["primary_metrics"]
    }
    action_evidence = {
        str(run["seed"]): run["selected_checkpoint"]["action_counts"]
        for run in extended_manifest
    }

    checks = [
        _check(selection["promote_extended"] is True, "selection_gate_promoted", selection["reward_advantage_by_training_seed"]),
        _check(
            selection_seeds.isdisjoint(final_seeds)
            and validation_seeds.isdisjoint(selection_seeds)
            and validation_seeds.isdisjoint(final_seeds),
            "validation_selection_final_splits_disjoint",
            {
                "validation": sorted(validation_seeds),
                "selection": sorted(selection_seeds),
                "final": sorted(final_seeds),
            },
        ),
        _check(len(final_seeds) == 30 and raw_final_seeds == final_seeds, "locked_final_seed_coverage", sorted(final_seeds)),
        _check(len(raw) == 660 and len(unique_keys) == 660, "raw_row_cardinality_and_uniqueness", len(raw)),
        _check(raw_policies == EXPECTED_POLICIES and len(summary) == 12, "policy_and_summary_coverage", sorted(raw_policies)),
        _check(
            all(seeds == EXPECTED_TRAINING_SEEDS for seeds in learned_seed_map.values())
            and static_training_seeds == {""},
            "training_replica_labels",
            {key: sorted(value) for key, value in sorted(learned_seed_map.items())},
        ),
        _check(comparison_keys == expected_comparison_keys and len(comparisons) == 12, "predeclared_comparison_family", sorted(comparison_keys)),
        _check(
            all(run["status"] == "SUCCESS" for run in extended_manifest)
            and {run["seed"] for run in extended_manifest} == EXPECTED_TRAINING_SEEDS,
            "all_extended_training_replicas_succeeded",
            {str(run["seed"]): run["status"] for run in extended_manifest},
        ),
        _check(
            all(len(counts) == 2 and all(int(count) > 0 for count in counts) for counts in action_evidence.values()),
            "selected_checkpoints_choose_both_actions",
            action_evidence,
        ),
        _check(
            model_hashes == final_manifest["model_fingerprints"]
            and model_hashes == selection["model_fingerprints"],
            "model_fingerprints_match_selection_final_and_disk",
            {"files": len(model_hashes)},
        ),
        _check(
            analysis_hashes == final_manifest["analysis_fingerprints"]
            and analysis_hashes == selection["analysis_fingerprints"],
            "analysis_fingerprints_match_selection_final_and_disk",
            {"files": len(analysis_hashes)},
        ),
        _check(
            final_manifest["bootstrap_replicates"] == 5000
            and final_manifest["bootstrap_seed"] == 25033,
            "bootstrap_protocol_frozen",
            {
                "replicates": final_manifest["bootstrap_replicates"],
                "seed": final_manifest["bootstrap_seed"],
            },
        ),
    ]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON audit artifact path")
    args = parser.parse_args()
    report = audit()
    for check in report["checks"]:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['evidence']}")
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT_DIR / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(output)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
