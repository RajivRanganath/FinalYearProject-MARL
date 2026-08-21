"""Audit the frozen v3 deployment-ensemble selection and final artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
EXPECTED_FINAL_POLICIES = {
    "Extended QMIX Replica Mean",
    "QMIX Unanimous Ensemble",
    "Entropy Threshold",
    "Battery + Entropy",
    "Greedy",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _check(condition: bool, name: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "evidence": evidence}


def audit() -> dict[str, Any]:
    root = ROOT_DIR / "results" / "training_v3"
    decision = _json(root / "selection" / "selection_decision.json")
    selection_manifest = _json(root / "selection" / "evaluation_manifest.json")
    final_manifest = _json(root / "final" / "evaluation_manifest.json")
    raw = _rows(root / "final" / "raw.csv")
    summary = _rows(root / "final" / "summary.csv")
    comparisons = _rows(root / "final" / "candidate_comparisons.csv")

    selection_seeds = {int(seed) for seed in selection_manifest["environment_seeds"]}
    final_seeds = {int(seed) for seed in final_manifest["environment_seeds"]}
    previous_seeds = (
        set(range(201, 251))
        | set(range(1001, 1031))
        | set(range(2001, 2031))
        | set(range(3001, 3031))
    )
    current_models = {
        name: _sha256(ROOT_DIR / name) for name in final_manifest["model_fingerprints"]
    }
    current_analysis = {
        name: _sha256(ROOT_DIR / name) for name in final_manifest["analysis_fingerprints"]
    }
    policies = {row["policy"] for row in raw}
    row_keys = {
        (row["policy"], row["training_seed"], row["environment_seed"])
        for row in raw
    }
    reference_seeds = {
        int(float(row["training_seed"]))
        for row in raw
        if row["policy"] == "Extended QMIX Replica Mean"
    }
    candidate_rows = [row for row in raw if row["policy"] == "QMIX Unanimous Ensemble"]
    reference_reward = next(
        float(row["raw_episode_reward_mean"])
        for row in summary
        if row["policy"] == "Extended QMIX Replica Mean"
    )
    candidate_reward = next(
        float(row["raw_episode_reward_mean"])
        for row in summary
        if row["policy"] == "QMIX Unanimous Ensemble"
    )
    reward_comparison = next(
        row for row in comparisons
        if row["comparator"] == "Extended QMIX Replica Mean"
        and row["metric"] == "raw_episode_reward"
    )
    historical_reward_p_holm = float(
        decision["candidate_evidence"]["QMIX Unanimous Ensemble"]["reward_p_holm"]
    )

    checks = [
        _check(
            decision["promoted_candidate"] == "QMIX Unanimous Ensemble",
            "predeclared_selection_gate_promoted_unanimous",
            decision["candidate_evidence"],
        ),
        _check(
            historical_reward_p_holm <= 0.05,
            "historical_winner_also_passes_current_holm_guard",
            historical_reward_p_holm,
        ),
        _check(
            len(selection_seeds) == 20
            and len(final_seeds) == 30
            and selection_seeds.isdisjoint(final_seeds)
            and final_seeds.isdisjoint(previous_seeds),
            "new_final_split_is_disjoint",
            {"selection": sorted(selection_seeds), "final": sorted(final_seeds)},
        ),
        _check(
            len(raw) == 210 and len(row_keys) == 210,
            "final_row_cardinality_and_uniqueness",
            len(raw),
        ),
        _check(
            policies == EXPECTED_FINAL_POLICIES and len(summary) == 5,
            "final_policy_coverage",
            sorted(policies),
        ),
        _check(
            reference_seeds == {101, 102, 103}
            and len(candidate_rows) == 30
            and all(not row["training_seed"] for row in candidate_rows),
            "replica_and_ensemble_labels",
            {"reference_training_seeds": sorted(reference_seeds), "ensemble_rows": len(candidate_rows)},
        ),
        _check(
            len(comparisons) == 12
            and {row["comparator"] for row in comparisons}
            == {"Extended QMIX Replica Mean", "Entropy Threshold"},
            "predeclared_final_comparison_family",
            len(comparisons),
        ),
        _check(
            current_models == selection_manifest["model_fingerprints"]
            == final_manifest["model_fingerprints"]
            == decision["model_fingerprints"],
            "model_fingerprints_match_selection_final_and_disk",
            {"files": len(current_models)},
        ),
        _check(
            current_analysis == selection_manifest["analysis_fingerprints"]
            == final_manifest["analysis_fingerprints"]
            == decision["analysis_fingerprints"],
            "analysis_fingerprints_match_selection_final_and_disk",
            {"files": len(current_analysis)},
        ),
        _check(
            candidate_reward > reference_reward
            and float(reward_comparison["ci95_low"]) > 0.0,
            "final_reward_gain_replication",
            {
                "reference": reference_reward,
                "candidate": candidate_reward,
                "advantage_ci95": [
                    float(reward_comparison["ci95_low"]),
                    float(reward_comparison["ci95_high"]),
                ],
            },
        ),
        _check(
            (root / "final" / "raw.csv").exists(),
            "final_consumption_guard_will_block_rerun",
            "results/training_v3/final/raw.csv",
        ),
    ]
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    for check in report["checks"]:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['evidence']}")
    if args.output:
        path = args.output if args.output.is_absolute() else ROOT_DIR / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(path)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
