"""Audit the rejected Refined QMIX training/selection experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from deployment.evaluate_training_v3 import V3_FINAL_SEEDS, V3_SELECTION_SEEDS
from deployment.evaluate_training_v4 import ANALYSIS_FILES
from training.refined_protocol import REFINED_FINAL_SEEDS, REFINED_SELECTION_SEEDS


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def audit() -> dict:
    root = ROOT_DIR / "results" / "training_v4"
    decision = json.loads((root / "selection" / "selection_decision.json").read_text())
    manifest = json.loads((root / "selection" / "evaluation_manifest.json").read_text())
    training = json.loads(
        (ROOT_DIR / "results" / "upgrade_experiments" / "training_manifest_refined_full.json").read_text()
    )
    raw = _rows(root / "selection" / "raw.csv")
    invalidation = json.loads((root / "INVALIDATED.json").read_text())
    optimizer_lrs = {}
    all_optimizer_lrs = []
    resume_evidence = {}
    weights_changed = {}
    for run in training:
        checkpoint = Path(run["selected_checkpoint"]["checkpoint"])
        state = torch.load(checkpoint / "opt.th", map_location="cpu", weights_only=True)
        optimizer_lrs[str(run["seed"])] = [
            float(group["lr"]) for group in state["param_groups"]
        ]
        for optimizer_path in sorted(Path(run["run_dir"]).glob("models/*/*/opt.th")):
            optimizer = torch.load(
                optimizer_path, map_location="cpu", weights_only=True
            )
            all_optimizer_lrs.append([
                float(group["lr"]) for group in optimizer["param_groups"]
            ])
        source_root = Path(run["resume_checkpoint_root"])
        available_steps = sorted(
            int(path.name) for path in source_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
        command_checkpoint = next(
            value.split("=", 1)[1] for value in run["command"]
            if value.startswith("checkpoint_path=")
        )
        requested_step = int(next(
            value.split("=", 1)[1] for value in run["command"]
            if value.startswith("load_step=")
        ))
        actual_step = max(available_steps) if requested_step == 0 else max(
            step for step in available_steps if step <= requested_step
        )
        source_checkpoint = source_root / str(actual_step)
        required_files = [source_checkpoint / name for name in ("agent.th", "mixer.th", "opt.th")]
        resume_evidence[str(run["seed"])] = {
            "command_root_matches_manifest": Path(command_checkpoint) == source_root,
            "requested_step": requested_step,
            "actual_resolved_step": actual_step,
            "manifest_resolved_step": run["resume_checkpoint_step_resolved"],
            "required_files_present": all(path.is_file() for path in required_files),
            "source_sha256": {
                path.name: _sha(path) for path in required_files if path.is_file()
            },
        }
        weights_changed[str(run["seed"])] = (
            _sha(source_checkpoint / "agent.th") != _sha(checkpoint / "agent.th")
        )
    current_models = {name: _sha(ROOT_DIR / name) for name in decision["model_fingerprints"]}
    changed_analysis = {
        name: {"frozen": frozen, "current": _sha(ROOT_DIR / name)}
        for name, frozen in decision["analysis_fingerprints"].items()
        if _sha(ROOT_DIR / name) != frozen
    }
    required_analysis = {str(path) for path in ANALYSIS_FILES}
    historically_unfingerprinted = sorted(
        required_analysis - set(decision["analysis_fingerprints"])
    )
    actual_selection_seeds = {int(row["environment_seed"]) for row in raw}
    actual_policies = {row["policy"] for row in raw}
    learned_training_seeds = {
        int(float(row["training_seed"]))
        for row in raw
        if row["training_seed"]
    }
    all_protocol_splits = [
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
    disjoint = all(
        first.isdisjoint(second)
        for index, first in enumerate(all_protocol_splits)
        for second in all_protocol_splits[index + 1:]
    )
    checks = [
        {
            "name": "strict_gate_rejected_candidate",
            "passed": decision["promote_refined"] is False,
            "evidence": decision["evidence"],
        },
        {
            "name": "optimizer_lr_mismatch_detected_and_v4_invalidated",
            "passed": (
                len(training) == 3
                and all(run["status"] == "SUCCESS" for run in training)
                and all(run["learning_rate"] == 1e-4 for run in training)
                and all(values == [3e-4] for values in optimizer_lrs.values())
                and len(all_optimizer_lrs) == 37
                and all(values == [3e-4] for values in all_optimizer_lrs)
                and all(weights_changed.values())
                and invalidation["status"] == "INVALIDATED_PROTOCOL"
            ),
            "evidence": {
                "declared": {str(run["seed"]): run["learning_rate"] for run in training},
                "serialized_optimizer": optimizer_lrs,
                "all_refined_checkpoints": len(all_optimizer_lrs),
                "all_serialized_lrs": sorted({tuple(values) for values in all_optimizer_lrs}),
                "weights_changed_from_source": weights_changed,
                "invalidation": invalidation["status"],
            },
        },
        {
            "name": "exact_resume_steps_audited",
            "passed": (
                {
                    run["seed"]: run["resume_checkpoint_step_resolved"]
                    for run in training
                } == {101: 360288, 102: 360288, 103: 289440}
                and all(
                    item["command_root_matches_manifest"]
                    and item["actual_resolved_step"] == item["manifest_resolved_step"]
                    and item["required_files_present"]
                    for item in resume_evidence.values()
                )
            ),
            "evidence": resume_evidence,
        },
        {
            "name": "selected_checkpoints_choose_both_actions",
            "passed": all(run["selected_checkpoint"]["chooses_both_actions"] for run in training),
            "evidence": {
                str(run["seed"]): run["selected_checkpoint"]["action_counts"] for run in training
            },
        },
        {
            "name": "selection_contract_complete_exact_and_disjoint",
            "passed": (
                len(raw) == 160 and len({
                (row["policy"], row["training_seed"], row["environment_seed"]) for row in raw
                }) == 160
                and actual_selection_seeds == set(REFINED_SELECTION_SEEDS)
                and actual_policies == {
                    "Extended QMIX", "Refined QMIX", "Entropy Threshold", "Battery + Entropy"
                }
                and learned_training_seeds == {101, 102, 103}
                and manifest["environment_seeds"] == list(REFINED_SELECTION_SEEDS)
                and decision["selection_seeds"] == list(REFINED_SELECTION_SEEDS)
                and decision["final_seeds_locked"] == list(REFINED_FINAL_SEEDS)
                and disjoint
            ),
            "evidence": {
                "rows": len(raw),
                "selection_seeds": sorted(actual_selection_seeds),
                "policies": sorted(actual_policies),
                "training_seeds": sorted(learned_training_seeds),
                "all_splits_disjoint": disjoint,
            },
        },
        {
            "name": "model_hashes_match_selection_and_disk",
            "passed": current_models == decision["model_fingerprints"] == manifest["model_fingerprints"],
            "evidence": {"files": len(current_models)},
        },
        {
            "name": "frozen_analysis_snapshot_incomplete_and_current_code_separated",
            "passed": (
                decision["analysis_fingerprints"] == manifest["analysis_fingerprints"]
                and bool(changed_analysis)
                and bool(historically_unfingerprinted)
                and invalidation["status"] == "INVALIDATED_PROTOCOL"
            ),
            "evidence": {
                "changed_after_frozen_selection": sorted(changed_analysis),
                "historically_unfingerprinted_dependencies": historically_unfingerprinted,
            },
        },
        {
            "name": "locked_final_was_not_consumed",
            "passed": (
                not (root / "final" / "raw.csv").exists()
                and not (root / "final" / "CONSUMPTION_STARTED.json").exists()
            ),
            "evidence": "v4 final raw data and consumption sentinel absent",
        },
    ]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
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
