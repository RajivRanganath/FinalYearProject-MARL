"""Consolidate the three completed Refined QMIX run summaries."""

from __future__ import annotations

import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
RUNS = {
    101: "20260819T204004.589881Z",
    102: "20260819T204911.075929Z",
    103: "20260819T205937.331224Z",
}


def main() -> None:
    source_manifest = json.loads(
        (
            ROOT_DIR / "results" / "upgrade_experiments"
            / "training_manifest_extended_full.json"
        ).read_text()
    )
    summaries = []
    for seed, run_id in RUNS.items():
        path = (
            ROOT_DIR / "results" / "upgrade_experiments" / "coordinated"
            / "qmix" / f"seed{seed}" / run_id / "summary.json"
        )
        summary = json.loads(path.read_text())
        source = next(
            item for item in source_manifest
            if item["seed"] == seed
            and item["scenario"] == summary["scenario"]
            and item["regime"] == summary["regime"]
            and item["algorithm"] == summary["algorithm"]
            and item["ablation"] == "full"
        )
        selected_source_step = int(Path(source["selected_checkpoint"]["checkpoint"]).name)
        command_step = next(
            int(value.split("=", 1)[1])
            for value in summary["command"]
            if value.startswith("load_step=")
        )
        checkpoint_root = Path(summary["resume_checkpoint_root"])
        available = sorted(
            int(path.name) for path in checkpoint_root.iterdir()
            if path.is_dir() and path.name.isdigit() and (path / "agent.th").is_file()
        )
        resolved_step = (
            max(available)
            if command_step == 0
            else min(available, key=lambda step: abs(step - command_step))
        )
        if resolved_step != selected_source_step:
            raise RuntimeError(
                f"Seed {seed} resolved source {resolved_step}, expected validation-selected "
                f"step {selected_source_step}"
            )
        summary["resume_checkpoint_step_resolved"] = resolved_step
        summary["resume_resolution_audit"] = (
            "exact load_step"
            if command_step == resolved_step
            else "legacy load_step=0 resolved latest; latest equals validation-selected step"
        )
        summaries.append(summary)
    output = (
        ROOT_DIR / "results" / "upgrade_experiments"
        / "training_manifest_refined_full.json"
    )
    output.write_text(json.dumps(summaries, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
