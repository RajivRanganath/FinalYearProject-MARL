"""Strict split-locked evaluation of weight-updated Refined QMIX."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from deployment.evaluate_all import BatteryEntropy, EntropyThreshold, evaluate_episode
from deployment.split_lock import FinalSplitLock, atomic_dataframe_csv
from training.policy_runtime import ONNXPolicy, ObservationMaskPolicy
from training.refined_protocol import (
    PROMOTION_RULE,
    REFINED_FINAL_SEEDS,
    REFINED_SELECTION_SEEDS,
)
from training.training_profiles import get_training_profile


CANDIDATE = "Refined QMIX"
REFERENCE = "Extended QMIX"
PRIMARY_METRICS: Mapping[str, float] = {
    "raw_episode_reward": 1.0,
    "event_recall": 1.0,
    "total_energy_consumption": -1.0,
    "mean_aoi": -1.0,
    "redundant_sampling": -1.0,
    "network_coverage": 1.0,
}
METRICS = [
    *PRIMARY_METRICS,
    "samples_delivered",
    "samples_requested",
    "channel_blocks",
    "sample_action_fraction",
]
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 45_033
ANALYSIS_FILES = (
    Path("deployment/evaluate_training_v4.py"),
    Path("deployment/evaluate_all.py"),
    Path("environment/multi_agent_env.py"),
    Path("environment/pettingzoo_env.py"),
    Path("environment/single_agent_env.py"),
    Path("environment/energy_model.py"),
    Path("shared_config.py"),
    Path("training/policy_runtime.py"),
    Path("training/refined_protocol.py"),
    Path("training/training_profiles.py"),
    Path("deployment/split_lock.py"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True
    )
    return result.stdout.strip() or "unknown"


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT_DIR, capture_output=True, text=True
    )
    return bool(result.stdout.strip())


def _profile_policies(profile: str) -> List[Any]:
    settings = get_training_profile(profile)
    root = ROOT_DIR / "results" / "upgrade_models" / profile / "coordinated"
    policies: List[Any] = []
    for path in sorted(root.glob("qmix_seed*.onnx")):
        seed = int(path.stem.split("seed")[-1])
        if seed not in {101, 102, 103}:
            continue
        policy: Any = ONNXPolicy(path, include_agent_id=settings.include_agent_id)
        policy.train_seed = seed
        if settings.mask_neighbor_signal:
            policy = ObservationMaskPolicy(
                policy, [shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE]
            )
            policy.train_seed = seed
        policy.model_path = path
        policies.append(policy)
    if len(policies) != 3:
        raise FileNotFoundError(f"Expected three {profile} replicas under {root}")
    return policies


def _policy_sets() -> Dict[str, List[Any]]:
    return {
        REFERENCE: _profile_policies("extended"),
        CANDIDATE: _profile_policies("refined"),
        "Entropy Threshold": [EntropyThreshold()],
        "Battery + Entropy": [BatteryEntropy()],
    }


def _model_fingerprints() -> Dict[str, str]:
    files: List[Path] = []
    for profile in ("extended", "refined"):
        root = ROOT_DIR / "results" / "upgrade_models" / profile / "coordinated"
        files.extend(sorted(root.glob("qmix_seed*.onnx*")))
    return {str(path.relative_to(ROOT_DIR)): _sha256(path) for path in files}


def _analysis_fingerprints() -> Dict[str, str]:
    return {str(path): _sha256(ROOT_DIR / path) for path in ANALYSIS_FILES}


def _mean_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    half = float(stats.t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
    return mean, std, mean - half, mean + half


def _summaries(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_environment = raw.groupby(["policy", "environment_seed"], as_index=False)[METRICS].mean()
    summary_rows: List[Dict[str, Any]] = []
    for policy, frame in per_environment.groupby("policy"):
        row: Dict[str, Any] = {"policy": policy, "n_environment_seeds": len(frame)}
        for metric in METRICS:
            mean, std, low, high = _mean_ci(frame[metric].to_numpy(float))
            row.update({
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_ci95_low": low,
                f"{metric}_ci95_high": high,
            })
        summary_rows.append(row)
    seed_rows: List[Dict[str, Any]] = []
    learned = raw[raw.training_seed.notna()]
    for (policy, seed), frame in learned.groupby(["policy", "training_seed"]):
        row = {"policy": policy, "training_seed": int(seed)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(frame[metric].mean())
        seed_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(seed_rows), per_environment


def _two_way_bootstrap(
    raw: pd.DataFrame, comparator: str, metric: str, direction: float
) -> tuple[float, float]:
    candidate = raw[raw.policy == CANDIDATE].pivot(
        index="training_seed", columns="environment_seed", values=metric
    ).sort_index()
    environment_seeds = candidate.columns.to_numpy()
    training_seeds = candidate.index.to_numpy()
    baseline_rows = raw[raw.policy == comparator]
    learned_comparator = baseline_rows.training_seed.notna().all()
    if learned_comparator:
        baseline = baseline_rows.pivot(
            index="training_seed", columns="environment_seed", values=metric
        ).reindex(index=training_seeds, columns=environment_seeds)
    else:
        baseline = baseline_rows.set_index("environment_seed")[metric].reindex(environment_seeds)
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled_training = rng.choice(training_seeds, len(training_seeds), replace=True)
        sampled_environment = rng.choice(environment_seeds, len(environment_seeds), replace=True)
        differences: List[float] = []
        for training_seed in sampled_training:
            candidate_values = candidate.loc[training_seed, sampled_environment].to_numpy(float)
            if learned_comparator:
                baseline_values = baseline.loc[training_seed, sampled_environment].to_numpy(float)
            else:
                baseline_values = baseline.loc[sampled_environment].to_numpy(float)
            differences.extend((direction * (candidate_values - baseline_values)).tolist())
        draws[index] = np.mean(differences)
    return tuple(float(value) for value in np.percentile(draws, [2.5, 97.5]))


def _holm_adjust(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def _comparisons(raw: pd.DataFrame, per_environment: pd.DataFrame) -> pd.DataFrame:
    candidate = per_environment[per_environment.policy == CANDIDATE].set_index("environment_seed")
    rows: List[Dict[str, Any]] = []
    for comparator in (REFERENCE, "Entropy Threshold"):
        baseline = per_environment[per_environment.policy == comparator].set_index("environment_seed")
        for metric, direction in PRIMARY_METRICS.items():
            difference = direction * (
                candidate.loc[baseline.index, metric].to_numpy(float)
                - baseline[metric].to_numpy(float)
            )
            mean, std, low, high = _mean_ci(difference)
            test = stats.ttest_1samp(difference, 0.0)
            bootstrap_low, bootstrap_high = _two_way_bootstrap(
                raw, comparator, metric, direction
            )
            rows.append({
                "comparator": comparator,
                "metric": metric,
                "engineering_advantage_mean": mean,
                "environment_ci95_low": low,
                "environment_ci95_high": high,
                "two_way_bootstrap_ci95_low": bootstrap_low,
                "two_way_bootstrap_ci95_high": bootstrap_high,
                "statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "paired_cohens_dz": mean / std if std else np.nan,
            })
    frame = pd.DataFrame(rows)
    frame["p_value_holm"] = _holm_adjust(frame.p_value.to_numpy(float))
    return frame


def _selection_decision(
    raw: pd.DataFrame,
    comparisons: pd.DataFrame,
    model_hashes: Dict[str, str],
    analysis_hashes: Dict[str, str],
) -> Dict[str, Any]:
    candidate = raw[raw.policy == CANDIDATE]
    baseline = raw[raw.policy == REFERENCE]
    per_seed: Dict[str, float] = {}
    for seed in (101, 102, 103):
        per_seed[str(seed)] = float(
            candidate[candidate.training_seed == seed].raw_episode_reward.mean()
            - baseline[baseline.training_seed == seed].raw_episode_reward.mean()
        )
    rows = comparisons[comparisons.comparator == REFERENCE].set_index("metric")
    evidence = {
        "mean_reward_advantage": float(rows.loc["raw_episode_reward", "engineering_advantage_mean"]),
        "reward_environment_ci95_low": float(rows.loc["raw_episode_reward", "environment_ci95_low"]),
        "reward_two_way_bootstrap_ci95_low": float(rows.loc["raw_episode_reward", "two_way_bootstrap_ci95_low"]),
        "recall_advantage": float(rows.loc["event_recall", "engineering_advantage_mean"]),
        "energy_advantage": float(rows.loc["total_energy_consumption", "engineering_advantage_mean"]),
        "redundancy_advantage": float(rows.loc["redundant_sampling", "engineering_advantage_mean"]),
        "reward_advantage_by_training_seed": per_seed,
    }
    promoted = (
        evidence["reward_environment_ci95_low"] > 0.0
        and evidence["reward_two_way_bootstrap_ci95_low"] > 0.0
        and min(per_seed.values()) > 0.0
        and evidence["recall_advantage"] >= -0.002
        and evidence["energy_advantage"] >= -0.10
        and evidence["redundancy_advantage"] >= -1.0
    )
    return {
        "rule": PROMOTION_RULE,
        "selection_seeds": list(REFINED_SELECTION_SEEDS),
        "final_seeds_locked": list(REFINED_FINAL_SEEDS),
        "evidence": evidence,
        "promote_refined": bool(promoted),
        "model_fingerprints": model_hashes,
        "analysis_fingerprints": analysis_hashes,
    }


def _write_report(
    output: Path,
    split: str,
    summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    decision: Dict[str, Any],
) -> None:
    indexed = summary.set_index("policy")
    lines = [
        "# Refined QMIX Weight-Update Evaluation\n\n",
        f"Split: `{split}`. Refined QMIX was declared as a low-learning-rate warm-start; the historical run is superseded by the protocol invalidation record.\n\n",
        "| Policy | Reward | Recall | Energy | AoI | Redundancy |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for policy in (REFERENCE, CANDIDATE, "Entropy Threshold", "Battery + Entropy"):
        row = indexed.loc[policy]
        lines.append(
            f"| {policy} | {row.raw_episode_reward_mean:.2f} | {row.event_recall_mean:.3f} | "
            f"{row.total_energy_consumption_mean:.2f} | {row.mean_aoi_mean:.2f} | "
            f"{row.redundant_sampling_mean:.2f} |\n"
        )
    lines.extend(["\n## Training-replica reward means\n\n", "| Policy | Seed | Reward |\n", "|---|---:|---:|\n"])
    for _, row in seed_summary.sort_values(["policy", "training_seed"]).iterrows():
        lines.append(f"| {row.policy} | {int(row.training_seed)} | {row.raw_episode_reward_mean:.2f} |\n")
    lines.extend(["\n## Paired comparisons\n\n", "| Comparator | Metric | Advantage | Environment CI | Two-way bootstrap CI | Holm p |\n", "|---|---|---:|---:|---:|---:|\n"])
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row.comparator} | {row.metric} | {row.engineering_advantage_mean:.4f} | "
            f"[{row.environment_ci95_low:.4f}, {row.environment_ci95_high:.4f}] | "
            f"[{row.two_way_bootstrap_ci95_low:.4f}, {row.two_way_bootstrap_ci95_high:.4f}] | "
            f"{row.p_value_holm:.3e} |\n"
        )
    lines.extend(["\n## Promotion boundary\n\n", f"{decision['rule']}\n\n", f"Promote Refined QMIX: **{decision.get('promote_refined')}**.\n"])
    (output / "REPORT.md").write_text("".join(lines))


def evaluate(split: str) -> pd.DataFrame:
    if split not in {"selection", "final"}:
        raise ValueError("split must be selection or final")
    root = ROOT_DIR / "results" / "training_v4"
    output = root / split
    if (output / "raw.csv").exists():
        raise RuntimeError(f"The v4 {split} split has already been consumed")
    invalidation_path = root / "INVALIDATED.json"
    if split == "final" and invalidation_path.is_file():
        invalidation = json.loads(invalidation_path.read_text())
        if invalidation.get("status") == "INVALIDATED_PROTOCOL":
            raise RuntimeError(
                "The v4 protocol is invalidated; its untouched final split must not be consumed"
            )
    seeds = REFINED_SELECTION_SEEDS if split == "selection" else REFINED_FINAL_SEEDS
    model_hashes = _model_fingerprints()
    analysis_hashes = _analysis_fingerprints()
    decision_path = root / "selection" / "selection_decision.json"
    if split == "final":
        if (output / "CONSUMPTION_STARTED.json").exists():
            raise RuntimeError("The v4 final split has already been consumed")
        if not decision_path.exists():
            raise RuntimeError("Selection must be frozen before final evaluation")
        decision = json.loads(decision_path.read_text())
        if not decision.get("promote_refined"):
            raise RuntimeError("Refined weights failed the predeclared promotion rule")
        if decision["model_fingerprints"] != model_hashes:
            raise RuntimeError("Model fingerprints changed after selection")
        if decision["analysis_fingerprints"] != analysis_hashes:
            raise RuntimeError("Analysis fingerprints changed after selection")
        split_lock = FinalSplitLock(output, {
            "protocol": "training_v4",
            "split": "final",
            "environment_seeds": list(seeds),
            "model_fingerprints": model_hashes,
            "analysis_fingerprints": analysis_hashes,
        })
        split_lock.acquire()
    else:
        decision = {}
        split_lock = None

    rows: List[Dict[str, Any]] = []
    for label, policies in _policy_sets().items():
        print(f"EVALUATE {label} ({len(policies)} replica(s))")
        for policy in policies:
            for seed in seeds:
                metric, _, _, _ = evaluate_episode(
                    label, policy, "volatile", "coordinated", seed
                )
                rows.append(metric)
    raw = pd.DataFrame(rows)
    summary, seed_summary, per_environment = _summaries(raw)
    comparisons = _comparisons(raw, per_environment)
    if split == "selection":
        decision = _selection_decision(raw, comparisons, model_hashes, analysis_hashes)
    output.mkdir(parents=True, exist_ok=True)
    atomic_dataframe_csv(raw, output / "raw.csv")
    atomic_dataframe_csv(summary, output / "summary.csv")
    atomic_dataframe_csv(seed_summary, output / "per_training_seed.csv")
    atomic_dataframe_csv(per_environment, output / "per_environment_seed.csv")
    atomic_dataframe_csv(comparisons, output / "paired_comparisons.csv")
    if split == "selection":
        decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    manifest = {
        "split": split,
        "environment_seeds": list(seeds),
        "git_sha": _git_sha(),
        "git_worktree_dirty": _git_dirty(),
        "model_fingerprints": model_hashes,
        "analysis_fingerprints": analysis_hashes,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(output, split, summary, seed_summary, comparisons, decision)
    if split_lock is not None:
        split_lock.mark_complete({"raw_rows": len(raw)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["selection", "final"], default="selection")
    args = parser.parse_args()
    print(evaluate(args.split)[[
        "policy", "raw_episode_reward_mean", "event_recall_mean",
        "total_energy_consumption_mean", "mean_aoi_mean",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
