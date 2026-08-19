"""Split-locked screening of deployable ensembles built from Extended QMIX.

This iteration does not alter the environment, reward, or trained weights.  It
tests whether majority or unanimous voting across the three independently
trained Extended QMIX replicas improves deployment performance.  Seeds
231--250 are development-only; seeds 4001--4030 may be evaluated once after a
candidate passes the predeclared promotion rule and all fingerprints freeze.
"""

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
from deployment.evaluate_all import (
    BatteryEntropy,
    EntropyThreshold,
    GreedyHeuristic,
    evaluate_episode,
)
from training.policy_runtime import ONNXPolicy, ObservationMaskPolicy


V3_SELECTION_SEEDS = tuple(range(231, 251))
V3_FINAL_SEEDS = tuple(range(4001, 4031))
CANDIDATES: Mapping[str, int] = {
    "QMIX Majority Ensemble": 2,
    "QMIX Unanimous Ensemble": 3,
}
REFERENCE = "Extended QMIX Replica Mean"
PRIMARY_METRICS: Mapping[str, float] = {
    "raw_episode_reward": 1.0,
    "event_recall": 1.0,
    "total_energy_consumption": -1.0,
    "mean_aoi": -1.0,
    "redundant_sampling": -1.0,
    "network_coverage": 1.0,
}
PROMOTION_LIMITS = {
    "minimum_reward_advantage": 0.0,
    "minimum_reward_ci95_low": 0.0,
    "maximum_reward_p_holm": 0.05,
    "minimum_recall_advantage": -0.005,
    "minimum_energy_advantage": -0.25,
    "minimum_redundancy_advantage": -2.0,
}
ANALYSIS_FILES = (
    Path("deployment/evaluate_training_v3.py"),
    Path("deployment/evaluate_all.py"),
    Path("environment/multi_agent_env.py"),
    Path("environment/pettingzoo_env.py"),
    Path("environment/single_agent_env.py"),
    Path("environment/energy_model.py"),
    Path("shared_config.py"),
    Path("training/policy_runtime.py"),
)
METRICS = [
    "raw_episode_reward",
    "event_recall",
    "total_energy_consumption",
    "mean_aoi",
    "redundant_sampling",
    "network_coverage",
    "samples_delivered",
    "samples_requested",
    "channel_blocks",
    "sample_action_fraction",
]


class VotingEnsemblePolicy:
    """Scale-invariant vote over recurrent per-replica greedy actions."""

    train_seed = None

    def __init__(self, paths: Sequence[Path], minimum_sample_votes: int):
        if len(paths) != 3 or minimum_sample_votes not in {2, 3}:
            raise ValueError("Voting ensemble requires three replicas and a 2- or 3-vote rule")
        self.policies = [ONNXPolicy(path, include_agent_id=False) for path in paths]
        self.minimum_sample_votes = minimum_sample_votes

    def reset(self) -> None:
        for policy in self.policies:
            policy.reset()

    def select_action(self, agent_id: str, obs: np.ndarray, info: dict) -> int:
        decision_obs = np.asarray(obs, dtype=np.float32).copy()
        decision_obs[shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE] = 0.0
        mask = np.asarray(info.get("action_mask", [1, 1]))
        if not mask[shared_config.ACTION_SAMPLE]:
            # Recurrent state must still advance for every replica.
            for policy in self.policies:
                policy.q_values(agent_id, decision_obs)
            return shared_config.ACTION_SLEEP
        votes = 0
        for policy in self.policies:
            q_values = policy.q_values(agent_id, decision_obs).copy()
            q_values[mask == 0] = -np.inf
            votes += int(np.argmax(q_values) == shared_config.ACTION_SAMPLE)
        return (
            shared_config.ACTION_SAMPLE
            if votes >= self.minimum_sample_votes
            else shared_config.ACTION_SLEEP
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


def _model_paths() -> List[Path]:
    root = ROOT_DIR / "results" / "upgrade_models" / "extended" / "coordinated"
    paths = sorted(root.glob("qmix_seed*.onnx"))
    if len(paths) != 3:
        raise FileNotFoundError(f"Expected three Extended QMIX ONNX models, found {len(paths)}")
    return paths


def _fingerprints(paths: Sequence[Path]) -> tuple[Dict[str, str], Dict[str, str]]:
    model_files: List[Path] = []
    for path in paths:
        model_files.extend([path, path.with_name(path.name + ".data")])
    models = {str(path.relative_to(ROOT_DIR)): _sha256(path) for path in model_files}
    analysis = {str(path): _sha256(ROOT_DIR / path) for path in ANALYSIS_FILES}
    return models, analysis


def _replica_policies(paths: Sequence[Path]) -> List[Any]:
    policies: List[Any] = []
    for path in paths:
        policy: Any = ONNXPolicy(path, include_agent_id=False)
        policy.train_seed = int(path.stem.split("seed")[-1])
        policy = ObservationMaskPolicy(
            policy, [shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE]
        )
        policy.train_seed = int(path.stem.split("seed")[-1])
        policies.append(policy)
    return policies


def _policy_sets(paths: Sequence[Path], candidate: str | None = None) -> Dict[str, List[Any]]:
    policies: Dict[str, List[Any]] = {
        REFERENCE: _replica_policies(paths),
        "Entropy Threshold": [EntropyThreshold()],
        "Battery + Entropy": [BatteryEntropy()],
        "Greedy": [GreedyHeuristic()],
    }
    choices = CANDIDATES if candidate is None else {candidate: CANDIDATES[candidate]}
    for label, votes in choices.items():
        policies[label] = [VotingEnsemblePolicy(paths, votes)]
    return policies


def _mean_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    half = float(stats.t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
    return mean, std, mean - half, mean + half


def _summaries(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_environment = raw.groupby(["policy", "environment_seed"], as_index=False)[METRICS].mean()
    rows: List[Dict[str, Any]] = []
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
        rows.append(row)
    return pd.DataFrame(rows), per_environment


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def _candidate_comparisons(per_environment: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    available = set(per_environment.policy)
    candidates = [candidate for candidate in CANDIDATES if candidate in available]
    for candidate in candidates:
        frame = per_environment[per_environment.policy == candidate].set_index("environment_seed")
        for comparator in (REFERENCE, "Entropy Threshold"):
            reference = per_environment[per_environment.policy == comparator].set_index(
                "environment_seed"
            )
            for metric, direction in PRIMARY_METRICS.items():
                difference = direction * (
                    frame.loc[reference.index, metric].to_numpy(float)
                    - reference[metric].to_numpy(float)
                )
                mean, std, low, high = _mean_ci(difference)
                test = stats.ttest_1samp(difference, 0.0)
                rows.append({
                    "candidate": candidate,
                    "comparator": comparator,
                    "metric": metric,
                    "engineering_advantage_mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "statistic": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "paired_cohens_dz": mean / std if std else np.nan,
                })
    frame = pd.DataFrame(rows)
    frame["p_value_holm"] = _holm_adjust(frame["p_value"].to_numpy(float))
    reward_rows = (frame.metric == "raw_episode_reward") & (frame.comparator == REFERENCE)
    reward_p = frame.loc[reward_rows, "p_value"].to_numpy(float)
    frame["reward_p_holm"] = np.nan
    frame.loc[reward_rows, "reward_p_holm"] = _holm_adjust(reward_p)
    return frame


def _promotion_decision(comparisons: pd.DataFrame) -> Dict[str, Any]:
    evidence: Dict[str, Dict[str, float | bool]] = {}
    eligible: List[str] = []
    for candidate in CANDIDATES:
        rows = comparisons[
            (comparisons.candidate == candidate) & (comparisons.comparator == REFERENCE)
        ].set_index("metric")
        item: Dict[str, float | bool] = {
            "reward_advantage": float(rows.loc["raw_episode_reward", "engineering_advantage_mean"]),
            "reward_ci95_low": float(rows.loc["raw_episode_reward", "ci95_low"]),
            "reward_p_holm": float(rows.loc["raw_episode_reward", "reward_p_holm"]),
            "recall_advantage": float(rows.loc["event_recall", "engineering_advantage_mean"]),
            "energy_advantage": float(rows.loc["total_energy_consumption", "engineering_advantage_mean"]),
            "redundancy_advantage": float(rows.loc["redundant_sampling", "engineering_advantage_mean"]),
        }
        passed = (
            item["reward_advantage"] > PROMOTION_LIMITS["minimum_reward_advantage"]
            and item["reward_ci95_low"] > PROMOTION_LIMITS["minimum_reward_ci95_low"]
            and item["reward_p_holm"] <= PROMOTION_LIMITS["maximum_reward_p_holm"]
            and item["recall_advantage"] >= PROMOTION_LIMITS["minimum_recall_advantage"]
            and item["energy_advantage"] >= PROMOTION_LIMITS["minimum_energy_advantage"]
            and item["redundancy_advantage"] >= PROMOTION_LIMITS["minimum_redundancy_advantage"]
        )
        item["passed"] = passed
        evidence[candidate] = item
        if passed:
            eligible.append(candidate)
    promoted = max(
        eligible,
        key=lambda name: float(evidence[name]["reward_advantage"]),
        default=None,
    )
    return {
        "rule": (
            "Promote the highest-reward candidate only when its paired reward 95% CI versus "
            "the three-replica Extended mean is above zero, the candidate-screening Holm reward "
            "p-value is at most 0.05, recall advantage is at least -0.005, "
            "energy advantage at least -0.25, and redundancy advantage at least -2.0."
        ),
        "limits": PROMOTION_LIMITS,
        "candidate_evidence": evidence,
        "promoted_candidate": promoted,
    }


def _write_report(
    output: Path,
    split: str,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    decision: Dict[str, Any],
) -> None:
    indexed = summary.set_index("policy")
    lines = [
        "# QMIX Deployment Ensemble Evaluation\n\n",
        f"Split: `{split}`. This experiment changes deployment aggregation only; trained weights, environment, and reward are unchanged.\n\n",
        "| Policy | Reward | Recall | Energy | AoI | Redundancy | Sample fraction |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    order = ["Entropy Threshold", "Battery + Entropy", "Greedy", REFERENCE, *CANDIDATES]
    for policy in order:
        if policy not in indexed.index:
            continue
        row = indexed.loc[policy]
        lines.append(
            f"| {policy} | {row.raw_episode_reward_mean:.2f} | {row.event_recall_mean:.3f} | "
            f"{row.total_energy_consumption_mean:.2f} | {row.mean_aoi_mean:.2f} | "
            f"{row.redundant_sampling_mean:.2f} | {row.sample_action_fraction_mean:.3f} |\n"
        )
    lines.extend([
        "\n## Paired candidate comparisons\n\n",
        "Positive values favor the candidate in the engineering direction. Holm adjustment covers the displayed comparison family.\n\n",
        "| Candidate | Comparator | Metric | Advantage | 95% CI | Holm p |\n",
        "|---|---|---|---:|---:|---:|\n",
    ])
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row.candidate} | {row.comparator} | {row.metric} | "
            f"{row.engineering_advantage_mean:.4f} | "
            f"[{row.ci95_low:.4f}, {row.ci95_high:.4f}] | {row.p_value_holm:.3e} |\n"
        )
    lines.extend([
        "\n## Decision\n\n",
        f"{decision['rule']}\n\n",
        f"Promoted candidate: `{decision['promoted_candidate']}`.\n",
    ])
    (output / "REPORT.md").write_text("".join(lines))


def evaluate(split: str) -> pd.DataFrame:
    if split not in {"selection", "final"}:
        raise ValueError("split must be selection or final")
    root = ROOT_DIR / "results" / "training_v3"
    output = root / split
    if (output / "raw.csv").exists():
        raise RuntimeError(f"The v3 {split} split has already been consumed")
    paths = _model_paths()
    model_hashes, analysis_hashes = _fingerprints(paths)
    decision_path = root / "selection" / "selection_decision.json"
    seeds = V3_SELECTION_SEEDS if split == "selection" else V3_FINAL_SEEDS
    candidate: str | None = None
    if split == "final":
        if not decision_path.exists():
            raise RuntimeError("Selection must be completed before final evaluation")
        frozen = json.loads(decision_path.read_text())
        candidate = frozen.get("promoted_candidate")
        if candidate not in CANDIDATES:
            raise RuntimeError("No deployment candidate passed the predeclared promotion rule")
        if frozen["model_fingerprints"] != model_hashes:
            raise RuntimeError("Model fingerprints changed after selection")
        if frozen["analysis_fingerprints"] != analysis_hashes:
            raise RuntimeError("Analysis fingerprints changed after selection")

    rows: List[Dict[str, Any]] = []
    for label, policies in _policy_sets(paths, candidate).items():
        print(f"EVALUATE {label} ({len(policies)} replica(s))")
        for policy in policies:
            for seed in seeds:
                metric, _, _, _ = evaluate_episode(
                    label, policy, "volatile", "coordinated", seed
                )
                rows.append(metric)
    raw = pd.DataFrame(rows)
    summary, per_environment = _summaries(raw)
    comparisons = _candidate_comparisons(per_environment)
    if split == "selection":
        decision = _promotion_decision(comparisons)
        decision.update({
            "selection_seeds": list(seeds),
            "final_seeds_locked": list(V3_FINAL_SEEDS),
            "model_fingerprints": model_hashes,
            "analysis_fingerprints": analysis_hashes,
        })
    else:
        decision = json.loads(decision_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "raw.csv", index=False)
    per_environment.to_csv(output / "per_environment_seed.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    comparisons.to_csv(output / "candidate_comparisons.csv", index=False)
    if split == "selection":
        (output / "selection_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    manifest = {
        "split": split,
        "environment_seeds": list(seeds),
        "candidate": candidate,
        "git_sha": _git_sha(),
        "git_worktree_dirty": _git_dirty(),
        "model_fingerprints": model_hashes,
        "analysis_fingerprints": analysis_hashes,
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(output, split, summary, comparisons, decision)
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
