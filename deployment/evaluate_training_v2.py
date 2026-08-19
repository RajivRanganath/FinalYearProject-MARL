"""Critical, split-locked evaluation of the second QMIX training iteration.

The 211--230 selection split decides whether the extended profile is promoted.
Only a frozen, promoted model set may be evaluated once on 3001--3030.  The
report keeps environment-seed uncertainty and training-seed variability
visible instead of silently treating 90 crossed evaluations as independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from deployment.evaluate_all import (
    AlwaysSample,
    AlwaysSleep,
    BatteryEntropy,
    EntropyThreshold,
    FixedInterval,
    GreedyHeuristic,
    METRICS,
    RandomFeasible,
    evaluate_episode,
)
from training.policy_runtime import ONNXPolicy, ObservationMaskPolicy
from training.training_profiles import get_training_profile


DIAGNOSTIC_METRICS = [
    *METRICS,
    "samples_delivered",
    "samples_requested",
    "channel_blocks",
]
PRIMARY_COMPARISON_METRICS: Mapping[str, float] = {
    "raw_episode_reward": 1.0,
    "event_recall": 1.0,
    "total_energy_consumption": -1.0,
    "mean_aoi": -1.0,
    "redundant_sampling": -1.0,
    "network_coverage": 1.0,
}
PRIMARY_COMPARATORS = ("Improved QMIX", "Entropy Threshold")
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 25_033
ANALYSIS_FILES = (
    Path("deployment/evaluate_training_v2.py"),
    Path("deployment/evaluate_all.py"),
    Path("environment/multi_agent_env.py"),
    Path("environment/pettingzoo_env.py"),
    Path("environment/single_agent_env.py"),
    Path("environment/energy_model.py"),
    Path("shared_config.py"),
    Path("training/policy_runtime.py"),
    Path("training/training_profiles.py"),
)


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot infer training seed from {path}")
    return int(match.group(1))


def _profile_policies(profile: str, regime: str) -> List[Any]:
    settings = get_training_profile(profile)
    root = ROOT_DIR / "results" / "upgrade_models" / profile / regime
    policies: List[Any] = []
    for path in sorted(root.glob("qmix_seed*.onnx")):
        seed = _seed_from_path(path)
        if seed not in shared_config.TRAIN_SEEDS[:3]:
            continue
        policy: Any = ONNXPolicy(path, include_agent_id=settings.include_agent_id)
        policy.train_seed = seed
        if settings.mask_neighbor_signal:
            policy = ObservationMaskPolicy(
                policy,
                [shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE],
            )
            policy.train_seed = seed
        policy.model_path = path
        policies.append(policy)
    if len(policies) != 3:
        raise FileNotFoundError(
            f"Expected three {profile} QMIX replicas under {root}, found {len(policies)}"
        )
    return policies


def _published_policies(regime: str) -> Dict[str, List[Any]]:
    root = ROOT_DIR / "results" / "learned_models" / regime
    output: Dict[str, List[Any]] = {}
    for algorithm in ("iql", "vdn", "qmix"):
        replicas: List[Any] = []
        for path in sorted(root.glob(f"{algorithm}_seed*.onnx")):
            seed = _seed_from_path(path)
            if seed not in shared_config.TRAIN_SEEDS[:3]:
                continue
            policy = ONNXPolicy(path, include_agent_id=True)
            policy.train_seed = seed
            policy.model_path = path
            replicas.append(policy)
        if len(replicas) != 3:
            raise FileNotFoundError(
                f"Expected three published {algorithm.upper()} replicas under {root}"
            )
        output[f"Published {algorithm.upper()}"] = replicas
    return output


def _policy_sets(regime: str) -> Dict[str, List[Any]]:
    policies: Dict[str, List[Any]] = {
        "Always Sleep": [AlwaysSleep()],
        "Always Sample": [AlwaysSample()],
        "Random Feasible": [RandomFeasible()],
        "Fixed Interval": [FixedInterval()],
        "Entropy Threshold": [EntropyThreshold()],
        "Battery + Entropy": [BatteryEntropy()],
        "Greedy": [GreedyHeuristic()],
    }
    policies.update(_published_policies(regime))
    policies["Improved QMIX"] = _profile_policies("improved", regime)
    policies["Extended QMIX"] = _profile_policies("extended", regime)
    return policies


def _mean_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if len(values) > 1:
        half = float(stats.t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
    else:
        half = 0.0
    return mean, std, mean - half, mean + half


def _summaries(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_environment = raw.groupby(
        ["policy", "environment_seed"], as_index=False
    )[DIAGNOSTIC_METRICS].mean()
    rows: List[Dict[str, Any]] = []
    for policy, frame in per_environment.groupby("policy"):
        row: Dict[str, Any] = {"policy": policy, "n_environment_seeds": len(frame)}
        for metric in DIAGNOSTIC_METRICS:
            mean, std, low, high = _mean_ci(frame[metric].to_numpy(float))
            row.update({
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_ci95_low": low,
                f"{metric}_ci95_high": high,
            })
        rows.append(row)

    learned = raw[raw["training_seed"].notna()].copy()
    seed_rows: List[Dict[str, Any]] = []
    for (policy, train_seed), frame in learned.groupby(["policy", "training_seed"]):
        row = {
            "policy": policy,
            "training_seed": int(train_seed),
            "n_environment_seeds": frame["environment_seed"].nunique(),
        }
        for metric in DIAGNOSTIC_METRICS:
            mean, std, low, high = _mean_ci(frame[metric].to_numpy(float))
            row.update({
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_ci95_low": low,
                f"{metric}_ci95_high": high,
            })
        seed_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(seed_rows), per_environment


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm family-wise adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def _two_way_bootstrap_difference(
    raw: pd.DataFrame,
    comparator: str,
    metric: str,
    direction: float,
) -> tuple[float, float]:
    candidate = raw[raw.policy == "Extended QMIX"].pivot(
        index="training_seed", columns="environment_seed", values=metric
    ).sort_index()
    baseline_rows = raw[raw.policy == comparator]
    environment_seeds = candidate.columns.to_numpy()
    training_seeds = candidate.index.to_numpy()
    if baseline_rows["training_seed"].notna().all():
        baseline = baseline_rows.pivot(
            index="training_seed", columns="environment_seed", values=metric
        ).reindex(index=training_seeds, columns=environment_seeds)
        paired_training = True
    else:
        baseline = baseline_rows.set_index("environment_seed")[metric].reindex(environment_seeds)
        paired_training = False

    rng = np.random.RandomState(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled_training = rng.choice(training_seeds, size=len(training_seeds), replace=True)
        sampled_environment = rng.choice(environment_seeds, size=len(environment_seeds), replace=True)
        differences: List[float] = []
        for train_seed in sampled_training:
            candidate_values = candidate.loc[train_seed, sampled_environment].to_numpy(float)
            if paired_training:
                baseline_values = baseline.loc[train_seed, sampled_environment].to_numpy(float)
            else:
                baseline_values = baseline.loc[sampled_environment].to_numpy(float)
            differences.extend((direction * (candidate_values - baseline_values)).tolist())
        draws[index] = float(np.mean(differences))
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def _comparisons(raw: pd.DataFrame, per_environment: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    candidate = per_environment[per_environment.policy == "Extended QMIX"].set_index(
        "environment_seed"
    )
    for comparator in PRIMARY_COMPARATORS:
        baseline = per_environment[per_environment.policy == comparator].set_index(
            "environment_seed"
        )
        common = candidate.index.intersection(baseline.index)
        for metric, direction in PRIMARY_COMPARISON_METRICS.items():
            difference = direction * (
                candidate.loc[common, metric].to_numpy(float)
                - baseline.loc[common, metric].to_numpy(float)
            )
            mean, std, low, high = _mean_ci(difference)
            test = stats.ttest_1samp(difference, 0.0)
            bootstrap_low, bootstrap_high = _two_way_bootstrap_difference(
                raw, comparator, metric, direction
            )
            rows.append({
                "comparator": comparator,
                "metric": metric,
                "n_environment_pairs": len(common),
                "extended_engineering_advantage_mean": mean,
                "environment_seed_ci95_low": low,
                "environment_seed_ci95_high": high,
                "two_way_bootstrap_ci95_low": bootstrap_low,
                "two_way_bootstrap_ci95_high": bootstrap_high,
                "statistic": float(test.statistic),
                "p_value_unadjusted": float(test.pvalue),
                "paired_cohens_dz": float(mean / std) if std else np.nan,
                "test": "two-sided one-sample t-test on paired environment-seed differences",
                "assumption": (
                    "t-test treats environment seeds as sampling units after averaging replicas; "
                    "the two-way bootstrap also resamples training seeds"
                ),
            })
    frame = pd.DataFrame(rows)
    frame["p_value_holm"] = _holm_adjust(frame["p_value_unadjusted"].to_numpy(float))
    frame["holm_supported_0.05"] = frame["p_value_holm"] < 0.05
    return frame


def _model_fingerprints(regime: str) -> Dict[str, str]:
    paths: List[Path] = []
    for profile in ("improved", "extended"):
        paths.extend(
            sorted((ROOT_DIR / "results" / "upgrade_models" / profile / regime).glob("qmix_seed*.onnx*"))
        )
    published_root = ROOT_DIR / "results" / "learned_models" / regime
    for algorithm in ("iql", "vdn", "qmix"):
        paths.extend(sorted(published_root.glob(f"{algorithm}_seed*.onnx*")))
    return {
        str(path.relative_to(ROOT_DIR)): _sha256(path)
        for path in sorted(set(paths))
    }


def _analysis_fingerprints() -> Dict[str, str]:
    return {str(path): _sha256(ROOT_DIR / path) for path in ANALYSIS_FILES}


def _selection_decision(
    raw: pd.DataFrame,
    model_fingerprints: Dict[str, str],
    analysis_fingerprints: Dict[str, str],
    scenario: str,
    regime: str,
) -> Dict[str, Any]:
    candidate = raw[raw.policy == "Extended QMIX"]
    baseline = raw[raw.policy == "Improved QMIX"]
    per_seed: Dict[str, float] = {}
    for seed in shared_config.TRAIN_SEEDS[:3]:
        extended_mean = candidate[candidate.training_seed == seed]["raw_episode_reward"].mean()
        improved_mean = baseline[baseline.training_seed == seed]["raw_episode_reward"].mean()
        per_seed[str(seed)] = float(extended_mean - improved_mean)
    mean_advantage = float(np.mean(list(per_seed.values())))
    return {
        "selection_rule": (
            "Promote only if mean reward improves and every one of the three paired "
            "training replicas improves on predeclared selection seeds 211--230."
        ),
        "selection_seeds": shared_config.V2_SELECTION_SEEDS,
        "scenario": scenario,
        "regime": regime,
        "reward_advantage_by_training_seed": per_seed,
        "mean_reward_advantage": mean_advantage,
        "promote_extended": bool(mean_advantage > 0 and min(per_seed.values()) > 0),
        "model_fingerprints": model_fingerprints,
        "analysis_fingerprints": analysis_fingerprints,
    }


def _format_interval(row: pd.Series, metric: str, digits: int = 3) -> str:
    return (
        f"{row[f'{metric}_mean']:.{digits}f} "
        f"[{row[f'{metric}_ci95_low']:.{digits}f}, {row[f'{metric}_ci95_high']:.{digits}f}]"
    )


def _write_report(
    output: Path,
    split: str,
    scenario: str,
    regime: str,
    seeds: Sequence[int],
    summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    decision: Dict[str, Any],
) -> None:
    key_order = [
        "Entropy Threshold",
        "Battery + Entropy",
        "Published QMIX",
        "Improved QMIX",
        "Extended QMIX",
    ]
    indexed = summary.set_index("policy")
    lines = [
        "# Extended QMIX Critical Evaluation\n\n",
        f"Split: `{split}`. Scenario: `{scenario}`. Regime: `{regime}`. ",
        f"Paired environment seeds: `{seeds[0]}`--`{seeds[-1]}` ({len(seeds)} seeds).\n\n",
        "Extended QMIX changes only training duration relative to Improved QMIX. "
        "The reward and environment are unchanged. Learned-policy means average three "
        "training replicas per environment seed before the displayed t intervals.\n\n",
        "| Policy | Reward mean [95% CI] | Recall mean [95% CI] | Energy mean [95% CI] | AoI mean [95% CI] | Redundant samples | Channel blocks |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for name in key_order:
        row = indexed.loc[name]
        lines.append(
            f"| {name} | {_format_interval(row, 'raw_episode_reward', 2)} | "
            f"{_format_interval(row, 'event_recall')} | "
            f"{_format_interval(row, 'total_energy_consumption', 2)} | "
            f"{_format_interval(row, 'mean_aoi', 2)} | "
            f"{row['redundant_sampling_mean']:.2f} | {row['channel_blocks_mean']:.2f} |\n"
        )

    lines.extend([
        "\n## Training-seed variability\n\n",
        "| Policy | Training seed | Reward mean | Recall mean | Energy mean | AoI mean |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ])
    focus = seed_summary[seed_summary.policy.isin(["Improved QMIX", "Extended QMIX"])]
    for _, row in focus.sort_values(["policy", "training_seed"]).iterrows():
        lines.append(
            f"| {row['policy']} | {int(row['training_seed'])} | "
            f"{row['raw_episode_reward_mean']:.2f} | {row['event_recall_mean']:.3f} | "
            f"{row['total_energy_consumption_mean']:.2f} | {row['mean_aoi_mean']:.2f} |\n"
        )

    lines.extend([
        "\n## Paired comparisons\n\n",
        "Positive advantage means Extended QMIX is better in the engineering direction. "
        "The environment-seed interval averages replicas; the two-way bootstrap resamples both "
        "training and environment seeds. Holm adjustment covers all comparisons below.\n\n",
        "| Comparator | Metric | Advantage | Environment CI | Two-way bootstrap CI | Holm p |\n",
        "|---|---|---:|---:|---:|---:|\n",
    ])
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row['comparator']} | {row['metric']} | "
            f"{row['extended_engineering_advantage_mean']:.4f} | "
            f"[{row['environment_seed_ci95_low']:.4f}, {row['environment_seed_ci95_high']:.4f}] | "
            f"[{row['two_way_bootstrap_ci95_low']:.4f}, {row['two_way_bootstrap_ci95_high']:.4f}] | "
            f"{row['p_value_holm']:.3e} |\n"
        )

    if split == "selection":
        lines.extend([
            "\n## Promotion decision\n\n",
            f"Rule: {decision['selection_rule']}\n\n",
            f"Reward advantages by training seed: `{decision['reward_advantage_by_training_seed']}`. ",
            f"Promote Extended QMIX: **{decision['promote_extended']}**.\n",
        ])
    else:
        lines.extend([
            "\n## Interpretation boundary\n\n",
            "This final split was evaluated after the profile and model hashes were frozen. "
            "No further tuning on these seeds is valid. Three training seeds remain a small "
            "sample for algorithm-level generalisation, so training-seed conclusions are cautious.\n",
        ])
    (output / "REPORT.md").write_text("".join(lines))


def evaluate_v2(
    split: str = "selection",
    scenario: str = "volatile",
    regime: str = "coordinated",
) -> pd.DataFrame:
    if split not in {"selection", "final"}:
        raise ValueError("split must be selection or final")
    seeds = (
        shared_config.V2_SELECTION_SEEDS
        if split == "selection"
        else shared_config.V2_TEST_SEEDS
    )
    root = ROOT_DIR / "results" / "training_v2"
    output = root / split
    if (output / "raw.csv").exists():
        raise RuntimeError(f"The v2 {split} split has already been consumed")
    fingerprints = _model_fingerprints(regime)
    analysis_fingerprints = _analysis_fingerprints()
    decision_path = root / "selection" / "selection_decision.json"
    if split == "final":
        if not decision_path.exists():
            raise RuntimeError("Run and freeze the selection evaluation before final evaluation")
        decision = json.loads(decision_path.read_text())
        if not decision.get("promote_extended"):
            raise RuntimeError("Extended profile did not satisfy the predeclared promotion rule")
        if decision.get("scenario") != scenario or decision.get("regime") != regime:
            raise RuntimeError("Scenario/regime changed after selection")
        if decision.get("model_fingerprints") != fingerprints:
            raise RuntimeError("Frozen model fingerprints changed after selection")
        if decision.get("analysis_fingerprints") != analysis_fingerprints:
            raise RuntimeError("Frozen analysis-code fingerprints changed after selection")
    else:
        decision = {}

    rows: List[Dict[str, Any]] = []
    for label, replicas in _policy_sets(regime).items():
        print(f"EVALUATE {label} ({len(replicas)} replica(s))")
        for policy in replicas:
            for seed in seeds:
                metric, _, _, _ = evaluate_episode(label, policy, scenario, regime, seed)
                rows.append(metric)
    raw = pd.DataFrame(rows)
    summary, seed_summary, per_environment = _summaries(raw)
    comparisons = _comparisons(raw, per_environment)
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "raw.csv", index=False)
    per_environment.to_csv(output / "per_environment_seed.csv", index=False)
    seed_summary.to_csv(output / "per_training_seed.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)

    if split == "selection":
        decision = _selection_decision(
            raw, fingerprints, analysis_fingerprints, scenario, regime
        )
        (output / "selection_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    manifest = {
        "split": split,
        "scenario": scenario,
        "regime": regime,
        "environment_seeds": seeds,
        "git_sha": _git_sha(),
        "git_worktree_dirty": _git_dirty(),
        "model_fingerprints": fingerprints,
        "analysis_fingerprints": analysis_fingerprints,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_comparators": PRIMARY_COMPARATORS,
        "primary_metrics": list(PRIMARY_COMPARISON_METRICS),
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(
        output, split, scenario, regime, seeds, summary, seed_summary, comparisons, decision
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["selection", "final"], default="selection")
    parser.add_argument("--scenario", choices=shared_config.SCENARIOS, default="volatile")
    parser.add_argument("--regime", choices=shared_config.REGIMES, default="coordinated")
    args = parser.parse_args()
    result = evaluate_v2(args.split, args.scenario, args.regime)
    print(result[result.policy.isin([
        "Entropy Threshold", "Improved QMIX", "Extended QMIX"
    ])][[
        "policy",
        "raw_episode_reward_mean",
        "event_recall_mean",
        "total_energy_consumption_mean",
        "mean_aoi_mean",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
