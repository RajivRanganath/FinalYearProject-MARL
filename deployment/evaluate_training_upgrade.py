"""Evaluate the upgraded QMIX profile on a fresh, predeclared holdout.

The original test seeds informed the published ablation analysis.  This script
therefore uses ``UPGRADE_TEST_SEEDS`` and compares the frozen published QMIX,
the validation-selected upgraded QMIX, and the strongest causal heuristic on
identical environment seeds.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from deployment.evaluate_all import EntropyThreshold, METRICS, evaluate_episode
from training.policy_runtime import ONNXPolicy, ObservationMaskPolicy


def _seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot infer training seed from {path}")
    return int(match.group(1))


def _model_policies(root: Path, improved: bool) -> List[Any]:
    paths = sorted(root.glob("qmix_seed*.onnx"))
    if len(paths) < 3:
        raise FileNotFoundError(f"Expected at least three QMIX replicas under {root}, found {len(paths)}")
    policies: List[Any] = []
    for path in paths:
        seed = _seed_from_path(path)
        if seed not in shared_config.TRAIN_SEEDS[:3]:
            continue
        policy = ONNXPolicy(path, include_agent_id=not improved)
        policy.train_seed = seed
        if improved:
            policy = ObservationMaskPolicy(
                policy,
                masked_indices=[shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE],
            )
            policy.train_seed = seed
        policies.append(policy)
    if len(policies) != 3:
        raise RuntimeError(f"Expected seeds {shared_config.TRAIN_SEEDS[:3]} under {root}")
    return policies


def _summarise(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_environment = raw.groupby(
        ["policy", "environment_seed"], as_index=False
    )[METRICS].mean()
    rows: List[Dict[str, Any]] = []
    for policy, frame in per_environment.groupby("policy"):
        row: Dict[str, Any] = {"policy": policy, "n_environment_seeds": len(frame)}
        for metric in METRICS:
            values = frame[metric].to_numpy(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            half = float(stats.t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
            row.update({
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_ci95_low": mean - half,
                f"{metric}_ci95_high": mean + half,
            })
        rows.append(row)
    return pd.DataFrame(rows), per_environment


def _paired(per_environment: pd.DataFrame) -> pd.DataFrame:
    directions = {
        "raw_episode_reward": 1.0,
        "event_recall": 1.0,
        "total_energy_consumption": -1.0,
        "mean_aoi": -1.0,
    }
    upgraded = per_environment[per_environment.policy == "Upgraded QMIX"].set_index(
        "environment_seed"
    )
    rows: List[Dict[str, Any]] = []
    for comparator in ("Published QMIX", "Entropy Threshold"):
        baseline = per_environment[per_environment.policy == comparator].set_index(
            "environment_seed"
        )
        common = upgraded.index.intersection(baseline.index)
        for metric, direction in directions.items():
            difference = direction * (
                upgraded.loc[common, metric].to_numpy(float)
                - baseline.loc[common, metric].to_numpy(float)
            )
            test = stats.ttest_1samp(difference, 0.0)
            std = float(difference.std(ddof=1))
            mean = float(difference.mean())
            half = float(stats.t.ppf(0.975, len(common) - 1) * std / math.sqrt(len(common)))
            rows.append({
                "comparator": comparator,
                "metric": metric,
                "n_pairs": len(common),
                "upgraded_engineering_advantage_mean": mean,
                "upgraded_engineering_advantage_std": std,
                "upgraded_engineering_advantage_ci95_low": mean - half,
                "upgraded_engineering_advantage_ci95_high": mean + half,
                "statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "paired_cohens_dz": float(mean / std) if std else np.nan,
                "test": "two-sided one-sample t-test on paired seed differences",
            })
    return pd.DataFrame(rows)


def evaluate_upgrade(
    scenario: str = "volatile",
    regime: str = "coordinated",
    seeds: Iterable[int] = shared_config.UPGRADE_TEST_SEEDS,
) -> pd.DataFrame:
    seeds = list(seeds)
    policies = {
        "Published QMIX": _model_policies(
            ROOT_DIR / "results" / "learned_models" / regime,
            improved=False,
        ),
        "Upgraded QMIX": _model_policies(
            ROOT_DIR / "results" / "upgrade_models" / "improved" / regime,
            improved=True,
        ),
        "Entropy Threshold": [EntropyThreshold()],
    }
    rows: List[Dict[str, Any]] = []
    for label, replicas in policies.items():
        print(f"EVALUATE {label}")
        for policy in replicas:
            for seed in seeds:
                metric, _, _, _ = evaluate_episode(label, policy, scenario, regime, seed)
                rows.append(metric)

    raw = pd.DataFrame(rows)
    summary, per_environment = _summarise(raw)
    comparisons = _paired(per_environment)
    output = ROOT_DIR / "results" / "training_upgrade" / regime
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "raw.csv", index=False)
    per_environment.to_csv(output / "per_environment_seed.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)

    lines = [
        "# Training Upgrade Evaluation\n\n",
        f"Scenario: `{scenario}`. Regime: `{regime}`. Fresh paired holdout: "
        f"`{seeds[0]}`--`{seeds[-1]}` ({len(seeds)} seeds).\n\n",
        "The upgraded profile was selected using validation seeds 201--210. "
        "The original 1001--1030 test set was not reused for this claim.\n\n",
        "| Policy | Reward mean [95% CI] | Recall mean [95% CI] | Energy mean [95% CI] | AoI mean [95% CI] |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for _, row in summary.sort_values("policy").iterrows():
        lines.append(
            f"| {row['policy']} | {row['raw_episode_reward_mean']:.2f} "
            f"[{row['raw_episode_reward_ci95_low']:.2f}, {row['raw_episode_reward_ci95_high']:.2f}] | "
            f"{row['event_recall_mean']:.3f} "
            f"[{row['event_recall_ci95_low']:.3f}, {row['event_recall_ci95_high']:.3f}] | "
            f"{row['total_energy_consumption_mean']:.2f} "
            f"[{row['total_energy_consumption_ci95_low']:.2f}, {row['total_energy_consumption_ci95_high']:.2f}] | "
            f"{row['mean_aoi_mean']:.2f} "
            f"[{row['mean_aoi_ci95_low']:.2f}, {row['mean_aoi_ci95_high']:.2f}] |\n"
        )
    lines.extend([
        "\nPositive paired advantage means the upgraded policy is better in the engineering direction; "
        "energy and AoI signs are reversed because lower is better. Tests are exploratory and unadjusted for multiple comparisons.\n\n",
        "| Comparator | Metric | Upgraded advantage [95% CI] | t | p | Cohen dz |\n",
        "|---|---|---:|---:|---:|---:|\n",
    ])
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row['comparator']} | {row['metric']} | "
            f"{row['upgraded_engineering_advantage_mean']:.4f} "
            f"[{row['upgraded_engineering_advantage_ci95_low']:.4f}, "
            f"{row['upgraded_engineering_advantage_ci95_high']:.4f}] | "
            f"{row['statistic']:.3f} | {row['p_value']:.3e} | "
            f"{row['paired_cohens_dz']:.3f} |\n"
        )
    (output / "REPORT.md").write_text("".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="volatile", choices=shared_config.SCENARIOS)
    parser.add_argument("--regime", default="coordinated", choices=shared_config.REGIMES)
    parser.add_argument("--n-seeds", type=int, default=30)
    args = parser.parse_args()
    summary = evaluate_upgrade(
        scenario=args.scenario,
        regime=args.regime,
        seeds=shared_config.UPGRADE_TEST_SEEDS[: args.n_seeds],
    )
    print(summary[[
        "policy",
        "raw_episode_reward_mean",
        "event_recall_mean",
        "total_energy_consumption_mean",
        "mean_aoi_mean",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
