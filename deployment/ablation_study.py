"""Retrain-and-evaluate QMIX ablations on a common coordinated test task.

Every variant removes one named component during training. Evaluation always
uses the full coordinated environment and objective on the locked 30 seeds, so
reward values remain comparable. This deliberately does not evaluate a frozen
full policy under a modified reward.
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
from deployment.evaluate_all import METRICS, evaluate_episode
from training.policy_runtime import ONNXPolicy
from training.train_all import train_suite


ABLATIONS = [
    "no_neighbor_signal",
    "no_redundancy",
    "no_aoi",
    "no_energy",
    "no_agent_id",
    "no_coordination_constraint",
]

DISPLAY_NAMES = {
    "full": "Full QMIX",
    "no_neighbor_signal": "No neighbor signal",
    "no_redundancy": "No redundancy penalty",
    "no_aoi": "No AoI term",
    "no_energy": "No energy term",
    "no_agent_id": "No agent ID",
    "no_coordination_constraint": "No coordination constraint",
}


class ObservationMaskPolicy:
    """Apply only the observation removal that was used while training."""

    def __init__(self, policy: ONNXPolicy, zero_neighbor: bool = False):
        self.policy = policy
        self.zero_neighbor = zero_neighbor
        self.train_seed = policy.train_seed

    def reset(self, *args: Any) -> None:
        self.policy.reset()

    @property
    def last_q(self) -> Dict[str, np.ndarray]:
        return self.policy.last_q

    def select_action(self, agent_id: str, obs: np.ndarray, info: dict) -> int:
        decision_obs = np.asarray(obs, dtype=np.float32).copy()
        if self.zero_neighbor:
            decision_obs[shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE] = 0.0
        return self.policy.select_action(agent_id, decision_obs, info)


def _model_paths(variant: str, seeds: Iterable[int]) -> List[Path]:
    root = (
        ROOT_DIR / "results" / "learned_models" / "coordinated"
        if variant == "full"
        else ROOT_DIR / "results" / "ablation_models" / variant / "coordinated"
    )
    paths = [root / f"qmix_seed{seed}.onnx" for seed in seeds]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing retrained ablation models: {missing}")
    return paths


def _load_policy(path: Path, variant: str) -> ObservationMaskPolicy:
    match = re.search(r"seed(\d+)", path.name)
    policy = ONNXPolicy(path, include_agent_id=variant != "no_agent_id")
    policy.train_seed = int(match.group(1)) if match else None
    return ObservationMaskPolicy(policy, zero_neighbor=variant == "no_neighbor_signal")


def train_ablations(
    variants: Iterable[str], seeds: List[int], scenario: str, t_max: int
) -> None:
    for variant in variants:
        print(f"\nRETRAIN ABLATION: {variant}")
        summaries = train_suite(
            algorithms=["qmix"],
            seeds=seeds,
            regimes=["coordinated"],
            t_max=t_max,
            scenario=scenario,
            ablation=variant,
        )
        failed = [item for item in summaries if item.get("status") != "SUCCESS"]
        if failed:
            raise RuntimeError(f"Ablation training failed for {variant}: {failed}")


def _summarise(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Training-seed replicas are averaged within each held-out environment seed;
    # the 30 paired environment seeds are the sampling unit.
    per_environment = raw.groupby(
        ["ablation", "environment_seed"], as_index=False
    )[METRICS].mean()
    rows: List[Dict[str, Any]] = []
    for variant, frame in per_environment.groupby("ablation"):
        row: Dict[str, Any] = {
            "ablation": variant,
            "display_name": DISPLAY_NAMES[variant],
            "n_training_seeds": int(raw.loc[raw.ablation == variant, "training_seed"].nunique()),
            "n_environment_seeds": len(frame),
        }
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

    full = per_environment[per_environment.ablation == "full"].set_index("environment_seed")
    comparison_rows: List[Dict[str, Any]] = []
    directions = {
        "raw_episode_reward": 1.0,
        "event_recall": 1.0,
        "total_energy_consumption": -1.0,
        "mean_aoi": -1.0,
        "redundant_sampling": -1.0,
        "network_utility": 1.0,
    }
    for variant in sorted(set(per_environment.ablation) - {"full"}):
        ablated = per_environment[per_environment.ablation == variant].set_index("environment_seed")
        common = full.index.intersection(ablated.index)
        for metric, direction in directions.items():
            # Positive means the full model is better in the metric's
            # engineering direction.
            difference = direction * (
                full.loc[common, metric].to_numpy(float)
                - ablated.loc[common, metric].to_numpy(float)
            )
            test = stats.ttest_1samp(difference, 0.0)
            std = float(difference.std(ddof=1))
            comparison_rows.append({
                "ablation": variant,
                "display_name": DISPLAY_NAMES[variant],
                "metric": metric,
                "n_pairs": len(common),
                "full_model_advantage_mean": float(difference.mean()),
                "full_model_advantage_std": std,
                "test": "two-sided one-sample t-test on paired seed differences",
                "assumption": "paired environment-seed differences are approximately normal",
                "statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "paired_cohens_dz": float(difference.mean() / std) if std else np.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(comparison_rows), per_environment


def evaluate_ablations(
    variants: Iterable[str],
    train_seeds: List[int],
    test_seeds: List[int],
    scenario: str,
) -> pd.DataFrame:
    raw_rows: List[Dict[str, Any]] = []
    for variant in ["full", *variants]:
        print(f"EVALUATE ABLATION: {variant}")
        for path in _model_paths(variant, train_seeds):
            policy = _load_policy(path, variant)
            for environment_seed in test_seeds:
                # evaluate_episode uses the unablated coordinated environment:
                # common physics, common reward, and common held-out seed.
                metric, _, _, _ = evaluate_episode(
                    DISPLAY_NAMES[variant], policy, scenario, "coordinated", environment_seed
                )
                metric["ablation"] = variant
                metric["model_path"] = str(path)
                raw_rows.append(metric)

    raw = pd.DataFrame(raw_rows)
    summary, comparisons, per_environment = _summarise(raw)
    output = ROOT_DIR / "results" / "ablations"
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "ablation_raw.csv", index=False)
    summary.to_csv(output / "ablation_summary.csv", index=False)
    comparisons.to_csv(output / "ablation_paired_comparisons.csv", index=False)
    per_environment.to_csv(output / "ablation_per_environment_seed.csv", index=False)

    report = [
        "# Retrained QMIX Ablation Study\n\n",
        f"Scenario: `{scenario}`. Regime: `coordinated`. Training seeds: {train_seeds}. ",
        f"Paired held-out environment seeds: {len(test_seeds)} (`{test_seeds[0]}`-`{test_seeds[-1]}`).\n\n",
        "Each variant was retrained from scratch after removing exactly the named component. ",
        "Every resulting policy was evaluated in the same full coordinated environment and under the same full reward, so raw rewards are comparable. ",
        "Training-seed replicas were averaged within each environment seed before confidence intervals and paired tests were calculated.\n\n",
        "| Variant | Reward mean [95% CI] | Recall mean [95% CI] | Energy mean [95% CI] | Mean AoI [95% CI] |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for _, row in summary.sort_values("ablation").iterrows():
        report.append(
            f"| {row['display_name']} | {row['raw_episode_reward_mean']:.2f} "
            f"[{row['raw_episode_reward_ci95_low']:.2f}, {row['raw_episode_reward_ci95_high']:.2f}] | "
            f"{row['event_recall_mean']:.3f} [{row['event_recall_ci95_low']:.3f}, {row['event_recall_ci95_high']:.3f}] | "
            f"{row['total_energy_consumption_mean']:.2f} [{row['total_energy_consumption_ci95_low']:.2f}, {row['total_energy_consumption_ci95_high']:.2f}] | "
            f"{row['mean_aoi_mean']:.2f} [{row['mean_aoi_ci95_low']:.2f}, {row['mean_aoi_ci95_high']:.2f}] |\n"
        )
    report.extend([
        "\nPaired comparisons use a two-sided one-sample t-test on the 30 full-minus-ablated seed differences. ",
        "The accompanying CSV reports the statistic, p-value, and paired Cohen's dz; no causal importance claim should be made from a point estimate or p-value alone.\n",
    ])
    (output / "ABLATION_REPORT.md").write_text("".join(report))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Retrain every requested ablation first")
    parser.add_argument("--scenario", default="volatile", choices=shared_config.SCENARIOS)
    parser.add_argument("--t_max", type=int, default=60000)
    parser.add_argument("--train-seeds", default="101,102,103")
    parser.add_argument("--n-test-seeds", type=int, default=30)
    parser.add_argument("--variants", default=",".join(ABLATIONS))
    args = parser.parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = set(variants) - set(ABLATIONS)
    if unknown:
        raise ValueError(f"Unknown ablations: {sorted(unknown)}")
    train_seeds = [int(value) for value in args.train_seeds.split(",")]
    test_seeds = shared_config.TEST_SEEDS[: args.n_test_seeds]
    if args.train:
        train_ablations(variants, train_seeds, args.scenario, args.t_max)
    summary = evaluate_ablations(variants, train_seeds, test_seeds, args.scenario)
    print(summary[[
        "display_name", "raw_episode_reward_mean", "event_recall_mean",
        "total_energy_consumption_mean", "mean_aoi_mean",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
