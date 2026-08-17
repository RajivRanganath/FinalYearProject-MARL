"""Generate the final scientific figures exclusively from saved artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config

OUTPUT_DIR = ROOT_DIR / "results" / "figures"
REGIMES = ["independent", "coordinated"]
POLICY_ORDER = [
    "Always Sleep", "Always Sample", "Random Feasible", "Fixed Interval",
    "Entropy Threshold", "Battery + Entropy", "Greedy", "IQL", "VDN", "QMIX",
]
COLORS = {
    "Always Sleep": "#9ca3af", "Always Sample": "#6b7280",
    "Random Feasible": "#a78bfa", "Fixed Interval": "#f59e0b",
    "Entropy Threshold": "#2563eb", "Battery + Entropy": "#06b6d4",
    "Greedy": "#14b8a6", "IQL": "#ef4444", "VDN": "#f97316", "QMIX": "#16a34a",
}


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUTPUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")
    return path


def _summary(regime: str) -> pd.DataFrame:
    path = ROOT_DIR / "results" / "final" / regime / "benchmark_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path).set_index("policy").reindex(POLICY_ORDER).reset_index()
    frame["regime"] = regime
    return frame


def _ci_bar_figure(metric: str, ylabel: str, title: str, filename: str) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, regime in zip(axes, REGIMES):
        frame = _summary(regime)
        means = frame[f"{metric}_mean"].to_numpy(float)
        errors = frame[f"{metric}_ci95_high"].to_numpy(float) - means
        ax.bar(
            np.arange(len(frame)), means, yerr=errors, capsize=3,
            color=[COLORS[name] for name in frame.policy], edgecolor="black", linewidth=0.4,
        )
        ax.set_title(regime.capitalize())
        ax.set_ylabel(ylabel)
    axes[-1].set_xticks(np.arange(len(POLICY_ORDER)), POLICY_ORDER, rotation=35, ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, filename)


def plot_training_curves() -> Path:
    records: List[Dict[str, float | str | int]] = []
    for regime in REGIMES:
        for algorithm in ("iql", "vdn", "qmix"):
            for seed in shared_config.TRAIN_SEEDS[:3]:
                root = ROOT_DIR / "results" / "experiments" / regime / algorithm / f"seed{seed}"
                for summary_path in sorted(root.glob("*/summary.json"), reverse=True):
                    summary = json.loads(summary_path.read_text())
                    if summary.get("status") != "SUCCESS" or summary.get("ablation") != "full":
                        continue
                    metrics_path = summary_path.parent / "sacred_metrics.json"
                    metrics = json.loads(metrics_path.read_text())
                    curve = metrics.get("test_return_mean", metrics.get("return_mean"))
                    for step, value in zip(curve["steps"], curve["values"]):
                        records.append({
                            "regime": regime, "algorithm": algorithm.upper(),
                            "seed": seed, "step": int(step), "value": float(value),
                        })
                    break
    data = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, regime in zip(axes, REGIMES):
        subset = data[data.regime == regime]
        for algorithm in ("IQL", "VDN", "QMIX"):
            grouped = subset[subset.algorithm == algorithm].groupby("step").value
            mean, std, count = grouped.mean(), grouped.std(), grouped.count()
            half = stats.t.ppf(0.975, count - 1) * std / np.sqrt(count)
            ax.plot(mean.index, mean.values, label=algorithm, color=COLORS[algorithm])
            ax.fill_between(mean.index, mean - half, mean + half, color=COLORS[algorithm], alpha=0.18)
        ax.set_title(regime.capitalize())
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Held-out evaluation return")
        ax.legend()
    fig.suptitle("Training learning curves (mean and 95% CI across 3 training seeds)")
    fig.tight_layout()
    return _save(fig, "01_training_learning_curves.png")


def plot_action_distribution() -> Path:
    return _ci_bar_figure(
        "sample_action_fraction", "Fraction of SAMPLE actions",
        "Action-distribution diagnostic (30 held-out seeds, 95% CI)",
        "02_action_distribution.png",
    )


def plot_q_gap() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, regime in zip(axes, REGIMES):
        data = pd.read_csv(
            ROOT_DIR / "results" / "final" / regime / "q_diagnostics.csv",
            usecols=["policy", "q_sample_minus_sleep"],
        )
        arrays = []
        for policy in ("IQL", "VDN", "QMIX"):
            values = data.loc[data.policy == policy, "q_sample_minus_sleep"].to_numpy(float)
            stride = max(1, len(values) // 20000)
            arrays.append(values[::stride])
        ax.boxplot(arrays, tick_labels=["IQL", "VDN", "QMIX"], showfliers=False, whis=(5, 95))
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_title(regime.capitalize())
        ax.set_ylabel("Q(sample) - Q(sleep)")
    fig.suptitle("Learned Q-gap distributions across held-out trajectories")
    fig.tight_layout()
    return _save(fig, "03_q_sample_minus_sleep.png")


def plot_reward_components() -> Path:
    path = ROOT_DIR / "results" / "final" / "coordinated" / "reward_components.csv"
    keep = ["Battery + Entropy", "IQL", "VDN", "QMIX"]
    keys = ["policy", "training_seed", "environment_seed", "component"]
    # This source intentionally contains step/agent-level components and can be
    # hundreds of MB. Aggregate it in chunks rather than requiring multi-GB RAM.
    parts = []
    for chunk in pd.read_csv(path, usecols=[*keys, "value"], chunksize=250000):
        chunk = chunk[chunk.policy.isin(keep)]
        parts.append(chunk.groupby(keys, dropna=False, as_index=False).value.sum())
    episodes = pd.concat(parts, ignore_index=True).groupby(
        keys, dropna=False, as_index=False
    ).value.sum()
    per_environment = episodes.groupby(
        ["policy", "environment_seed", "component"], as_index=False
    ).value.mean()
    components = sorted(per_environment.component.unique())
    fig, axes = plt.subplots(2, math.ceil(len(components) / 2), figsize=(15, 7))
    axes = np.asarray(axes).ravel()
    policies = ["Battery + Entropy", "IQL", "VDN", "QMIX"]
    for ax, component in zip(axes, components):
        arrays = [
            per_environment.loc[
                (per_environment.policy == policy) & (per_environment.component == component), "value"
            ].to_numpy(float)
            for policy in policies
        ]
        ax.boxplot(arrays, tick_labels=policies, showfliers=False)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_title(component.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel("Episode component sum")
    for ax in axes[len(components):]:
        ax.axis("off")
    fig.suptitle("Coordinated-regime reward-component distributions (30 seeds)")
    fig.tight_layout()
    return _save(fig, "04_reward_component_distributions.png")


def plot_battery_trajectory() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, regime in zip(axes, REGIMES):
        data = pd.read_csv(
            ROOT_DIR / "results" / "final" / regime / "trajectories.csv",
            usecols=["policy", "environment_seed", "step", "battery_mean"],
        )
        data = data[data.policy.isin(["Battery + Entropy", "QMIX"])]
        per_env = data.groupby(["policy", "environment_seed", "step"], as_index=False).battery_mean.mean()
        for policy in ("Battery + Entropy", "QMIX"):
            curve = per_env[per_env.policy == policy].groupby("step").battery_mean
            mean, std, n = curve.mean(), curve.std(), curve.count()
            half = stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)
            ax.plot(mean.index, mean, color=COLORS[policy], label=policy)
            ax.fill_between(mean.index, mean - half, mean + half, color=COLORS[policy], alpha=0.18)
        ax.set_title(regime.capitalize())
        ax.set_xlabel("5-minute timestep")
        ax.set_ylabel("Mean battery state")
        ax.legend()
    fig.suptitle("Battery trajectories (mean and 95% CI across paired held-out seeds)")
    fig.tight_layout()
    return _save(fig, "09_battery_trajectory.png")


def plot_regime_comparison() -> Path:
    frames = {
        regime: pd.read_csv(
            ROOT_DIR / "results" / "final" / regime / "benchmark_per_environment_seed.csv"
        ) for regime in REGIMES
    }
    rows = []
    for policy in POLICY_ORDER:
        left = frames["coordinated"][frames["coordinated"].policy == policy].set_index("environment_seed")
        right = frames["independent"][frames["independent"].policy == policy].set_index("environment_seed")
        delta = left.raw_episode_reward - right.raw_episode_reward
        half = stats.t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
        rows.append((policy, delta.mean(), half))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(
        np.arange(len(rows)), [row[1] for row in rows], yerr=[row[2] for row in rows],
        color=[COLORS[row[0]] for row in rows], capsize=3, edgecolor="black", linewidth=0.4,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(rows)), [row[0] for row in rows], rotation=35, ha="right")
    ax.set_ylabel("Coordinated - independent reward")
    ax.set_title("Regime effect on common policy classes (paired seeds, 95% CI)")
    fig.tight_layout()
    return _save(fig, "10_independent_vs_coordinated.png")


def plot_ablation() -> Path:
    data = pd.read_csv(ROOT_DIR / "results" / "ablations" / "ablation_summary.csv")
    data = data.sort_values("raw_episode_reward_mean", ascending=False)
    mean = data.raw_episode_reward_mean.to_numpy(float)
    error = data.raw_episode_reward_ci95_high.to_numpy(float) - mean
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(np.arange(len(data)), mean, yerr=error, capsize=3, color="#16a34a", edgecolor="black")
    ax.set_xticks(np.arange(len(data)), data.display_name, rotation=30, ha="right")
    ax.set_ylabel("Full-objective episode reward")
    ax.set_title("Retrained QMIX ablations (3 training seeds, 30 paired test seeds)")
    fig.tight_layout()
    return _save(fig, "11_retrained_ablation_results.png")


def _pareto(metric_y: str, ylabel: str, filename: str, invert_y: bool = False) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, regime in zip(axes, REGIMES):
        data = _summary(regime)
        for _, row in data.iterrows():
            policy = row.policy
            x = row.total_energy_consumption_mean
            y = row[f"{metric_y}_mean"]
            xerr = row.total_energy_consumption_ci95_high - x
            yerr = row[f"{metric_y}_ci95_high"] - y
            ax.errorbar(
                x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=2,
                color=COLORS[policy], markeredgecolor="black", markersize=6,
            )
            ax.annotate(policy, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_title(regime.capitalize())
        ax.set_xlabel("Total energy consumption")
        ax.set_ylabel(ylabel)
        if invert_y:
            ax.invert_yaxis()
    fig.suptitle(f"Energy trade-off: {ylabel} (95% confidence intervals)")
    fig.tight_layout()
    return _save(fig, filename)


def generate_all_figures() -> List[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    outputs = [
        plot_training_curves(),
        plot_action_distribution(),
        plot_q_gap(),
        plot_reward_components(),
        _ci_bar_figure("event_recall", "Event recall", "Event recall comparison (95% CI)", "05_event_recall.png"),
        _ci_bar_figure("total_energy_consumption", "Energy consumed", "Energy consumption comparison (95% CI)", "06_energy_consumption.png"),
        _ci_bar_figure("mean_aoi", "Mean AoI (steps)", "Age of Information comparison (95% CI)", "07_mean_aoi.png"),
        _ci_bar_figure("redundant_sampling", "Neighbor co-sampling pairs", "Redundant sampling comparison (95% CI)", "08_redundant_sampling.png"),
        plot_battery_trajectory(),
        plot_regime_comparison(),
        plot_ablation(),
        _pareto("event_recall", "Event recall", "12_energy_vs_event_recall.png"),
        _pareto("mean_aoi", "Mean AoI (lower is better)", "13_energy_vs_aoi.png", invert_y=True),
    ]
    manifest = {
        "principle": "All values loaded from saved experiment CSV/JSON artifacts; no metric literals.",
        "sources": [
            "results/experiments/**/summary.json",
            "results/experiments/**/sacred_metrics.json",
            "results/final/{independent,coordinated}/*.csv",
            "results/ablations/*.csv",
        ],
        "figures": [str(path.relative_to(ROOT_DIR)) for path in outputs],
    }
    (OUTPUT_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return outputs


if __name__ == "__main__":
    generate_all_figures()
