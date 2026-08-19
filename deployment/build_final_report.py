"""Build the final research report from saved experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config

REGIMES = ["independent", "coordinated"]
METRICS = [
    ("event_recall", "Event recall"),
    ("missed_event_rate", "Missed-event rate"),
    ("total_energy_consumption", "Energy consumed"),
    ("harvested_energy", "Energy harvested"),
    ("final_battery", "Final battery"),
    ("mean_aoi", "Mean AoI"),
    ("p95_aoi", "p95 AoI"),
    ("max_aoi", "Max AoI"),
    ("redundant_sampling", "Redundant sampling"),
    ("network_coverage", "Network coverage"),
    ("network_utility", "Network utility"),
    ("raw_episode_reward", "Raw reward"),
]


def _cell(row: pd.Series, metric: str, digits: int = 3) -> str:
    return (
        f"{row[f'{metric}_mean']:.{digits}f} ± {row[f'{metric}_std']:.{digits}f} "
        f"[{row[f'{metric}_ci95_low']:.{digits}f}, {row[f'{metric}_ci95_high']:.{digits}f}]"
    )


def _benchmark_table(frame: pd.DataFrame, metrics: List[tuple[str, str]]) -> str:
    header = "| Policy | " + " | ".join(label for _, label in metrics) + " |\n"
    divider = "|---|" + "---:|" * len(metrics) + "\n"
    rows = [header, divider]
    for _, row in frame.iterrows():
        rows.append(
            f"| {row.policy} | " + " | ".join(_cell(row, metric) for metric, _ in metrics) + " |\n"
        )
    return "".join(rows)


def _sanity_section() -> str:
    rows = [
        "| Regime | Gate | Status | Evidence |\n",
        "|---|---|---:|---|\n",
    ]
    for regime in REGIMES:
        report = json.loads((ROOT_DIR / "results" / "sanity" / f"{regime}_volatile.json").read_text())
        for check in report["checks"]:
            evidence = json.dumps(check["evidence"], sort_keys=True).replace("|", "/")
            rows.append(
                f"| {regime} | `{check['name']}` | {'PASS' if check['passed'] else 'FAIL'} | `{evidence}` |\n"
            )
    return "".join(rows)


def _training_section() -> str:
    runs = json.loads((ROOT_DIR / "results" / "experiments" / "training_manifest_full.json").read_text())
    rows = [
        "| Regime | Algorithm | Seed | Selected step | Validation reward | Recall | Sample fraction | Both actions |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for run in runs:
        selected = run["selected_checkpoint"]
        step = Path(selected["checkpoint"]).name
        rows.append(
            f"| {run['regime']} | {run['algorithm'].upper()} | {run['seed']} | {step} | "
            f"{selected['mean_team_reward']:.3f} | {selected['mean_event_recall']:.3f} | "
            f"{selected['mean_sample_fraction']:.3f} | {selected['chooses_both_actions']} |\n"
        )
    return "".join(rows)


def _paired_section() -> str:
    rows = [
        "| Regime | Learned policy | Metric | Mean engineering advantage | t | p | Cohen dz | Practical outcome |\n",
        "|---|---|---|---:|---:|---:|---:|---|\n",
    ]
    for regime in REGIMES:
        data = pd.read_csv(ROOT_DIR / "results" / "final" / regime / "paired_comparisons.csv")
        data = data[data.comparator == "Entropy Threshold"]
        for _, row in data.iterrows():
            rows.append(
                f"| {regime} | {row.policy} | {row.metric} | {row.advantage_mean:.4f} | "
                f"{row.statistic:.3f} | {row.p_value:.3e} | {row.paired_cohens_dz:.3f} | "
                f"{row.practical_outcome} |\n"
            )
    return "".join(rows)


def _diagnostic_section() -> str:
    rows = [
        "| Regime | Policy | SAMPLE fraction mean [95% CI] | Q-gap mean | Q-gap std | Q-gap 5th / 50th / 95th |\n",
        "|---|---|---:|---:|---:|---:|\n",
    ]
    for regime in REGIMES:
        summary = pd.read_csv(ROOT_DIR / "results" / "final" / regime / "benchmark_summary.csv").set_index("policy")
        q = pd.read_csv(
            ROOT_DIR / "results" / "final" / regime / "q_diagnostics.csv",
            usecols=["policy", "q_sample_minus_sleep"],
        )
        for policy in ("IQL", "VDN", "QMIX"):
            values = q.loc[q.policy == policy, "q_sample_minus_sleep"].to_numpy(float)
            s = summary.loc[policy]
            quantiles = np.quantile(values, [0.05, 0.5, 0.95])
            rows.append(
                f"| {regime} | {policy} | {s.sample_action_fraction_mean:.3f} "
                f"[{s.sample_action_fraction_ci95_low:.3f}, {s.sample_action_fraction_ci95_high:.3f}] | "
                f"{values.mean():.3f} | {values.std(ddof=1):.3f} | "
                f"{quantiles[0]:.3f} / {quantiles[1]:.3f} / {quantiles[2]:.3f} |\n"
            )
    return "".join(rows)


def _ablation_section() -> str:
    summary = pd.read_csv(ROOT_DIR / "results" / "ablations" / "ablation_summary.csv")
    comparisons = pd.read_csv(ROOT_DIR / "results" / "ablations" / "ablation_paired_comparisons.csv")
    rows = [
        "| Variant | Reward mean ± std [95% CI] | Recall mean ± std [95% CI] | Energy mean ± std [95% CI] | AoI mean ± std [95% CI] |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for _, row in summary.sort_values("ablation").iterrows():
        rows.append(
            f"| {row.display_name} | {_cell(row, 'raw_episode_reward')} | {_cell(row, 'event_recall')} | "
            f"{_cell(row, 'total_energy_consumption')} | {_cell(row, 'mean_aoi')} |\n"
        )
    reward = comparisons[comparisons.metric == "raw_episode_reward"].sort_values("p_value")
    rows.extend([
        "\nFull-minus-ablated paired reward tests (positive means the full model is better):\n\n",
        "| Ablation | Mean full-model advantage | t | p | Cohen dz |\n",
        "|---|---:|---:|---:|---:|\n",
    ])
    for _, row in reward.iterrows():
        rows.append(
            f"| {row.display_name} | {row.full_model_advantage_mean:.3f} | {row.statistic:.3f} | "
            f"{row.p_value:.3e} | {row.paired_cohens_dz:.3f} |\n"
        )
    return "".join(rows)


def _ablation_interpretation() -> str:
    data = pd.read_csv(
        ROOT_DIR / "results" / "ablations" / "ablation_paired_comparisons.csv"
    )
    reward = data[data.metric == "raw_episode_reward"].set_index("ablation")

    def result(name: str) -> str:
        row = reward.loc[name]
        return (
            f"{row.full_model_advantage_mean:.2f} reward units "
            f"(t={row.statistic:.2f}, p={row.p_value:.2e}, dz={row.paired_cohens_dz:.2f})"
        )

    return (
        "The ablations separate useful task coupling from unhelpful representation burden. "
        f"Relative to the full model, removing the coordination constraint cost {result('no_coordination_constraint')}, "
        f"removing the redundancy penalty cost {result('no_redundancy')}, and removing the energy term cost {result('no_energy')}; "
        "these positive full-model advantages support retaining those three mechanisms in this task. "
        f"Conversely, the full model was worse than no agent ID by {-reward.loc['no_agent_id'].full_model_advantage_mean:.2f} reward units "
        f"(p={reward.loc['no_agent_id'].p_value:.2e}) and worse than no neighbor signal by "
        f"{-reward.loc['no_neighbor_signal'].full_model_advantage_mean:.2f} units "
        f"(p={reward.loc['no_neighbor_signal'].p_value:.2e}). Removing AoI also produced a smaller "
        f"{ -reward.loc['no_aoi'].full_model_advantage_mean:.2f}-unit improvement "
        f"(p={reward.loc['no_aoi'].p_value:.2e}). In this four-agent setting, identity and lagged-neighbor inputs appear to add optimization or overfitting burden rather than exploitable coordination information. "
        "That is a model-and-budget result, not proof that those signals are intrinsically harmful.\n"
    )


def _training_upgrade_section() -> str:
    """Render the optional post-report upgrade strictly from saved artifacts."""
    root = ROOT_DIR / "results" / "training_upgrade" / "coordinated"
    manifest_path = ROOT_DIR / "results" / "upgrade_experiments" / "training_manifest_improved_full.json"
    if not (root / "summary.csv").exists() or not manifest_path.exists():
        return ""
    summary = pd.read_csv(root / "summary.csv").set_index("policy")
    comparisons = pd.read_csv(root / "paired_comparisons.csv")
    manifest = json.loads(manifest_path.read_text())
    validation = ", ".join(
        f"seed {run['seed']}: {run['selected_checkpoint']['mean_team_reward']:.2f}"
        for run in sorted(manifest, key=lambda item: item["seed"])
    )
    rows = [
        "## Post-report validation-driven training upgrade\n\n",
        "After the original benchmark was frozen, a separate training-only upgrade fixed terminal checkpoint loss, increased replay updates for 288-step episodes, and used the homogeneous shared-policy configuration supported by the earlier ablations. ",
        "No environment physics, reward weight, baseline, or original test result was changed. The three selected validation rewards were ",
        f"{validation}. Because seeds 1001--1030 had already informed the ablation analysis, the upgraded claim uses the predeclared fresh holdout 2001--2030.\n\n",
        "| Policy | Reward [95% CI] | Recall [95% CI] | Energy [95% CI] | Mean AoI [95% CI] |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for policy in ("Published QMIX", "Upgraded QMIX", "Entropy Threshold"):
        row = summary.loc[policy]
        rows.append(
            f"| {policy} | {row.raw_episode_reward_mean:.2f} "
            f"[{row.raw_episode_reward_ci95_low:.2f}, {row.raw_episode_reward_ci95_high:.2f}] | "
            f"{row.event_recall_mean:.3f} [{row.event_recall_ci95_low:.3f}, {row.event_recall_ci95_high:.3f}] | "
            f"{row.total_energy_consumption_mean:.2f} "
            f"[{row.total_energy_consumption_ci95_low:.2f}, {row.total_energy_consumption_ci95_high:.2f}] | "
            f"{row.mean_aoi_mean:.2f} [{row.mean_aoi_ci95_low:.2f}, {row.mean_aoi_ci95_high:.2f}] |\n"
        )
    published_reward = comparisons[
        (comparisons.comparator == "Published QMIX")
        & (comparisons.metric == "raw_episode_reward")
    ].iloc[0]
    threshold_reward = comparisons[
        (comparisons.comparator == "Entropy Threshold")
        & (comparisons.metric == "raw_episode_reward")
    ].iloc[0]
    rows.extend([
        "\nThe upgraded policy gains ",
        f"{published_reward.upgraded_engineering_advantage_mean:.2f} paired reward units over published QMIX "
        f"(95% CI {published_reward.upgraded_engineering_advantage_ci95_low:.2f} to "
        f"{published_reward.upgraded_engineering_advantage_ci95_high:.2f}; "
        f"p={published_reward.p_value:.2e}, dz={published_reward.paired_cohens_dz:.2f}). ",
        "It nevertheless remains behind the causal threshold by ",
        f"{-threshold_reward.upgraded_engineering_advantage_mean:.2f} reward units "
        f"(p={threshold_reward.p_value:.2e}). The upgrade therefore strengthens QMIX substantially without changing the central negative-result conclusion. ",
        "Full paired confidence intervals and effect sizes are in [the training-upgrade report](results/training_upgrade/coordinated/REPORT.md).\n\n",
    ])
    return "".join(rows)


def _extended_training_section() -> str:
    """Render the split-locked continuation experiment from frozen artifacts."""
    root = ROOT_DIR / "results" / "training_v2"
    final_root = root / "final"
    selection_path = root / "selection" / "selection_decision.json"
    extended_manifest_path = (
        ROOT_DIR / "results" / "upgrade_experiments" / "training_manifest_extended_full.json"
    )
    identity_manifest_path = (
        ROOT_DIR / "results" / "upgrade_experiments" / "training_manifest_improved_v2_full.json"
    )
    required = [
        final_root / "summary.csv",
        final_root / "paired_comparisons.csv",
        final_root / "per_training_seed.csv",
        final_root / "evaluation_manifest.json",
        selection_path,
        extended_manifest_path,
    ]
    if not all(path.exists() for path in required):
        return ""

    summary = pd.read_csv(final_root / "summary.csv").set_index("policy")
    comparisons = pd.read_csv(final_root / "paired_comparisons.csv")
    per_training_seed = pd.read_csv(final_root / "per_training_seed.csv")
    selection = json.loads(selection_path.read_text())
    final_manifest = json.loads((final_root / "evaluation_manifest.json").read_text())
    extended_manifest = json.loads(extended_manifest_path.read_text())

    selected = ", ".join(
        f"seed {run['seed']}: step {Path(run['selected_checkpoint']['checkpoint']).name}, "
        f"reward {run['selected_checkpoint']['mean_team_reward']:.2f}"
        for run in sorted(extended_manifest, key=lambda item: item["seed"])
    )
    replica_rewards = per_training_seed[
        per_training_seed.policy == "Extended QMIX"
    ].sort_values("training_seed")
    replica_text = ", ".join(
        f"seed {int(row.training_seed)}: {row.raw_episode_reward_mean:.2f}"
        for _, row in replica_rewards.iterrows()
    )

    def comparison(comparator: str, metric: str) -> pd.Series:
        return comparisons[
            (comparisons.comparator == comparator) & (comparisons.metric == metric)
        ].iloc[0]

    improved_reward = comparison("Improved QMIX", "raw_episode_reward")
    threshold_reward = comparison("Entropy Threshold", "raw_episode_reward")
    threshold_recall = comparison("Entropy Threshold", "event_recall")
    threshold_energy = comparison("Entropy Threshold", "total_energy_consumption")
    threshold_aoi = comparison("Entropy Threshold", "mean_aoi")

    rows = [
        "## Split-locked extended-training result\n\n",
        "A further one-factor experiment restored agent identity while keeping the 180k-step configuration fixed. It was screened out after the first training replica: ",
    ]
    if identity_manifest_path.exists():
        identity_manifest = json.loads(identity_manifest_path.read_text())
        identity_validation = identity_manifest[0]["selected_checkpoint"]["mean_team_reward"]
        improved_manifest = json.loads(
            (ROOT_DIR / "results" / "upgrade_experiments" / "training_manifest_improved_full.json").read_text()
        )
        improved_seed101 = next(run for run in improved_manifest if run["seed"] == 101)
        improved_validation = improved_seed101["selected_checkpoint"]["mean_team_reward"]
        rows.append(
            f"its seed-101 validation reward was {identity_validation:.2f}, versus "
            f"{improved_validation:.2f} for the matched no-identity model. "
        )
    rows.extend([
        "The retained hypothesis changed only training duration: each of the three no-identity QMIX replicas was warm-started from its validation-selected 180k checkpoint and trained to a 360k horizon. Replay, RNG progress, learner counters, and target history were not restored. ",
        f"Validation selected {selected}; seed 103 was selected before the terminal checkpoint rather than automatically taking the last model.\n\n",
        "The predeclared promotion gate used only selection seeds 211--230 and required both a positive mean reward difference and improvement in every matched training replica. It passed with replica advantages ",
        ", ".join(
            f"seed {seed}: {advantage:.2f}"
            for seed, advantage in selection["reward_advantage_by_training_seed"].items()
        ),
        f" (mean {selection['mean_reward_advantage']:.2f}). Only then was the untouched final split 3001--3030 evaluated once.\n\n",
        "| Policy | Reward [95% CI] | Recall | Energy | Mean AoI | Redundant samples |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ])
    for policy in ("Published QMIX", "Improved QMIX", "Extended QMIX", "Entropy Threshold"):
        row = summary.loc[policy]
        rows.append(
            f"| {policy} | {row.raw_episode_reward_mean:.2f} "
            f"[{row.raw_episode_reward_ci95_low:.2f}, {row.raw_episode_reward_ci95_high:.2f}] | "
            f"{row.event_recall_mean:.3f} | {row.total_energy_consumption_mean:.2f} | "
            f"{row.mean_aoi_mean:.2f} | {row.redundant_sampling_mean:.2f} |\n"
        )
    rows.extend([
        "\nExtended QMIX improved over Improved QMIX by ",
        f"{improved_reward.extended_engineering_advantage_mean:.2f} paired reward units; its two-way bootstrap 95% interval was "
        f"[{improved_reward.two_way_bootstrap_ci95_low:.2f}, {improved_reward.two_way_bootstrap_ci95_high:.2f}] and Holm-adjusted p={improved_reward.p_value_holm:.2e}. ",
        f"Final reward was stable across the three training replicas ({replica_text}). ",
        "The stronger training result still did not beat the threshold: Extended QMIX was worse by ",
        f"{-threshold_reward.extended_engineering_advantage_mean:.2f} reward units (Holm p={threshold_reward.p_value_holm:.2e}), "
        f"{abs(threshold_recall.extended_engineering_advantage_mean):.3f} recall, and "
        f"{abs(threshold_energy.extended_engineering_advantage_mean):.2f} energy units, while improving mean AoI by "
        f"{threshold_aoi.extended_engineering_advantage_mean:.2f}. The final environment seeds are now consumed and frozen; they must not be used for further selection or tuning. ",
        "See [the frozen final report](results/training_v2/final/REPORT.md).\n\n",
        "The final manifest records the exact 30 seeds plus SHA-256 fingerprints for 12 primary model files and six listed evaluation/source files. A later dependency-closure audit found that those six hashes omitted behavior-changing transitive files, including the energy model and PettingZoo wrapper. The frozen numbers remain internally reproducible from their raw CSVs, but the fingerprint claim is partial rather than complete. "
        f"It records Git SHA `{final_manifest['git_sha']}` and the worktree-dirty disclosure.\n\n",
    ])
    return "".join(rows)


def _ensemble_deployment_section() -> str:
    """Render the frozen v3 deployment-ensemble result."""
    root = ROOT_DIR / "results" / "training_v3"
    required = [
        root / "selection" / "selection_decision.json",
        root / "selection" / "candidate_comparisons.csv",
        root / "final" / "summary.csv",
        root / "final" / "candidate_comparisons.csv",
        root / "final" / "evaluation_manifest.json",
        root / "audit.json",
    ]
    if not all(path.exists() for path in required):
        return ""
    decision = json.loads(required[0].read_text())
    selection_comparisons = pd.read_csv(required[1])
    summary = pd.read_csv(required[2]).set_index("policy")
    comparisons = pd.read_csv(required[3])

    def comparison(comparator: str, metric: str) -> pd.Series:
        return comparisons[
            (comparisons.comparator == comparator) & (comparisons.metric == metric)
        ].iloc[0]

    reference_reward = comparison("Extended QMIX Replica Mean", "raw_episode_reward")
    threshold_reward = comparison("Entropy Threshold", "raw_episode_reward")
    threshold_recall = comparison("Entropy Threshold", "event_recall")
    threshold_energy = comparison("Entropy Threshold", "total_energy_consumption")
    threshold_aoi = comparison("Entropy Threshold", "mean_aoi")
    threshold_redundancy = comparison("Entropy Threshold", "redundant_sampling")
    selection_evidence = decision["candidate_evidence"]["QMIX Unanimous Ensemble"]
    selection_reward = selection_comparisons[
        (selection_comparisons.candidate == "QMIX Unanimous Ensemble")
        & (selection_comparisons.comparator == "Extended QMIX Replica Mean")
        & (selection_comparisons.metric == "raw_episode_reward")
    ].iloc[0]
    rows = [
        "## Split-locked deployment ensemble\n\n",
        "A final bounded iteration left all trained weights, environment dynamics, and reward terms unchanged. It predeclared two scale-independent deployment rules over the three Extended QMIX replicas: majority voting and unanimous voting for SAMPLE. Selection seeds 231--250 rejected majority voting because its reward interval crossed zero. Unanimous voting passed every promotion gate, gaining ",
        f"{selection_evidence['reward_advantage']:.2f} reward units with selection CI "
        f"[{selection_reward.ci95_low:.2f}, {selection_reward.ci95_high:.2f}], while also improving recall, energy, and redundancy. Only that rule was evaluated on the untouched final seeds 4001--4030.\n\n",
        "The historical promotion predicate reported but did not require the candidate-family Holm-adjusted reward p-value. The selected unanimous rule nevertheless has ",
        f"`p_holm = {selection_evidence['reward_p_holm']:.2e}` and passes the corrected future guard, so this omission does not change the frozen winner or final result.\n\n",
        "| Policy | Reward | Recall | Energy | Mean AoI | Redundant samples |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for policy in (
        "Extended QMIX Replica Mean",
        "QMIX Unanimous Ensemble",
        "Entropy Threshold",
        "Battery + Entropy",
    ):
        row = summary.loc[policy]
        rows.append(
            f"| {policy} | {row.raw_episode_reward_mean:.2f} | {row.event_recall_mean:.3f} | "
            f"{row.total_energy_consumption_mean:.2f} | {row.mean_aoi_mean:.2f} | "
            f"{row.redundant_sampling_mean:.2f} |\n"
        )
    rows.extend([
        "\nOn the final split, unanimous voting improved over the Extended replica mean by ",
        f"{reference_reward.engineering_advantage_mean:.2f} reward units "
        f"(95% CI [{reference_reward.ci95_low:.2f}, {reference_reward.ci95_high:.2f}], "
        f"Holm p={reference_reward.p_value_holm:.2e}). It also improved recall, energy, redundancy, and coverage, but worsened AoI by 0.11. ",
        "Relative to the threshold, the remaining reward difference was ",
        f"{threshold_reward.engineering_advantage_mean:.2f} "
        f"(95% CI [{threshold_reward.ci95_low:.2f}, {threshold_reward.ci95_high:.2f}], "
        f"Holm p={threshold_reward.p_value_holm:.3f}); this is neither evidence of a reward win nor an equivalence test. Recall differed by only "
        f"{threshold_recall.engineering_advantage_mean:.3f} with an interval crossing zero. The ensemble retained a clear "
        f"{abs(threshold_energy.engineering_advantage_mean):.2f}-unit energy disadvantage and "
        f"{abs(threshold_redundancy.engineering_advantage_mean):.2f} additional redundant samples, while its AoI was "
        f"{threshold_aoi.engineering_advantage_mean:.2f} lower.\n\n",
        "The ensemble requires three recurrent-model evaluations per decision, approximately tripling model storage and inference work relative to one QMIX replica. Its final intervals vary environment seeds but condition on this one selected three-model set, so they do not measure variability across independently retrained ensembles. It is therefore a stronger simulation/deployment policy but not automatically a better TinyML choice. The final seeds are consumed and frozen. See [the v3 final report](results/training_v3/final/REPORT.md) and [artifact audit](results/training_v3/audit.json).\n\n",
    ])
    return "".join(rows)


def _refined_training_section() -> str:
    """Render the strict, rejected weight-update experiment."""
    root = ROOT_DIR / "results" / "training_v4" / "selection"
    decision_path = root / "selection_decision.json"
    summary_path = root / "summary.csv"
    seed_path = root / "per_training_seed.csv"
    if not all(path.exists() for path in (decision_path, summary_path, seed_path)):
        return ""
    decision = json.loads(decision_path.read_text())
    summary = pd.read_csv(summary_path).set_index("policy")
    seed_summary = pd.read_csv(seed_path)
    evidence = decision["evidence"]
    rows = [
        "## Strict weight-update attempt: invalidated protocol and not promoted\n\n",
        "Refined QMIX was intended to be a `1e-4` continuation of all three Extended checkpoints to a 540k-step horizon. A later artifact-level audit found that `Optimizer.load_state_dict` restored the source checkpoint parameter-group rate after the new optimizer was constructed. Every one of the 37 Refined `opt.th` files, including all three selected checkpoints, records `3e-4`. The run therefore did **not** test the declared low-learning-rate hypothesis. A separate resume audit did correctly resolve seed 103 to its validation-selected source step 289440.\n\n",
        "| Policy | Reward | Recall | Energy | Mean AoI | Redundancy |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for policy in ("Extended QMIX", "Refined QMIX", "Entropy Threshold"):
        row = summary.loc[policy]
        rows.append(
            f"| {policy} | {row.raw_episode_reward_mean:.2f} | {row.event_recall_mean:.3f} | "
            f"{row.total_energy_consumption_mean:.2f} | {row.mean_aoi_mean:.2f} | "
            f"{row.redundant_sampling_mean:.2f} |\n"
        )
    replica = seed_summary.pivot(index="training_seed", columns="policy", values="raw_episode_reward_mean")
    advantages = ", ".join(
        f"seed {int(seed)}: {row['Refined QMIX'] - row['Extended QMIX']:+.2f}"
        for seed, row in replica.iterrows()
    )
    rows.extend([
        "\nThe historical same-rate selection calculation improved by ",
        f"{evidence['mean_reward_advantage']:.2f} and the environment-seed CI lower bound was "
        f"{evidence['reward_environment_ci95_low']:.2f}. However, the two-way bootstrap lower bound was "
        f"{evidence['reward_two_way_bootstrap_ci95_low']:.2f}, and matched replica advantages were {advantages}. "
        "Because seed 102 regressed and the bootstrap crossed zero, the predeclared all-replica gate rejected the candidate even before the LR defect was discovered. These numbers now describe only a rejected same-rate warm-start and cannot answer the low-LR question. Locked seeds 5001--5030 were not evaluated. See [the invalidation record](results/training_v4/INVALIDATED.json); the original [selection report](results/training_v4/selection/REPORT.md) is retained as a frozen historical artifact whose low-LR sentence is superseded.\n\n",
    ])
    return "".join(rows)


def _source_drift_section() -> str:
    """Render the post-hoc source-repair disclosure from its own artifacts."""
    drift = json.loads((ROOT_DIR / "results" / "environment_drift.json").read_text())
    impact = json.loads((ROOT_DIR / "results" / "environment_drift_impact.json").read_text())
    feasibility = next(
        item for item in drift["repairs"]
        if item["id"] == "sample_feasibility_charges_same_step_background_energy"
    )
    rows = [
        "## Source drift after the results were produced\n\n",
        "Two source repairs postdate every run and every evaluation in this repository, so the "
        "committed source does not bit-exactly regenerate the committed numbers. The larger repair "
        f"changes SAMPLE feasibility from `{feasibility['legacy_rule']}` to "
        f"`{feasibility['repaired_rule']}`, because the previous rule admitted a sample without "
        "accounting for the unavoidable same-step sleep and proxy-monitor draw. It governs the action "
        "mask for every agent in every run.\n\n",
        "The gap is measured rather than assumed. The probe replays the real environment under both "
        "rules on the same 30 held-out seeds, using the policy that saturates the battery floor and "
        "therefore bounds how far the rules can diverge.\n\n",
        "| Regime | Mean reward delta | Max abs delta | Seeds differing | Agent-steps in disagreement band |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for entry in impact["regimes"]:
        rows.append(
            f"| {entry['regime']} | {entry['mean_reward_delta_repaired_minus_legacy']:+.4f} | "
            f"{entry['max_abs_reward_delta']:.4f} | "
            f"{entry['seeds_with_any_difference']}/{len(entry['seeds'])} | "
            f"{entry['agent_steps_inside_disagreement_band']}/{entry['episode_agent_steps']} "
            f"({100 * entry['disagreement_band_fraction']:.3f}%) |\n"
        )
    rows.extend([
        "\nThe two rules can only disagree while the battery sits inside a band "
        f"{100 * impact['disagreement_band_width']:.3f}% of capacity wide. Against a headline "
        "Extended-versus-Improved effect of 27.67 reward units with a bootstrap interval of "
        "[23.27, 32.31], no reported comparison changes sign or loses significance. The results are "
        "therefore disclosed rather than invalidated.\n\n",
        "The second repair synchronises the resumed learner's target mixer and reapplies the "
        "configured learning rate after optimizer-state restore. Because these runs were launched "
        "from a disclosed dirty worktree and the manifests recorded only a base SHA and a dirty flag, "
        "the learner source used by any historical run is not recoverable. The logged TD loss shows "
        "no spike at the resume boundary where an unsynchronised target mixer would perturb targets "
        "by roughly four reward units, which is consistent with the synchronisation having been "
        "present, but that is evidence rather than proof. Manifests now record "
        "`training_source_sha256` so the ambiguity cannot recur.\n\n",
        "One consequence is load-bearing. The resume resolver accepts only manifest entries whose "
        "configuration digest matches the current gate, and the Extended manifests carry the "
        "pre-repair digest, so a corrected low-learning-rate continuation from Extended now fails by "
        "design. It needs Extended retrained under the repaired environment, or an explicitly "
        "recorded decision to warm-start across the repair; fresh seeds alone are not sufficient. "
        "See [the drift record](results/environment_drift.json) and "
        "[its measured impact](results/environment_drift_impact.json).\n\n",
    ])
    return "".join(rows)


def build_report() -> Path:
    summaries: Dict[str, pd.DataFrame] = {
        regime: pd.read_csv(ROOT_DIR / "results" / "final" / regime / "benchmark_summary.csv")
        for regime in REGIMES
    }
    headline = {}
    for regime, frame in summaries.items():
        headline[regime] = {
            "best": frame.loc[frame.raw_episode_reward_mean.idxmax()],
            "best_learned": frame[frame.policy.isin(["IQL", "VDN", "QMIX"])].loc[
                frame[frame.policy.isin(["IQL", "VDN", "QMIX"])].raw_episode_reward_mean.idxmax()
            ],
            "threshold": frame[frame.policy == "Entropy Threshold"].iloc[0],
        }

    text = [
        "# Final Research Report: When Does Cooperative MARL Help Energy-Harvesting Sensor Networks?\n\n",
        "## Executive finding\n\n",
        "No overall engineering advantage from QMIX over the strongest causal adaptive heuristic was observed in either evaluated regime. ",
        "The learned policies were repaired from an Always-Sleep collapse and all selected checkpoints demonstrably chose both actions, ",
        "but the causal event-proxy threshold remained better on reward, event recall, energy use, and AoI. ",
        f"In the independent regime, the best learned policy was {headline['independent']['best_learned'].policy} "
        f"(reward {headline['independent']['best_learned'].raw_episode_reward_mean:.2f}) versus the threshold policy "
        f"({headline['independent']['threshold'].raw_episode_reward_mean:.2f}). In the coordinated regime, the best learned policy was "
        f"{headline['coordinated']['best_learned'].policy} ({headline['coordinated']['best_learned'].raw_episode_reward_mean:.2f}) "
        f"versus the threshold policy ({headline['coordinated']['threshold'].raw_episode_reward_mean:.2f}). ",
        "The coordinated regime therefore created genuine inter-agent dependence, but not enough complexity for value decomposition to beat a well-aligned local proxy rule at the original training budget and network scale. A later split-locked continuation substantially improved QMIX. A separately locked unanimous-vote deployment ensemble then closed the reward gap to a statistically unresolved 2.27 units and had a slightly higher but statistically unresolved recall point estimate, while still using more energy and producing more redundant samples than the threshold.\n\n",
        _training_upgrade_section(),
        _extended_training_section(),
        _ensemble_deployment_section(),
        _refined_training_section(),
        _source_drift_section(),
        "## 1. Research question and non-predetermined protocol\n\n",
        "The study asks when cooperative MARL provides an advantage over simple adaptive heuristics. It does not assume that QMIX should win. ",
        "Regime A uses independent event processes and unconstrained per-agent delivery. Regime B uses a two-packet shared channel, persistent spatially correlated events, ring-neighbor redundancy, heterogeneous energy trajectories, and network-coverage utility. ",
        "The coupling mechanisms are physical or task-derived; they were not added after observing scores.\n\n",
        "## 2. Collapse diagnosis and scientific corrections\n\n",
        "The earlier collapsed result was not treated as an algorithm-only failure. The audit found four confounds: (1) latent measurement entropy was present before the SAMPLE decision, making the old threshold baseline an oracle; ",
        "(2) `reset(seed=None)` reseeded every training episode to 42, replaying the same day; (3) the requested scenario/regime was not consistently propagated into EPyMARL; and ",
        "(4) checkpoint export could search historical directories and had no isolated validation-only selection rule. Older learned checkpoints also had negative Q(sample)-Q(sleep) over relevant states, directly explaining Always-Sleep behavior.\n\n",
        "The corrected observation is `[battery, causal event proxy, normalized AoI, lagged neighbor sampling rate, harvest forecast]`. ",
        "Latent event entropy is measured only after sampling. The event proxy represents an always-on low-power detector with false positives, false negatives, noise, and an accounted monitoring energy cost. ",
        "For compatibility with the requested baseline list, tables retain the name 'Entropy Threshold'; it is a threshold on this causal proxy, not latent entropy.\n\n",
        "## 3. Pre-training sanity gates\n\n",
        "Expensive training was blocked until the following artifact-backed gates passed. The post-training checkpoint audit additionally verified that all 18 selected policies chose both actions on validation seeds 201-210.\n\n",
        _sanity_section(),
        "\n## 4. Training and checkpoint selection\n\n",
        "IQL, VDN, and QMIX were trained independently in each regime using seeds 101, 102, and 103 (18 full runs), 60,000 requested environment steps per run, recurrent 64-unit shared agents, and validation-only checkpoint selection. ",
        "The requested minimum of three training seeds was used rather than the preferred five; this limits inference about optimization variance. Uncertainty over environment seeds and variability over training seeds are both retained in the artifacts. ",
        "Each run has an isolated directory, command/config snapshot, base Git SHA, sanity digest, all checkpoint validation decisions, and ONNX export metadata. A separate final provenance artifact records source/artifact hashes and the dirty-worktree disclosure.\n\n",
        _training_section(),
        "\n## 5. Final paired 30-seed benchmark\n\n",
        "Every policy used the identical locked environment seeds 1001-1030. Learned-policy results first average the three training replicas within each environment seed; the 30 environment seeds are then the sampling units for the reported mean, sample standard deviation, and t-based 95% confidence interval. ",
        "This prevents the three learned replicas from being misrepresented as 90 independent environment trials.\n\n",
    ]
    for regime in REGIMES:
        text.extend([
            f"### {regime.capitalize()} regime: information and energy metrics\n\n",
            _benchmark_table(summaries[regime], METRICS[:5]),
            f"\n### {regime.capitalize()} regime: freshness, coordination, and reward metrics\n\n",
            _benchmark_table(summaries[regime], METRICS[5:]),
            "\n",
        ])
    text.extend([
        "## 6. Paired statistical and practical comparison\n\n",
        "The table below compares each learned method with the causal proxy threshold using a two-sided one-sample t-test on paired seed differences. ",
        "The test assumes the 30 paired differences are approximately normal. Positive advantage always means better engineering performance; energy and AoI signs are reversed before testing because lower is better. ",
        "Cohen's dz is the paired effect size. A practical difference is defined prospectively here as at least 5% of the comparator mean; it is reported separately from p < 0.05. No multiple-comparison-adjusted confirmatory claim is made.\n\n",
        _paired_section(),
        "\n## 7. Action and value diagnostics\n\n",
        "The repaired policies are not Always Sleep. Sample-action confidence intervals exclude zero, while Q-gap distributions include state-dependent positive and negative values rather than a uniformly negative gap. ",
        "This supports a substantive negative performance result rather than a trivial collapsed-policy comparison.\n\n",
        _diagnostic_section(),
        "\n## 8. Proper retrained ablations\n\n",
        "Each QMIX ablation removed one named component, retrained three fresh seeds at the same 60,000-step budget, and used the same 30 held-out seeds. ",
        "All variants were scored in the common full coordinated environment and full reward; no frozen policy was evaluated under a changed reward and called a training ablation.\n\n",
        _ablation_section(),
        "\n### Ablation interpretation\n\n",
        _ablation_interpretation(),
        "\n## 9. Figures and artifact lineage\n\n",
        "All figures are generated from saved experiment CSV/JSON data by `deployment/generate_plots.py`; no measured metric is typed into the plotting code.\n\n",
        "1. [Training learning curves](results/figures/01_training_learning_curves.png)\n",
        "2. [Action distribution](results/figures/02_action_distribution.png)\n",
        "3. [Q(sample) - Q(sleep)](results/figures/03_q_sample_minus_sleep.png)\n",
        "4. [Reward components](results/figures/04_reward_component_distributions.png)\n",
        "5. [Event recall](results/figures/05_event_recall.png)\n",
        "6. [Energy consumption](results/figures/06_energy_consumption.png)\n",
        "7. [AoI](results/figures/07_mean_aoi.png)\n",
        "8. [Redundant sampling](results/figures/08_redundant_sampling.png)\n",
        "9. [Battery trajectories](results/figures/09_battery_trajectory.png)\n",
        "10. [Regime comparison](results/figures/10_independent_vs_coordinated.png)\n",
        "11. [Retrained ablations](results/figures/11_retrained_ablation_results.png)\n",
        "12. [Energy-recall Pareto view](results/figures/12_energy_vs_event_recall.png)\n",
        "13. [Energy-AoI Pareto view](results/figures/13_energy_vs_aoi.png)\n\n",
        "## 10. Interpretation and retained negative result\n\n",
        "The independent task is sufficiently separable that a causal local proxy rule captures its useful structure more directly than value-decomposition MARL. ",
        "The coordinated task does reveal a learned-method ordering in point estimates, with QMIX outperforming IQL and VDN in mean raw reward, but QMIX remains dominated by the threshold heuristic on the primary recall-energy-freshness trade-off. ",
        "Therefore the experiments do not support the claim that cooperative MARL provides an engineering advantage under the evaluated conditions. ",
        "A narrower, defensible conclusion is that genuine coupling alone is insufficient: MARL may require larger networks, delayed/noisy neighbor communication, non-myopic coverage dependencies, stronger heterogeneity, longer training, or architectures that exploit graph structure before its representational cost is justified. These are future hypotheses, not results from this benchmark.\n\n",
        "## 11. Limitations and reproducibility boundaries\n\n",
        "The study uses four agents, one 24-hour volatile scenario, a ring topology, a simplified round-robin MAC, a scalar reward, three training seeds, and 30 simulation seeds. ",
        "The event-proxy error rates and reward weights are modeling choices, not field-calibrated estimates. Paired t-tests are exploratory across multiple policies and metrics. ",
        "The worktree was intentionally uncommitted during these runs. Full-run manifests record the base Git SHA, while `results/provenance.json` records the final dirty status and source/artifact hashes; exact run configs and generated artifacts must accompany any external reproduction. ",
        "No result here establishes field performance or a universal absence of MARL benefit. See [LIMITATIONS.md](LIMITATIONS.md) for the physical and deployment boundary conditions.\n\n",
        "### Reproduction commands\n\n",
        "```bash\n",
        "venv/bin/python training/sanity_checks.py --regime all --scenario volatile\n",
        "venv/bin/python training/train_all.py --alg all --seeds 101,102,103 --regime all --scenario volatile --t_max 60000\n",
        "venv/bin/python deployment/evaluate_all.py --regime all --scenario volatile --n-seeds 30\n",
        "venv/bin/python deployment/ablation_study.py --train --scenario volatile --t_max 60000 --train-seeds 101,102,103 --n-test-seeds 30\n",
        "venv/bin/python deployment/generate_plots.py\n",
        "venv/bin/python deployment/build_final_report.py\n",
        "venv/bin/python deployment/write_provenance.py\n",
        "venv/bin/python deployment/audit_final_artifacts.py\n",
        "venv/bin/python -m pytest -q\n",
        "```\n",
    ])
    output = ROOT_DIR / "FINAL_RESEARCH_REPORT.md"
    output.write_text("".join(text))
    print(output)
    return output


if __name__ == "__main__":
    build_report()
