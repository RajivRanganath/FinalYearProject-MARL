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
        "No advantage from QMIX over the strongest causal adaptive heuristic was observed in either evaluated regime. ",
        "The learned policies were repaired from an Always-Sleep collapse and all selected checkpoints demonstrably chose both actions, ",
        "but the causal event-proxy threshold remained better on reward, event recall, energy use, and AoI. ",
        f"In the independent regime, the best learned policy was {headline['independent']['best_learned'].policy} "
        f"(reward {headline['independent']['best_learned'].raw_episode_reward_mean:.2f}) versus the threshold policy "
        f"({headline['independent']['threshold'].raw_episode_reward_mean:.2f}). In the coordinated regime, the best learned policy was "
        f"{headline['coordinated']['best_learned'].policy} ({headline['coordinated']['best_learned'].raw_episode_reward_mean:.2f}) "
        f"versus the threshold policy ({headline['coordinated']['threshold'].raw_episode_reward_mean:.2f}). ",
        "The coordinated regime therefore created genuine inter-agent dependence, but not enough complexity for value decomposition to beat a well-aligned local proxy rule at this training budget and network scale.\n\n",
        _training_upgrade_section(),
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
        "venv/bin/pytest -q\n",
        "```\n",
    ])
    output = ROOT_DIR / "FINAL_RESEARCH_REPORT.md"
    output.write_text("".join(text))
    print(output)
    return output


if __name__ == "__main__":
    build_report()
