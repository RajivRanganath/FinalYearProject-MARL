"""Paired 30-seed benchmark for both scientific experiment regimes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from environment.pettingzoo_env import IoTSensorEnv
from training.policy_runtime import ONNXPolicy


class StatelessPolicy:
    train_seed: Optional[int] = None

    def reset(self, seed: int) -> None:
        self.seed = seed


class AlwaysSleep(StatelessPolicy):
    def select_action(self, agent_id: str, obs: np.ndarray, info: dict) -> int:
        return shared_config.ACTION_SLEEP


class AlwaysSample(StatelessPolicy):
    def select_action(self, agent_id: str, obs: np.ndarray, info: dict) -> int:
        return shared_config.ACTION_SAMPLE if info["action_mask"][1] else shared_config.ACTION_SLEEP


class RandomFeasible(StatelessPolicy):
    def reset(self, seed: int) -> None:
        self.rng = np.random.RandomState(seed + 7919)

    def select_action(self, agent_id: str, obs: np.ndarray, info: dict) -> int:
        available = np.flatnonzero(info["action_mask"])
        return int(self.rng.choice(available))


class FixedInterval(StatelessPolicy):
    def select_action(self, agent_id: str, obs: np.ndarray, info: dict) -> int:
        # A deployable staggered schedule is stronger than synchronising every
        # node and is fair to the learned coordination methods.
        idx = int(agent_id.split("_")[1])
        interval = shared_config.BASELINE_FIXED_INTERVAL_N
        offset = (idx * interval) // shared_config.NUM_AGENTS
        due = (info["timestep"] - offset) % interval == 0
        return shared_config.ACTION_SAMPLE if due and info["action_mask"][1] else shared_config.ACTION_SLEEP


class EntropyThreshold(StatelessPolicy):
    def select_action(self, agent_id: str, obs: np.ndarray, info: dict) -> int:
        proxy = obs[shared_config.STATE_INDEX_EVENT_PROXY]
        return shared_config.ACTION_SAMPLE if (
            proxy >= shared_config.BASELINE_RULE_ENTROPY_THRESHOLD and info["action_mask"][1]
        ) else shared_config.ACTION_SLEEP


class BatteryEntropy(StatelessPolicy):
    def select_action(self, agent_id: str, obs: np.ndarray, info: dict) -> int:
        battery = obs[shared_config.STATE_INDEX_RESIDUAL_ENERGY]
        proxy = obs[shared_config.STATE_INDEX_EVENT_PROXY]
        aoi = obs[shared_config.STATE_INDEX_AOI]
        should_sample = (proxy >= 0.60 and battery >= 0.10) or (aoi >= 0.70 and battery >= 0.60)
        return shared_config.ACTION_SAMPLE if should_sample and info["action_mask"][1] else shared_config.ACTION_SLEEP


class GreedyHeuristic(StatelessPolicy):
    def select_action(self, agent_id: str, obs: np.ndarray, info: dict) -> int:
        if not info["action_mask"][1]:
            return shared_config.ACTION_SLEEP
        proxy = obs[shared_config.STATE_INDEX_EVENT_PROXY]
        neighbor_rate = obs[shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE]
        p_event = (1.0 - shared_config.EVENT_PROXY_FALSE_NEGATIVE_RATE) if proxy >= 0.60 else shared_config.EVENT_PROXY_FALSE_POSITIVE_RATE
        w = shared_config.REWARD_WEIGHTS
        sample_value = p_event * w["w_info"] - w["w_energy"] - neighbor_rate * w["w_redundancy"]
        sleep_value = -p_event * w["w_miss"]
        return shared_config.ACTION_SAMPLE if sample_value > sleep_value else shared_config.ACTION_SLEEP


def _reset_policy(policy: Any, seed: int) -> None:
    try:
        policy.reset(seed)
    except TypeError:
        policy.reset()


def evaluate_episode(
    policy_name: str,
    policy: Any,
    scenario: str,
    regime: str,
    seed: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    env = IoTSensorEnv(scenario=scenario, regime=regime, seed=seed)
    obs, infos = env.reset(seed=seed)
    _reset_policy(policy, seed)

    reward = energy = harvested = 0.0
    events = captures = samples = requests = rejections = channel_blocks = 0
    redundant_pairs = 0.0
    aoi_values: List[float] = []
    coverage_values: List[float] = []
    trajectory: List[Dict[str, Any]] = []
    components: List[Dict[str, Any]] = []
    q_records: List[Dict[str, Any]] = []
    action_counts = np.zeros(2, dtype=int)
    done = False
    step = 0

    while not done:
        actions: Dict[str, int] = {}
        for aid in env.possible_agents:
            # Policies receive a deliberately sanitised info object: no latent
            # entropy or post-decision diagnostic can leak into the action.
            decision_info = {
                "timestep": step,
                "action_mask": infos[aid]["action_mask"],
            }
            action = policy.select_action(aid, obs[aid], decision_info)
            actions[aid] = action
            action_counts[action] += 1
            if hasattr(policy, "last_q") and aid in policy.last_q:
                q = policy.last_q[aid]
                q_records.append({
                    "policy": policy_name,
                    "training_seed": getattr(policy, "train_seed", None),
                    "environment_seed": seed,
                    "regime": regime,
                    "step": step,
                    "agent": aid,
                    "q_sleep": float(q[0]),
                    "q_sample": float(q[1]),
                    "q_sample_minus_sleep": float(q[1] - q[0]),
                })

        obs, rewards, terms, truncs, infos = env.step(actions)
        step += 1
        reward += sum(rewards.values())
        delivered_this_step = sum(int(info["sample_delivered"]) for info in infos.values())
        samples += delivered_this_step
        requests += sum(int(info["sample_requested"]) for info in infos.values())
        rejections += sum(int(info["sample_rejected"]) for info in infos.values())
        channel_blocks += sum(int(info.get("channel_blocked", False)) for info in infos.values())
        redundant_pairs += sum(info.get("neighbor_co_samplers", 0) for info in infos.values()) / 2.0

        for aid, info in infos.items():
            event = int(info["is_high_entropy"])
            events += event
            captures += int(event and info["sample_delivered"])
            energy += info["consumed_energy"]
            harvested += info["harvested_energy"]
            aoi_values.append(float(info["aoi"]))
            coverage_values.append(float(info.get("network_coverage", 1.0)))
            for name, value in info["reward_components"].items():
                components.append({
                    "policy": policy_name,
                    "training_seed": getattr(policy, "train_seed", None),
                    "environment_seed": seed,
                    "regime": regime,
                    "step": step,
                    "agent": aid,
                    "component": name,
                    "value": float(value),
                })

        batteries = [info["battery"] for info in infos.values()]
        trajectory.append({
            "policy": policy_name,
            "training_seed": getattr(policy, "train_seed", None),
            "environment_seed": seed,
            "regime": regime,
            "step": step,
            "battery_mean": float(np.mean(batteries)),
            "battery_min": float(np.min(batteries)),
            "samples": delivered_this_step,
            "events": sum(int(info["is_high_entropy"]) for info in infos.values()),
            "coverage": float(np.mean([info.get("network_coverage", 1.0) for info in infos.values()])),
        })
        done = all(terms.values()) or all(truncs.values())

    final_batteries = [info["battery"] for info in infos.values()]
    recall = captures / max(1, events)
    metric = {
        "policy": policy_name,
        "training_seed": getattr(policy, "train_seed", None),
        "environment_seed": seed,
        "scenario": scenario,
        "regime": regime,
        "raw_episode_reward": reward,
        "event_recall": recall,
        "missed_event_rate": 1.0 - recall,
        "total_energy_consumption": energy,
        "harvested_energy": harvested,
        "final_battery": float(np.mean(final_batteries)),
        "mean_aoi": float(np.mean(aoi_values)),
        "p95_aoi": float(np.percentile(aoi_values, 95)),
        "max_aoi": float(np.max(aoi_values)),
        "redundant_sampling": redundant_pairs,
        "network_coverage": float(np.mean(coverage_values)),
        "network_utility": float(np.mean(coverage_values) * recall),
        "samples_delivered": samples,
        "samples_requested": requests,
        "sample_rejections": rejections,
        "channel_blocks": channel_blocks,
        "sleep_actions": int(action_counts[0]),
        "sample_actions": int(action_counts[1]),
        "sample_action_fraction": float(action_counts[1] / action_counts.sum()),
    }
    return metric, trajectory, components, q_records


METRICS = [
    "event_recall", "missed_event_rate", "total_energy_consumption", "harvested_energy",
    "final_battery", "mean_aoi", "p95_aoi", "max_aoi", "redundant_sampling",
    "network_coverage", "network_utility", "raw_episode_reward", "sample_action_fraction",
]


def _summary(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Average across training seeds at each environment seed, then use the 30
    # paired environment seeds as the CI sampling units.
    per_environment = raw.groupby(["policy", "environment_seed"], as_index=False)[METRICS].mean()
    rows = []
    for policy, frame in per_environment.groupby("policy"):
        row: Dict[str, Any] = {"policy": policy, "n_environment_seeds": len(frame)}
        for metric in METRICS:
            values = frame[metric].to_numpy(float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            half = float(stats.t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
            row.update({
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_ci95_low": mean - half,
                f"{metric}_ci95_high": mean + half,
            })
        rows.append(row)
    seed_summary = raw.groupby(["policy", "training_seed"], dropna=False, as_index=False)[METRICS].agg(["mean", "std"])
    seed_summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in seed_summary.columns]
    return pd.DataFrame(rows), seed_summary, per_environment


def _paired_comparisons(
    per_environment: pd.DataFrame,
    comparators: Iterable[str] = ("Battery + Entropy", "Entropy Threshold"),
) -> pd.DataFrame:
    rows = []
    directions = {
        "raw_episode_reward": 1,
        "event_recall": 1,
        "total_energy_consumption": -1,
        "mean_aoi": -1,
    }
    for comparator in comparators:
        base = per_environment[per_environment.policy == comparator].set_index("environment_seed")
        for policy in ("IQL", "VDN", "QMIX"):
            learned = per_environment[per_environment.policy == policy].set_index("environment_seed")
            common = learned.index.intersection(base.index)
            if len(common) < 2:
                continue
            for metric, direction in directions.items():
                # Positive is an engineering advantage for the learned policy;
                # negative is an engineering disadvantage, including for
                # metrics such as energy and AoI where lower is preferable.
                difference = direction * (
                    learned.loc[common, metric].to_numpy()
                    - base.loc[common, metric].to_numpy()
                )
                test = stats.ttest_1samp(difference, popmean=0.0)
                std = float(np.std(difference, ddof=1))
                mean = float(np.mean(difference))
                threshold = 0.05 * max(abs(float(base.loc[common, metric].mean())), 1e-9)
                practical = abs(mean) >= threshold
                outcome = "negligible"
                if practical:
                    outcome = "learned advantage" if mean > 0 else "learned disadvantage"
                rows.append({
                    "policy": policy,
                    "comparator": comparator,
                    "metric": metric,
                    "n_pairs": len(common),
                    "advantage_mean": mean,
                    "advantage_std": std,
                    "test": "two-sided one-sample t-test on paired differences",
                    "assumption": "paired seed differences are approximately normal",
                    "statistic": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "paired_cohens_dz": mean / std if std else np.nan,
                    "statistically_supported_0.05": bool(test.pvalue < 0.05),
                    "practical_threshold_5pct_comparator": threshold,
                    "absolute_practically_meaningful": bool(practical),
                    "practical_outcome": outcome,
                })
    return pd.DataFrame(rows)


def _learned_policies(regime: str) -> Dict[str, List[ONNXPolicy]]:
    root = ROOT_DIR / "results" / "learned_models" / regime
    result: Dict[str, List[ONNXPolicy]] = {"IQL": [], "VDN": [], "QMIX": []}
    for label in result:
        for path in sorted(root.glob(f"{label.lower()}_seed*.onnx")):
            match = re.search(r"seed(\d+)", path.name)
            if not match or int(match.group(1)) not in shared_config.TRAIN_SEEDS[:3]:
                continue
            policy = ONNXPolicy(path)
            policy.train_seed = int(match.group(1))
            result[label].append(policy)
    return result


def run_benchmark(
    regime: str,
    scenario: str = "volatile",
    seeds: Iterable[int] = shared_config.TEST_SEEDS,
    require_learned: bool = True,
) -> pd.DataFrame:
    seeds = list(seeds)
    if seeds != shared_config.TEST_SEEDS:
        print(f"WARNING: non-final seed set supplied ({len(seeds)} seeds)")
    static = {
        "Always Sleep": AlwaysSleep(),
        "Always Sample": AlwaysSample(),
        "Random Feasible": RandomFeasible(),
        "Fixed Interval": FixedInterval(),
        "Entropy Threshold": EntropyThreshold(),
        "Battery + Entropy": BatteryEntropy(),
        "Greedy": GreedyHeuristic(),
    }
    learned = _learned_policies(regime)
    if require_learned and any(len(items) < 3 for items in learned.values()):
        counts = {name: len(items) for name, items in learned.items()}
        raise RuntimeError(f"Expected at least three trained seeds per algorithm for {regime}: {counts}")

    raw, trajectories, components, q_records = [], [], [], []
    policy_sets: Dict[str, List[Any]] = {name: [policy] for name, policy in static.items()}
    policy_sets.update(learned)
    for name, replicas in policy_sets.items():
        for policy in replicas:
            for seed in seeds:
                metric, traj, comp, q_diag = evaluate_episode(name, policy, scenario, regime, seed)
                raw.append(metric)
                trajectories.extend(traj)
                components.extend(comp)
                q_records.extend(q_diag)

    output = ROOT_DIR / "results" / "final" / regime
    output.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(raw)
    summary, train_seed_summary, per_environment = _summary(raw_df)
    paired = _paired_comparisons(per_environment)
    raw_df.to_csv(output / "benchmark_raw.csv", index=False)
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    train_seed_summary.to_csv(output / "training_seed_summary.csv", index=False)
    per_environment.to_csv(output / "benchmark_per_environment_seed.csv", index=False)
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    pd.DataFrame(trajectories).to_csv(output / "trajectories.csv", index=False)
    pd.DataFrame(components).to_csv(output / "reward_components.csv", index=False)
    pd.DataFrame(q_records).to_csv(output / "q_diagnostics.csv", index=False)
    print(summary[["policy", "event_recall_mean", "total_energy_consumption_mean", "mean_aoi_mean", "raw_episode_reward_mean"]].to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", default="all", choices=["independent", "coordinated", "all"])
    parser.add_argument("--scenario", default="volatile", choices=shared_config.SCENARIOS)
    parser.add_argument("--allow-missing-learned", action="store_true")
    parser.add_argument(
        "--statistics-only", action="store_true",
        help="Rebuild paired-comparison tables from saved per-environment results",
    )
    parser.add_argument("--n-seeds", type=int, default=30, help="Diagnostic only; final benchmark requires 30")
    args = parser.parse_args()
    regimes = list(shared_config.REGIMES) if args.regime == "all" else [args.regime]
    if args.statistics_only:
        for regime in regimes:
            output = ROOT_DIR / "results" / "final" / regime
            per_environment = pd.read_csv(output / "benchmark_per_environment_seed.csv")
            _paired_comparisons(per_environment).to_csv(output / "paired_comparisons.csv", index=False)
        return
    seeds = shared_config.TEST_SEEDS[: args.n_seeds]
    for regime in regimes:
        run_benchmark(regime, args.scenario, seeds, require_learned=not args.allow_missing_learned)


if __name__ == "__main__":
    main()
