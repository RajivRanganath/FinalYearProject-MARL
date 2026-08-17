"""Isolated, sanity-gated, multi-seed IQL/VDN/QMIX training pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
EPYMARL_SRC = ROOT_DIR / "training" / "epymarl" / "src"
MAIN_SCRIPT = EPYMARL_SRC / "main.py"
for path in (ROOT_DIR, EPYMARL_SRC):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import shared_config
from environment.pettingzoo_env import IoTSensorEnv
from training.export_onnx import export_agent_to_onnx
from training.policy_runtime import TorchCheckpointPolicy
from training.sanity_checks import config_digest, run_sanity_checks


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT_DIR, capture_output=True, text=True, check=False
    )
    return bool(result.stdout.strip())


def _evaluate_checkpoint(
    checkpoint: Path,
    scenario: str,
    regime: str,
    ablation: str,
    seeds: Iterable[int],
) -> Dict[str, Any]:
    include_agent_id = ablation != "no_agent_id"
    policy = TorchCheckpointPolicy(checkpoint, include_agent_id=include_agent_id)
    returns, recalls, sample_fractions = [], [], []
    action_counts = np.zeros(2, dtype=int)
    for seed in seeds:
        env = IoTSensorEnv(scenario=scenario, regime=regime, ablation=ablation, seed=seed)
        obs, infos = env.reset(seed=seed)
        policy.reset()
        total_reward = 0.0
        events = captures = samples = steps = 0
        done = False
        while not done:
            actions = {}
            for aid in env.possible_agents:
                action = policy.select_action(aid, obs[aid], infos[aid])
                actions[aid] = action
                action_counts[action] += 1
            obs, rewards, terms, truncs, infos = env.step(actions)
            total_reward += sum(rewards.values())
            events += sum(int(info["is_high_entropy"]) for info in infos.values())
            captures += sum(int(info["is_high_entropy"] and info["sample_delivered"]) for info in infos.values())
            samples += sum(int(info["sample_delivered"]) for info in infos.values())
            steps += 1
            done = all(terms.values()) or all(truncs.values())
        returns.append(total_reward)
        recalls.append(captures / max(1, events))
        sample_fractions.append(samples / (steps * shared_config.NUM_AGENTS))
    return {
        "checkpoint": str(checkpoint),
        "validation_seeds": list(seeds),
        "mean_team_reward": float(np.mean(returns)),
        "std_team_reward": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "mean_event_recall": float(np.mean(recalls)),
        "mean_sample_fraction": float(np.mean(sample_fractions)),
        "action_counts": action_counts.tolist(),
        "chooses_both_actions": bool(np.all(action_counts > 0)),
    }


def _new_sacred_run(algorithm: str, map_name: str, before: set[Path]) -> Optional[Path]:
    root = EPYMARL_SRC.parent / "results" / "sacred" / algorithm / map_name
    candidates = set(path.parent for path in root.glob("*/run.json")) if root.exists() else set()
    created = candidates - before
    return max(created, key=lambda path: path.stat().st_mtime) if created else None


def _sanity_gate(scenario: str, regime: str, supplied: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    report = supplied or run_sanity_checks(scenario, regime)
    if not report["all_passed"]:
        failed = [item["name"] for item in report["checks"] if not item["passed"]]
        raise RuntimeError(f"Training blocked by failed sanity gates: {failed}")
    expected = config_digest(scenario, regime)
    if report["config_digest"] != expected:
        raise RuntimeError("Sanity report/config digest mismatch")
    return report


def run_training_experiment(
    algorithm: str,
    seed: int,
    t_max: int,
    scenario: str = "volatile",
    regime: str = "independent",
    ablation: str = "full",
    lr: float = 5e-4,
    sanity_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if algorithm not in {"iql", "vdn", "qmix"}:
        raise ValueError(f"Unsupported algorithm {algorithm}")
    gate = _sanity_gate(scenario, regime, sanity_report)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = ROOT_DIR / "results" / "experiments" / regime / algorithm / f"seed{seed}" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    map_name = f"iot_{regime}"
    sacred_root = EPYMARL_SRC.parent / "results" / "sacred" / algorithm / map_name
    sacred_before = set(path.parent for path in sacred_root.glob("*/run.json")) if sacred_root.exists() else set()

    save_interval = max(shared_config.EPISODE_LENGTH_TIMESTEPS, t_max // 4)
    anneal = max(shared_config.EPISODE_LENGTH_TIMESTEPS, int(t_max * 0.6))
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        f"--config={algorithm}",
        "--env-config=iot",
        "with",
        f"seed={seed}",
        f"t_max={t_max}",
        f"lr={lr}",
        "use_cuda=False",
        "batch_size=16",
        "buffer_size=5000",
        f"epsilon_anneal_time={anneal}",
        "epsilon_finish=0.05",
        "save_model=True",
        f"save_model_interval={save_interval}",
        f"test_interval={save_interval}",
        f"env_args.scenario={scenario}",
        f"env_args.regime={regime}",
        f"env_args.ablation={ablation}",
        f"env_args.map_name={map_name}",
        f"obs_agent_id={'False' if ablation == 'no_agent_id' else 'True'}",
        f"local_results_path={run_dir}",
    ]
    config_snapshot = {
        "algorithm": algorithm,
        "seed": seed,
        "t_max": t_max,
        "learning_rate": lr,
        "scenario": scenario,
        "regime": regime,
        "ablation": ablation,
        "validation_seeds": shared_config.VAL_SEEDS,
        "test_seeds_locked": shared_config.TEST_SEEDS,
        "config_digest": gate["config_digest"],
        "git_sha": _git_sha(),
        "git_worktree_dirty": _git_dirty(),
        "command": cmd,
        "checkpoint_rule": "maximum mean team reward on validation seeds only",
    }
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2) + "\n")
    (run_dir / "sanity_gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    print(f"\nTRAIN {algorithm.upper()} regime={regime} seed={seed} t_max={t_max}")
    started = time.time()
    process = subprocess.run(cmd, cwd=ROOT_DIR)
    elapsed = time.time() - started
    if process.returncode != 0:
        summary = {**config_snapshot, "status": "FAILED", "elapsed_seconds": elapsed}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    checkpoint_root = run_dir / "models"
    checkpoints = sorted(
        [path for path in checkpoint_root.glob("**/*") if path.is_dir() and (path / "agent.th").exists()],
        key=lambda path: int(path.name),
    )
    if not checkpoints:
        raise RuntimeError(f"Training succeeded but no checkpoints were written under {checkpoint_root}")

    decisions = [
        _evaluate_checkpoint(path, scenario, regime, ablation, shared_config.VAL_SEEDS)
        for path in checkpoints
    ]
    selected = max(decisions, key=lambda item: item["mean_team_reward"])
    (run_dir / "checkpoint_validation.json").write_text(json.dumps(decisions, indent=2) + "\n")

    # Ablations are first-class retrained policies.  Keep them outside the
    # canonical model directory so a study can never overwrite the full model.
    exported_dir = (
        ROOT_DIR / "results" / "learned_models" / regime
        if ablation == "full"
        else ROOT_DIR / "results" / "ablation_models" / ablation / regime
    )
    exported_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = exported_dir / f"{algorithm}_seed{seed}.onnx"
    export_metadata = export_agent_to_onnx(selected["checkpoint"], str(onnx_path))

    sacred_run = _new_sacred_run(algorithm, map_name, sacred_before)
    if sacred_run:
        for name in ("metrics.json", "config.json", "run.json"):
            source = sacred_run / name
            if source.exists():
                shutil.copy2(source, run_dir / f"sacred_{name}")

    summary = {
        **config_snapshot,
        "status": "SUCCESS",
        "elapsed_seconds": round(elapsed, 3),
        "run_dir": str(run_dir),
        "selected_checkpoint": selected,
        "all_checkpoint_decisions": str(run_dir / "checkpoint_validation.json"),
        "onnx": export_metadata,
        "sacred_run": str(sacred_run) if sacred_run else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def train_suite(
    algorithms: List[str],
    seeds: List[int],
    regimes: List[str],
    t_max: int,
    scenario: str,
    ablation: str = "full",
) -> List[Dict[str, Any]]:
    reports = {regime: _sanity_gate(scenario, regime) for regime in regimes}
    summaries: List[Dict[str, Any]] = []
    for regime in regimes:
        for algorithm in algorithms:
            for seed in seeds:
                summaries.append(run_training_experiment(
                    algorithm=algorithm,
                    seed=seed,
                    t_max=t_max,
                    scenario=scenario,
                    regime=regime,
                    ablation=ablation,
                    sanity_report=reports[regime],
                ))
    manifest = ROOT_DIR / "results" / "experiments" / f"training_manifest_{ablation}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", default="all", choices=["iql", "vdn", "qmix", "all"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", default="101,102,103")
    parser.add_argument("--regime", default="all", choices=["independent", "coordinated", "all"])
    parser.add_argument("--scenario", default="volatile", choices=shared_config.SCENARIOS)
    parser.add_argument("--ablation", default="full", choices=[
        "full", "no_neighbor_signal", "no_redundancy", "no_aoi", "no_energy",
        "no_agent_id", "no_coordination_constraint",
    ])
    parser.add_argument("--t_max", type=int, default=60000)
    args = parser.parse_args()
    algorithms = ["iql", "vdn", "qmix"] if args.alg == "all" else [args.alg]
    seeds = [args.seed] if args.seed is not None else [int(value) for value in args.seeds.split(",")]
    regimes = list(shared_config.REGIMES) if args.regime == "all" else [args.regime]
    summaries = train_suite(algorithms, seeds, regimes, args.t_max, args.scenario, args.ablation)
    failed = [item for item in summaries if item["status"] != "SUCCESS"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
