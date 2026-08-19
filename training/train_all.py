"""Isolated, sanity-gated, multi-seed IQL/VDN/QMIX training pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

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
from training.training_profiles import PROFILES, get_training_profile
from training.refined_protocol import REFINED_FINAL_SEEDS


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
    include_agent_id: bool = True,
    mask_neighbor_signal: bool = False,
) -> Dict[str, Any]:
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
                decision_obs = np.asarray(obs[aid], dtype=np.float32).copy()
                if mask_neighbor_signal:
                    decision_obs[shared_config.STATE_INDEX_NEIGHBOR_SAMPLING_RATE] = 0.0
                action = policy.select_action(aid, decision_obs, infos[aid])
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


def _resume_checkpoint_root(
    resume_profile: str,
    regime: str,
    algorithm: str,
    seed: int,
    scenario: str = "volatile",
    expected_config_digest: Optional[str] = None,
) -> tuple[Path, int]:
    """Resolve a completed run without silently choosing by test performance."""
    root = ROOT_DIR / "results" / "upgrade_experiments"
    manifest = root / f"training_manifest_{resume_profile}_full.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Resume manifest not found: {manifest}")
    matches = [
        item for item in json.loads(manifest.read_text())
        if item.get("status") == "SUCCESS"
        and item.get("profile") == resume_profile
        and item.get("regime") == regime
        and item.get("scenario") == scenario
        and item.get("algorithm") == algorithm
        and item.get("seed") == seed
        and item.get("ablation", "full") == "full"
        and (
            expected_config_digest is None
            or item.get("config_digest") == expected_config_digest
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one completed {resume_profile}/{regime}/{algorithm}/seed{seed} "
            f"manifest entry, found {len(matches)}"
        )
    selected = Path(matches[0]["selected_checkpoint"]["checkpoint"])
    if not selected.is_absolute():
        selected = ROOT_DIR / selected
    checkpoint_root = selected.parent
    required = {"agent.th", "opt.th"}
    if algorithm in {"vdn", "qmix"}:
        required.add("mixer.th")
    missing = sorted(name for name in required if not (selected / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"Selected resume checkpoint is incomplete ({', '.join(missing)} missing): {selected}"
        )
    return checkpoint_root, int(selected.name)


TRAINING_SOURCE_FILES = (
    "training/train_all.py",
    "training/training_profiles.py",
    "training/sanity_checks.py",
    "training/epymarl/src/learners/q_learner.py",
    "training/epymarl/src/controllers/basic_controller.py",
    "training/epymarl/src/runners/episode_runner.py",
    "training/epymarl/src/modules/agents/rnn_agent.py",
    "training/epymarl/src/modules/mixers/qmix.py",
    "training/epymarl/src/modules/mixers/vdn.py",
)


def _training_source_digest() -> Dict[str, str]:
    """Fingerprint the code that defines training semantics, not just the SHA."""
    return {
        relative: _sha256(ROOT_DIR / relative)
        for relative in TRAINING_SOURCE_FILES
        if (ROOT_DIR / relative).is_file()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optimizer_learning_rates(checkpoint: Path) -> List[float]:
    state = torch.load(checkpoint / "opt.th", map_location="cpu", weights_only=True)
    groups = state.get("param_groups", [])
    rates = [float(group["lr"]) for group in groups if "lr" in group]
    if not rates:
        raise RuntimeError(f"No optimizer learning rate found in {checkpoint / 'opt.th'}")
    return rates


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def _export_directory(
    profile: str, scenario: str, regime: str, ablation: str
) -> Path:
    """Keep historical volatile/full paths while isolating every other contract."""
    if profile != "baseline":
        base = ROOT_DIR / "results" / "upgrade_models" / profile
        if scenario == "volatile" and ablation == "full":
            return base / regime
        return base / scenario / regime / ablation
    if ablation == "full":
        base = ROOT_DIR / "results" / "learned_models"
        return base / regime if scenario == "volatile" else base / scenario / regime
    base = ROOT_DIR / "results" / "ablation_models" / ablation
    return base / regime if scenario == "volatile" else base / scenario / regime


def run_training_experiment(
    algorithm: str,
    seed: int,
    t_max: int,
    scenario: str = "volatile",
    regime: str = "independent",
    ablation: str = "full",
    lr: Optional[float] = None,
    profile: str = "baseline",
    resume_profile: Optional[str] = None,
    sanity_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if algorithm not in {"iql", "vdn", "qmix"}:
        raise ValueError(f"Unsupported algorithm {algorithm}")
    settings = get_training_profile(profile)
    effective_lr = settings.learning_rate if lr is None else lr
    include_agent_id = settings.include_agent_id and ablation != "no_agent_id"
    mask_neighbor_signal = settings.mask_neighbor_signal or ablation == "no_neighbor_signal"
    resume_root: Optional[Path] = None
    resume_step: Optional[int] = None
    gate = _sanity_gate(scenario, regime, sanity_report)
    if resume_profile is not None:
        source_settings = get_training_profile(resume_profile)
        if (
            source_settings.include_agent_id != settings.include_agent_id
            or source_settings.mask_neighbor_signal != settings.mask_neighbor_signal
            or source_settings.hidden_dim != settings.hidden_dim
        ):
            raise ValueError("Resume profile must preserve input masking, agent identity, and hidden size")
        if ablation != "full":
            raise ValueError("Automatic profile continuation currently supports only the full model")
        resume_root, resume_step = _resume_checkpoint_root(
            resume_profile,
            regime,
            algorithm,
            seed,
            scenario=scenario,
            expected_config_digest=gate["config_digest"],
        )
        if t_max <= resume_step:
            raise ValueError(
                f"Continuation t_max={t_max} must be greater than source step {resume_step}"
            )
    source_git_sha = _git_sha()
    source_git_dirty = _git_dirty()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    experiment_root = "experiments" if profile == "baseline" else "upgrade_experiments"
    run_dir = ROOT_DIR / "results" / experiment_root / regime / algorithm / f"seed{seed}" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    map_name = f"iot_{regime}"
    sacred_root = EPYMARL_SRC.parent / "results" / "sacred" / algorithm / map_name
    sacred_before = set(path.parent for path in sacred_root.glob("*/run.json")) if sacred_root.exists() else set()

    save_interval = max(
        shared_config.EPISODE_LENGTH_TIMESTEPS,
        t_max // settings.checkpoint_count,
    )
    anneal = max(
        shared_config.EPISODE_LENGTH_TIMESTEPS,
        int(t_max * settings.epsilon_anneal_fraction),
    )
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        f"--config={algorithm}",
        "--env-config=iot",
        "with",
        f"seed={seed}",
        f"t_max={t_max}",
        f"lr={effective_lr}",
        f"gamma={settings.gamma}",
        "use_cuda=False",
        f"batch_size={settings.batch_size}",
        f"buffer_size={settings.buffer_size}",
        f"hidden_dim={settings.hidden_dim}",
        f"updates_per_episode={settings.updates_per_episode}",
        f"target_update_interval_or_tau={settings.target_update_interval_or_tau}",
        f"epsilon_anneal_time={anneal}",
        f"epsilon_finish={settings.epsilon_finish}",
        "save_model=True",
        f"save_model_interval={save_interval}",
        f"test_interval={save_interval}",
        f"env_args.scenario={scenario}",
        f"env_args.regime={regime}",
        f"env_args.ablation={ablation}",
        f"env_args.mask_neighbor_signal={mask_neighbor_signal}",
        f"env_args.map_name={map_name}",
        f"obs_agent_id={include_agent_id}",
        f"local_results_path={run_dir}",
    ]
    if resume_root is not None:
        cmd.extend([f"checkpoint_path={resume_root}", f"load_step={resume_step}"])
    config_snapshot = {
        "algorithm": algorithm,
        "seed": seed,
        "t_max": t_max,
        "profile": profile,
        "training_profile": settings.to_dict(),
        "learning_rate": effective_lr,
        "include_agent_id": include_agent_id,
        "mask_neighbor_signal": mask_neighbor_signal,
        "scenario": scenario,
        "regime": regime,
        "ablation": ablation,
        "validation_seeds": shared_config.VAL_SEEDS,
        "test_seeds_locked": (
            shared_config.TEST_SEEDS
            if profile == "baseline"
            else (
                shared_config.V2_TEST_SEEDS
                if profile in {"improved_v2", "extended"}
                else (
                    REFINED_FINAL_SEEDS
                    if profile == "refined"
                    else shared_config.UPGRADE_TEST_SEEDS
                )
            )
        ),
        "config_digest": gate["config_digest"],
        "git_sha": source_git_sha,
        "git_worktree_dirty": source_git_dirty,
        # ``git_sha`` alone cannot identify the code that ran, because these
        # runs are launched from a disclosed dirty worktree.  Hash the learner
        # and launcher sources directly so a later audit can tell which
        # training semantics produced a checkpoint instead of inferring it from
        # file modification times.
        "training_source_sha256": _training_source_digest(),
        "resume_profile": resume_profile,
        "resume_checkpoint_root": str(resume_root) if resume_root else None,
        "resume_checkpoint_step": resume_step,
        "resume_checkpoint_sha256": (
            {
                name: _sha256(resume_root / str(resume_step) / name)
                for name in sorted(
                    {"agent.th", "opt.th"}
                    | ({"mixer.th"} if algorithm in {"vdn", "qmix"} else set())
                )
            }
            if resume_root is not None and resume_step is not None
            else None
        ),
        "continuation_semantics": (
            "checkpoint warm-start with restored online weights and optimizer moments; "
            "fresh replay buffer, target-network bootstrap, learner counters, and RNG stream"
            if resume_root is not None
            else None
        ),
        "command": cmd,
        "checkpoint_rule": "maximum mean team reward on validation seeds only",
    }
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2) + "\n")
    (run_dir / "sanity_gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    print(f"\nTRAIN {algorithm.upper()} regime={regime} seed={seed} t_max={t_max}")
    started = time.time()
    try:
        process = subprocess.run(cmd, cwd=ROOT_DIR)
    except KeyboardInterrupt:
        elapsed = time.time() - started
        summary = {**config_snapshot, "status": "ABORTED", "elapsed_seconds": round(elapsed, 3)}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise
    except Exception as exc:
        elapsed = time.time() - started
        summary = {
            **config_snapshot,
            "status": "LAUNCH_FAILED",
            "elapsed_seconds": round(elapsed, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
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
        summary = {
            **config_snapshot,
            "status": "POSTPROCESS_FAILED",
            "elapsed_seconds": round(elapsed, 3),
            "error": f"Training succeeded but no checkpoints were written under {checkpoint_root}",
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    decisions = [
        _evaluate_checkpoint(
            path,
            scenario,
            regime,
            ablation,
            shared_config.VAL_SEEDS,
            include_agent_id=include_agent_id,
            mask_neighbor_signal=mask_neighbor_signal,
        )
        for path in checkpoints
    ]
    selected = max(decisions, key=lambda item: item["mean_team_reward"])
    selected_path = Path(selected["checkpoint"])
    selected_lrs = _optimizer_learning_rates(selected_path)
    selected["optimizer_learning_rates"] = selected_lrs
    selected["optimizer_lr_matches_config"] = bool(
        all(np.isclose(value, effective_lr, rtol=0.0, atol=1e-12) for value in selected_lrs)
    )
    (run_dir / "checkpoint_validation.json").write_text(json.dumps(decisions, indent=2) + "\n")
    if not selected["optimizer_lr_matches_config"]:
        summary = {
            **config_snapshot,
            "status": "POSTPROCESS_FAILED",
            "elapsed_seconds": round(elapsed, 3),
            "error": (
                f"Selected checkpoint optimizer LR {selected_lrs} does not match "
                f"configured LR {effective_lr}"
            ),
            "run_dir": _portable_path(run_dir),
            "selected_checkpoint": selected,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    # Ablations are first-class retrained policies.  Keep them outside the
    # canonical model directory so a study can never overwrite the full model.
    exported_dir = _export_directory(profile, scenario, regime, ablation)
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
        "run_dir": _portable_path(run_dir),
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
    profile: str = "baseline",
    resume_profile: Optional[str] = None,
    lr: Optional[float] = None,
) -> List[Dict[str, Any]]:
    reports = {regime: _sanity_gate(scenario, regime) for regime in regimes}
    summaries: List[Dict[str, Any]] = []
    for regime in regimes:
        for algorithm in algorithms:
            for seed in seeds:
                summary = run_training_experiment(
                    algorithm=algorithm,
                    seed=seed,
                    t_max=t_max,
                    scenario=scenario,
                    regime=regime,
                    ablation=ablation,
                    profile=profile,
                    resume_profile=resume_profile,
                    lr=lr,
                    sanity_report=reports[regime],
                )
                summaries.append(summary)
                # Persist each terminal result immediately.  A later failure or
                # interrupt must not erase evidence from completed replicas.
                write_training_manifest([summary], profile=profile, ablation=ablation)
    return summaries


def _merge_training_summaries(
    existing: List[Dict[str, Any]], new: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Replace matching replicas while retaining other completed suite seeds."""
    keyed: Dict[tuple, Dict[str, Any]] = {}
    for item in [*existing, *new]:
        key = (
            item.get("profile", "baseline"),
            item.get("ablation", "full"),
            item.get("scenario", "volatile"),
            item.get("regime"),
            item.get("algorithm"),
            item.get("seed"),
            item.get("config_digest"),
        )
        keyed[key] = item
    return [keyed[key] for key in sorted(keyed, key=lambda value: tuple(map(str, value)))]


def write_training_manifest(
    summaries: List[Dict[str, Any]], profile: str, ablation: str
) -> Path:
    manifest_root = ROOT_DIR / "results" / (
        "experiments" if profile == "baseline" else "upgrade_experiments"
    )
    manifest_name = (
        f"training_manifest_{ablation}.json"
        if profile == "baseline"
        else f"training_manifest_{profile}_{ablation}.json"
    )
    manifest = manifest_root / manifest_name
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output = summaries
    if manifest.exists():
        output = _merge_training_summaries(json.loads(manifest.read_text()), summaries)
    manifest.write_text(json.dumps(output, indent=2) + "\n")
    return manifest


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
    parser.add_argument("--profile", default="baseline", choices=sorted(PROFILES))
    parser.add_argument(
        "--resume-profile",
        choices=sorted(PROFILES),
        help="Resume each replica from that profile's validation-selected checkpoint",
    )
    parser.add_argument("--lr", type=float, help="Override the selected profile learning rate")
    parser.add_argument("--t_max", type=int)
    args = parser.parse_args()
    algorithms = ["iql", "vdn", "qmix"] if args.alg == "all" else [args.alg]
    seeds = [args.seed] if args.seed is not None else [int(value) for value in args.seeds.split(",")]
    regimes = list(shared_config.REGIMES) if args.regime == "all" else [args.regime]
    t_max = args.t_max or get_training_profile(args.profile).recommended_t_max
    summaries = train_suite(
        algorithms,
        seeds,
        regimes,
        t_max,
        args.scenario,
        args.ablation,
        profile=args.profile,
        resume_profile=args.resume_profile,
        lr=args.lr,
    )
    failed = [item for item in summaries if item["status"] != "SUCCESS"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
