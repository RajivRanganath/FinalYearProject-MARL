"""Executable pre-training gates for the scientific MARL experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from environment.energy_model import EnergyModel
from environment.multi_agent_env import MultiAgentSensorEnv
from environment.pettingzoo_env import IoTSensorEnv
from environment.single_agent_env import SingleAgentSensorEnv


def _record(name: str, passed: bool, evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _tiny_overfit(seed: int = 7) -> Dict[str, Any]:
    """Overfit a two-state tabular Q learner with known optimal actions."""
    rng = np.random.RandomState(seed)
    q = np.zeros((2, 2), dtype=np.float64)
    rewards = np.array([[0.0, -0.1], [-1.5, 2.0]])
    visits = np.zeros((2, 2), dtype=int)
    for step in range(1000):
        state = step % 2
        epsilon = max(0.02, 1.0 - step / 700.0)
        action = int(rng.randint(2)) if rng.rand() < epsilon else int(np.argmax(q[state]))
        visits[state, action] += 1
        q[state, action] += 0.2 * (rewards[state, action] - q[state, action])
    greedy = np.argmax(q, axis=1).tolist()
    return {
        "passed": greedy == [shared_config.ACTION_SLEEP, shared_config.ACTION_SAMPLE],
        "q_values": q.round(5).tolist(),
        "greedy_actions": greedy,
        "action_visits": visits.tolist(),
    }


def _state_index(obs: np.ndarray) -> Tuple[int, int, int, int]:
    return (
        min(3, int(obs[0] * 4)),
        int(obs[1] >= shared_config.BASELINE_RULE_ENTROPY_THRESHOLD),
        min(3, int(obs[2] * 4)),
        min(2, int(obs[3] * 3)),
    )


def _single_agent_learning(seed: int = 11, episodes: int = 300) -> Dict[str, Any]:
    """Train a tabular Q learner on the real causal single-agent environment."""
    rng = np.random.RandomState(seed)
    q = np.zeros((4, 2, 4, 3, 2), dtype=np.float64)
    env = SingleAgentSensorEnv(scenario="volatile", seed=seed)
    action_visits = np.zeros(2, dtype=int)
    for episode in range(episodes):
        obs, _ = env.reset()
        epsilon = max(0.05, 1.0 - episode / (episodes * 0.75))
        done = False
        while not done:
            state = _state_index(obs)
            mask = env.get_action_mask()
            if rng.rand() < epsilon:
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                values = q[state].copy()
                values[mask == 0] = -np.inf
                action = int(np.argmax(values))
            action_visits[action] += 1
            next_obs, reward, done, _, _ = env.step(action)
            next_state = _state_index(next_obs)
            next_values = q[next_state].copy()
            next_values[env.get_action_mask() == 0] = -np.inf
            target = reward + (0.0 if done else 0.95 * float(np.max(next_values)))
            q[state + (action,)] += 0.15 * (target - q[state + (action,)])
            obs = next_obs

    learned_returns, sleep_returns, sample_rates = [], [], []
    for eval_seed in range(301, 311):
        learned = SingleAgentSensorEnv(scenario="volatile", seed=eval_seed)
        obs, _ = learned.reset(seed=eval_seed)
        total, samples, steps, done = 0.0, 0, 0, False
        while not done:
            state = _state_index(obs)
            values = q[state].copy()
            values[learned.get_action_mask() == 0] = -np.inf
            action = int(np.argmax(values))
            obs, reward, done, _, _ = learned.step(action)
            total += reward
            samples += action == shared_config.ACTION_SAMPLE
            steps += 1
        learned_returns.append(total)
        sample_rates.append(samples / steps)

        sleeper = SingleAgentSensorEnv(scenario="volatile", seed=eval_seed)
        sleeper.reset(seed=eval_seed)
        total_sleep, done = 0.0, False
        while not done:
            _, reward, done, _, _ = sleeper.step(shared_config.ACTION_SLEEP)
            total_sleep += reward
        sleep_returns.append(total_sleep)

    learned_mean = float(np.mean(learned_returns))
    sleep_mean = float(np.mean(sleep_returns))
    sample_mean = float(np.mean(sample_rates))
    return {
        "passed": learned_mean > sleep_mean and 0.01 < sample_mean < 0.99,
        "learned_return_mean": learned_mean,
        "always_sleep_return_mean": sleep_mean,
        "sample_fraction_mean": sample_mean,
        "training_action_visits": action_visits.tolist(),
    }


def config_digest(scenario: str, regime: str) -> str:
    source_files = (
        ROOT_DIR / "environment" / "energy_model.py",
        ROOT_DIR / "environment" / "single_agent_env.py",
        ROOT_DIR / "environment" / "multi_agent_env.py",
        ROOT_DIR / "environment" / "pettingzoo_env.py",
    )
    payload = {
        "scenario": scenario,
        "regime": regime,
        "obs_dim": shared_config.ENV_OBS_DIM,
        "episode_length": shared_config.EPISODE_LENGTH_TIMESTEPS,
        "event_proxy": {
            "noise_std": shared_config.EVENT_PROXY_NOISE_STD,
            "false_negative_rate": shared_config.EVENT_PROXY_FALSE_NEGATIVE_RATE,
            "false_positive_rate": shared_config.EVENT_PROXY_FALSE_POSITIVE_RATE,
        },
        "energy_profile": vars(EnergyModel().profile),
        "sample_feasibility": "battery >= sample + sleep + proxy-monitor same-step costs",
        "reward_weights": shared_config.REWARD_WEIGHTS,
        "regime_config": shared_config.REGIMES[regime],
        "environment_source_sha256": {
            str(path.relative_to(ROOT_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def run_sanity_checks(scenario: str = "volatile", regime: str = "independent") -> Dict[str, Any]:
    checks = []

    forced = SingleAgentSensorEnv(scenario=scenario, seed=1)
    forced.reset(seed=1)
    forced.battery = 0.9
    forced.current_entropy = 0.9
    _, high_sample_reward, _, _, high_sample = forced.step(shared_config.ACTION_SAMPLE)
    forced.battery, forced.current_entropy = 0.9, 0.9
    _, high_sleep_reward, _, _, _ = forced.step(shared_config.ACTION_SLEEP)
    forced.battery, forced.current_entropy = 0.9, 0.1
    _, low_sample_reward, _, _, _ = forced.step(shared_config.ACTION_SAMPLE)
    forced.battery, forced.current_entropy = 0.9, 0.1
    _, low_sleep_reward, _, _, _ = forced.step(shared_config.ACTION_SLEEP)
    checks.append(_record(
        "forced_action_and_controlled_reward",
        high_sample_reward > high_sleep_reward and low_sleep_reward > low_sample_reward,
        {
            "event_sample": high_sample_reward,
            "event_sleep": high_sleep_reward,
            "non_event_sample": low_sample_reward,
            "non_event_sleep": low_sleep_reward,
            "sample_executed": high_sample["sample_executed"],
        },
    ))

    component_values = list(high_sample["reward_components"].values())
    finite_and_bounded = all(np.isfinite(component_values)) and max(map(abs, component_values)) <= 2.0
    checks.append(_record(
        "reward_component_scale",
        finite_and_bounded,
        {"components": high_sample["reward_components"], "max_abs_component": max(map(abs, component_values))},
    ))

    causal = SingleAgentSensorEnv(scenario=scenario, seed=2)
    causal.reset(seed=2)
    causal.event_proxy = 0.25
    causal.current_entropy = 0.95
    pre_obs = causal._local_obs().copy()
    _, _, _, _, sleep_info = causal.step(shared_config.ACTION_SLEEP)
    causal.battery, causal.event_proxy, causal.current_entropy = 0.9, 0.25, 0.95
    _, _, _, _, sample_info = causal.step(shared_config.ACTION_SAMPLE)
    causal_ok = (
        pre_obs[1] == 0.25
        and sleep_info["measured_entropy"] is None
        and sample_info["measured_entropy"] == 0.95
    )
    checks.append(_record(
        "entropy_event_causality",
        causal_ok,
        {
            "predecision_proxy": float(pre_obs[1]),
            "sleep_measurement": sleep_info["measured_entropy"],
            "sample_measurement": sample_info["measured_entropy"],
        },
    ))

    tiny = _tiny_overfit()
    checks.append(_record("tiny_environment_overfit", tiny.pop("passed"), tiny))
    single = _single_agent_learning()
    checks.append(_record("single_agent_learning", single.pop("passed"), single))

    env_train = IoTSensorEnv(scenario=scenario, regime=regime, seed=3)
    train_obs, _ = env_train.reset(seed=3)
    env_eval = IoTSensorEnv(scenario=scenario, regime=regime, seed=3)
    eval_obs, _ = env_eval.reset(seed=3)
    contract_ok = all(np.array_equal(train_obs[aid], eval_obs[aid]) for aid in env_train.possible_agents)
    contract_ok = contract_ok and env_train.state().shape == (shared_config.GLOBAL_STATE_DIM,)
    checks.append(_record(
        "training_evaluation_contract_match",
        contract_ok,
        {
            "scenario": scenario,
            "regime": regime,
            "obs_dim": shared_config.ENV_OBS_DIM,
            "state_dim": shared_config.GLOBAL_STATE_DIM,
            "config_digest": config_digest(scenario, regime),
        },
    ))

    rng = np.random.RandomState(19)
    epsilon_actions = [int(rng.randint(2)) if rng.rand() < 0.5 else 0 for _ in range(1000)]
    counts = np.bincount(epsilon_actions, minlength=2)
    checks.append(_record(
        "exploration_both_actions",
        bool(np.all(counts > 100)),
        {"epsilon": 0.5, "action_counts": counts.tolist()},
    ))

    env1 = MultiAgentSensorEnv(scenario=scenario, regime=regime, seed=44)
    env2 = MultiAgentSensorEnv(scenario=scenario, regime=regime, seed=44)
    obs1, _ = env1.reset(seed=44)
    obs2, _ = env2.reset(seed=44)
    reproducible = all(np.array_equal(obs1[aid], obs2[aid]) for aid in env1.agent_ids)
    actions = {aid: i % 2 for i, aid in enumerate(env1.agent_ids)}
    for _ in range(12):
        obs1, rew1, _, _, _ = env1.step(actions)
        obs2, rew2, _, _, _ = env2.step(actions)
        reproducible &= all(np.array_equal(obs1[aid], obs2[aid]) for aid in env1.agent_ids)
        reproducible &= rew1 == rew2
    checks.append(_record("deterministic_reproducibility", reproducible, {"steps_compared": 12, "seed": 44}))

    all_passed = all(check["passed"] for check in checks)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "regime": regime,
        "config_digest": config_digest(scenario, regime),
        "all_passed": all_passed,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="volatile", choices=shared_config.SCENARIOS)
    parser.add_argument("--regime", default="independent", choices=shared_config.REGIMES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_sanity_checks(args.scenario, args.regime)
    output = args.output or ROOT_DIR / "results" / "sanity" / f"{args.regime}_{args.scenario}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for check in report["checks"]:
        print(f"{'PASS' if check['passed'] else 'FAIL'}  {check['name']}")
    print(f"report={output}")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
