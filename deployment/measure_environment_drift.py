"""Measure how far the repaired sample-feasibility rule moves published results.

Every training run and every evaluation in this repository predates the
`single_agent_env` repair that charges the unavoidable same-step sleep and
proxy-monitor energy before declaring SAMPLE feasible.  The repair is correct,
but it means the committed source is not the source that produced the committed
numbers.  Rather than assert that the difference is small, this script measures
it on the real multi-agent environment under the policy that stresses the
battery floor hardest: request SAMPLE whenever the mask allows it.

The output is written to `results/environment_drift_impact.json` and is cited by
`results/environment_drift.json`, `LIMITATIONS.md`, and the final report.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config
from environment.multi_agent_env import MultiAgentSensorEnv
from environment.single_agent_env import SingleAgentSensorEnv

REPAIRED_MASK = SingleAgentSensorEnv.get_action_mask


def _legacy_mask(self: SingleAgentSensorEnv) -> np.ndarray:
    """The pre-repair rule: ignore the same-step background draw."""
    can_sample = int(self.battery >= self.energy_model.sample_energy_cost)
    return np.array([1, can_sample], dtype=np.int8)


def _rollout(seed: int, regime: str) -> tuple[float, int]:
    """Run one battery-floor-stressing episode; return reward and band hits."""
    env = MultiAgentSensorEnv(scenario="volatile", regime=regime, seed=seed)
    env.reset(seed=seed)
    total = 0.0
    band_hits = 0
    while True:
        available = env.get_avail_actions()
        for agent_id in env.agent_ids:
            agent = env.agents[agent_id]
            low = agent.energy_model.sample_energy_cost
            high = agent.energy_model.sample_step_energy_cost
            if low <= agent.battery < high:
                band_hits += 1
        actions = {
            agent_id: (
                shared_config.ACTION_SAMPLE
                if available[agent_id][1] == 1
                else shared_config.ACTION_SLEEP
            )
            for agent_id in env.agent_ids
        }
        _, rewards, terminations, truncations, _ = env.step(actions)
        total += float(sum(rewards.values()))
        if all(terminations.values()) or all(truncations.values()):
            return total, band_hits


def _measure(regime: str, seeds: List[int]) -> Dict[str, Any]:
    legacy: List[float] = []
    repaired: List[float] = []
    band_hits = 0
    for seed in seeds:
        SingleAgentSensorEnv.get_action_mask = _legacy_mask
        legacy_reward, hits = _rollout(seed, regime)
        SingleAgentSensorEnv.get_action_mask = REPAIRED_MASK
        repaired_reward, _ = _rollout(seed, regime)
        legacy.append(legacy_reward)
        repaired.append(repaired_reward)
        band_hits += hits
    deltas = np.asarray(repaired) - np.asarray(legacy)
    agent_steps = len(seeds) * shared_config.EPISODE_LENGTH_TIMESTEPS * shared_config.NUM_AGENTS
    return {
        "regime": regime,
        "seeds": seeds,
        "episode_agent_steps": agent_steps,
        "agent_steps_inside_disagreement_band": band_hits,
        "disagreement_band_fraction": band_hits / agent_steps,
        "mean_reward_delta_repaired_minus_legacy": float(deltas.mean()),
        "max_abs_reward_delta": float(np.abs(deltas).max()),
        "seeds_with_any_difference": int((deltas != 0).sum()),
        "legacy_mean_reward": float(np.mean(legacy)),
        "repaired_mean_reward": float(np.mean(repaired)),
    }


def measure() -> Path:
    try:
        from environment.energy_model import EnergyModel

        model = EnergyModel()
        seeds = list(shared_config.TEST_SEEDS)
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "probe_policy": "request SAMPLE whenever the action mask allows it",
            "why_this_policy": (
                "It saturates the battery floor, so it is an upper bound on how often the "
                "legacy and repaired feasibility rules can disagree. Learned policies sample "
                "far less often and therefore drift less."
            ),
            "legacy_rule": "battery >= sample_energy_cost",
            "repaired_rule": "battery >= sample_energy_cost + sleep_energy_cost + proxy_monitor_energy_cost",
            "sample_energy_cost": model.sample_energy_cost,
            "background_step_energy_cost": model.background_step_energy_cost,
            "disagreement_band_width": model.sample_step_energy_cost - model.sample_energy_cost,
            "regimes": [_measure(regime, seeds) for regime in ("independent", "coordinated")],
        }
    finally:
        SingleAgentSensorEnv.get_action_mask = REPAIRED_MASK

    output = ROOT_DIR / "results" / "environment_drift_impact.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return output


if __name__ == "__main__":
    measure()
