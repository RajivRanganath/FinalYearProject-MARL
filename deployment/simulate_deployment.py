"""
Module C Stub — End-to-End Deployment Simulation with Full Metrics

This script bridges Module A (Environment) and Module B (ONNX Policy) to validate
the complete pipeline. It runs the trained MARL policy against the real IoTSensorEnv,
computes all metrics required by the Module B prompt (Phase 6), and compares against
the fixed-interval and rule-based baselines from Module A's baseline_results.md.

Metrics tracked:
  - Team reward (sum across agents per step)
  - Energy utilisation rate (% of timesteps where sample was executed)
  - Missed high-entropy events
  - Sample rejections (energy causality violations)
  - Simultaneous co-sampling overlap steps
  - Per-agent average battery level
"""

import os
import sys
import numpy as np
import onnxruntime as ort

# Add project root to sys.path to resolve imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import shared_config
from environment.pettingzoo_env import IoTSensorEnv


# ---------------------------------------------------------------------------
# Baseline policies (reimplemented here to run head-to-head in the same loop)
# ---------------------------------------------------------------------------

def fixed_interval_policy(agent_id, obs, step_count, info=None):
    """Sample every N timesteps regardless of state."""
    N = shared_config.BASELINE_FIXED_INTERVAL_N
    return shared_config.ACTION_SAMPLE if (step_count % N == 0) else shared_config.ACTION_SLEEP


def rule_based_policy(agent_id, obs, step_count, info=None):
    """If battery < 20%, sleep; otherwise sample."""
    residual_energy = obs[0]
    if residual_energy < shared_config.BASELINE_RULE_BATTERY_THRESHOLD:
        return shared_config.ACTION_SLEEP
    return shared_config.ACTION_SAMPLE


# ---------------------------------------------------------------------------
# ONNX MARL policy wrapper
# ---------------------------------------------------------------------------

class ONNXPolicy:
    """Wraps an ONNX model for per-agent inference with one-hot agent ID."""
    
    def __init__(self, onnx_path, n_agents, hidden_dim=64):
        self.session = ort.InferenceSession(onnx_path)
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
    
    def select_action(self, agent_id, obs, step_count=None, info=None):
        agent_idx = int(agent_id.split('_')[1])
        one_hot = np.zeros(self.n_agents, dtype=np.float32)
        one_hot[agent_idx] = 1.0
        
        full_obs = np.concatenate([obs, one_hot]).astype(np.float32)
        full_obs = np.expand_dims(full_obs, axis=0)
        hidden_in = np.zeros((1, self.hidden_dim), dtype=np.float32)
        
        outputs = self.session.run(None, {
            'obs': full_obs,
            'hidden_state_in': hidden_in
        })
        q_values = outputs[0][0]
        return int(np.argmax(q_values))


# ---------------------------------------------------------------------------
# Episode runner with full metric collection
# ---------------------------------------------------------------------------

def run_episode(env, policy_fn, scenario, seed):
    """
    Run a single episode and collect all Module B Phase 6 metrics.
    
    Args:
        env: IoTSensorEnv instance
        policy_fn: Callable(agent_id, obs, step_count, info) -> action
        scenario: "stable" or "volatile"
        seed: random seed
    
    Returns:
        dict of collected metrics
    """
    obs_dict, info_dict = env.reset(seed=seed)
    
    agent_ids = env.possible_agents
    n_agents = len(agent_ids)
    
    total_team_reward = 0.0
    total_samples_executed = 0
    total_samples_requested = 0
    total_rejections = 0
    total_missed_events = 0
    total_high_entropy_events = 0
    overlap_steps = {2: 0, 3: 0, 4: 0}
    agent_batteries = {aid: [] for aid in agent_ids}
    agent_rewards = {aid: 0.0 for aid in agent_ids}
    
    terminated = False
    truncated = False
    step_count = 0
    
    while not (terminated or truncated):
        step_count += 1
        actions = {}
        for agent_id in agent_ids:
            actions[agent_id] = policy_fn(agent_id, obs_dict[agent_id], step_count, info_dict.get(agent_id))
            if actions[agent_id] == shared_config.ACTION_SAMPLE:
                total_samples_requested += 1
        
        obs_dict, rewards_dict, terminations, truncations, info_dict = env.step(actions)
        
        # Collect per-step metrics
        samplers_this_step = 0
        for agent_id in agent_ids:
            info = info_dict[agent_id]
            agent_rewards[agent_id] += rewards_dict[agent_id]
            agent_batteries[agent_id].append(info["battery"])
            
            if info["action_executed"] == shared_config.ACTION_SAMPLE and not info["sample_rejected"]:
                total_samples_executed += 1
                samplers_this_step += 1
            
            if info["sample_rejected"]:
                total_rejections += 1
            
            if info["is_high_entropy"]:
                total_high_entropy_events += 1
                if info["action_executed"] == shared_config.ACTION_SLEEP:
                    total_missed_events += 1
        
        if samplers_this_step in overlap_steps:
            overlap_steps[samplers_this_step] += 1
        
        total_team_reward += sum(rewards_dict.values())
        terminated = all(terminations.values())
        truncated = all(truncations.values())
    
    # Compute derived metrics
    total_possible_samples = step_count * n_agents
    energy_util_rate = (total_samples_executed / total_possible_samples * 100) if total_possible_samples > 0 else 0
    avg_battery = {aid: float(np.mean(agent_batteries[aid])) for aid in agent_ids}
    overall_avg_battery = float(np.mean([avg_battery[aid] for aid in agent_ids]))
    total_overlap_steps = sum(count for k, count in overlap_steps.items())
    
    return {
        "team_reward": total_team_reward,
        "avg_battery": overall_avg_battery,
        "energy_util_rate": energy_util_rate,
        "samples_executed": total_samples_executed,
        "rejections": total_rejections,
        "missed_events": total_missed_events,
        "high_entropy_events": total_high_entropy_events,
        "overlap_steps": total_overlap_steps,
        "overlap_detail": overlap_steps,
        "steps": step_count,
        "per_agent_reward": agent_rewards,
        "per_agent_avg_battery": avg_battery,
    }


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

def print_comparison_table(scenario, results):
    """Print a formatted comparison table matching baseline_results.md format."""
    print(f"\n{'='*85}")
    print(f"  {scenario.upper()} SCENARIO — Policy Comparison")
    print(f"{'='*85}")
    print(f"{'Policy':<20} {'Team Reward':>12} {'Avg Battery':>12} {'Rejections':>11} {'Missed':>8} {'Overlap':>9} {'Energy%':>9}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*11} {'-'*8} {'-'*9} {'-'*9}")
    
    for name, r in results.items():
        print(f"{name:<20} {r['team_reward']:>12.2f} {r['avg_battery']:>12.4f} "
              f"{r['rejections']:>11d} {r['missed_events']:>8d} "
              f"{r['overlap_steps']:>9d} {r['energy_util_rate']:>8.1f}%")
    print()


def run_full_comparison(onnx_path, episodes=3):
    """Run all policies across both scenarios and print comparison tables."""
    
    marl_policy = ONNXPolicy(onnx_path, shared_config.NUM_AGENTS)
    
    policies = {
        "QMIX (Trained)": marl_policy.select_action,
        "Fixed-Interval": fixed_interval_policy,
        "Rule-Based": rule_based_policy,
    }
    
    for scenario in ["stable", "volatile"]:
        scenario_results = {}
        
        for policy_name, policy_fn in policies.items():
            # Average metrics over multiple episodes for robustness
            all_episode_results = []
            for ep in range(episodes):
                env = IoTSensorEnv(scenario=scenario, seed=shared_config.SEED + ep)
                result = run_episode(env, policy_fn, scenario, seed=shared_config.SEED + ep)
                all_episode_results.append(result)
            
            # Average the scalar metrics
            avg_result = {
                "team_reward": np.mean([r["team_reward"] for r in all_episode_results]),
                "avg_battery": np.mean([r["avg_battery"] for r in all_episode_results]),
                "energy_util_rate": np.mean([r["energy_util_rate"] for r in all_episode_results]),
                "rejections": int(np.mean([r["rejections"] for r in all_episode_results])),
                "missed_events": int(np.mean([r["missed_events"] for r in all_episode_results])),
                "overlap_steps": int(np.mean([r["overlap_steps"] for r in all_episode_results])),
                "high_entropy_events": int(np.mean([r["high_entropy_events"] for r in all_episode_results])),
                "samples_executed": int(np.mean([r["samples_executed"] for r in all_episode_results])),
            }
            scenario_results[policy_name] = avg_result
        
        print_comparison_table(scenario, scenario_results)
        
        # Print improvement analysis
        marl = scenario_results["QMIX (Trained)"]
        fixed = scenario_results["Fixed-Interval"]
        rule = scenario_results["Rule-Based"]
        
        print(f"  QMIX vs Fixed-Interval:")
        print(f"    Energy Savings: {fixed['energy_util_rate'] - marl['energy_util_rate']:+.1f}% utilisation difference")
        print(f"    Missed Events: {marl['missed_events']} vs {fixed['missed_events']} (QMIX {'better' if marl['missed_events'] < fixed['missed_events'] else 'worse'})")
        print(f"    Reward Improvement: {marl['team_reward'] - fixed['team_reward']:+.2f}")
        
        print(f"\n  QMIX vs Rule-Based:")
        print(f"    Energy Savings: {rule['energy_util_rate'] - marl['energy_util_rate']:+.1f}% utilisation difference")
        print(f"    Missed Events: {marl['missed_events']} vs {rule['missed_events']} (QMIX {'better' if marl['missed_events'] < rule['missed_events'] else 'worse'})")
        print(f"    Reward Improvement: {marl['team_reward'] - rule['team_reward']:+.2f}")
        print()


if __name__ == "__main__":
    onnx_file = os.path.join(project_root, "training", "policy.onnx")
    if not os.path.exists(onnx_file):
        print(f"Error: ONNX file not found at {onnx_file}")
        sys.exit(1)
    
    print("=" * 85)
    print("  MODULE B — FULL DEPLOYMENT SIMULATION & BASELINE COMPARISON")
    print("  ONNX Policy: training/policy.onnx")
    print("  Environment: environment/pettingzoo_env.IoTSensorEnv (Real)")
    print("=" * 85)
    
    run_full_comparison(onnx_file, episodes=5)
