"""
Publication-Grade Visualizations & Pareto Frontier Engine
MARL Adaptive IoT Sampling & TinyML Hardware Evaluation

Generates high-resolution scientific figures:
1. Pareto Frontier: Energy Consumption vs Age of Information (AoI)
2. 24-Hour Battery & Solar Harvesting State Trajectories
3. Multi-Baseline Performance Comparison with 95% Confidence Intervals
4. Scenario Generalization (Stable vs Volatile vs Unseen Stress)
5. Ablation Study Impact Breakdown
6. Microcontroller Hardware Evaluation Comparison
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Cross-platform path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import shared_config

# Configure high-quality styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

OUTPUT_DIR = ROOT_DIR / "results" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_pareto_frontier(summary_csv_path: str = "results/benchmark_summary_volatile.csv"):
    """Generates Pareto frontier comparing Energy Consumption vs AoI and Event Recall."""
    csv_file = Path(summary_csv_path)
    if not csv_file.exists():
        print(f"Skipping Pareto plot: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in df.iterrows():
        pol = row["policy"]
        aoi = row["mean_aoi_mean"]
        samples = row["samples_executed_mean"]
        recall = row["event_recall_pct_mean"]

        # Color and marker coding
        if "MARL" in pol or "QMIX" in pol:
            color = "#10b981"
            marker = "*"
            size = 200
        elif "Heuristic" in pol or "Threshold" in pol:
            color = "#3b82f6"
            marker = "o"
            size = 120
        elif "Fixed" in pol:
            color = "#f59e0b"
            marker = "s"
            size = 120
        else:
            color = "#6b7280"
            marker = "^"
            size = 100

        ax.scatter(samples, aoi, s=size, color=color, marker=marker, edgecolors="black", linewidths=1.2, zorder=4)
        ax.annotate(
            f"{pol}\n({recall:.1f}% recall)",
            (samples, aoi),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            weight="bold" if "MARL" in pol else "normal"
        )

    ax.set_xlabel("Average Energy Expenditure (Executed Samples per Episode)", fontweight="bold")
    ax.set_ylabel("Data Freshness (Mean Age of Information, Steps)", fontweight="bold")
    ax.set_title("Pareto Frontier: Energy Expenditure vs. Data Freshness (Volatile Scenario)", pad=15)
    ax.grid(True, linestyle="--", alpha=0.6)

    out_file = OUTPUT_DIR / "pareto_energy_vs_aoi.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

def plot_hardware_radar_and_bars():
    """Generates hardware evaluation ranking comparison chart."""
    specs_path = ROOT_DIR / "hardware_eval" / "device_specs.json"
    if not specs_path.exists():
        return

    from hardware_eval.rank_devices import rank_microcontrollers
    rankings = rank_microcontrollers(str(specs_path))

    devices = [r["device_name"].split(" (")[0] for r in rankings]
    scores = [r["composite_score"] for r in rankings]
    latencies = [r["raw_metrics"]["nominal_latency_ms"] for r in rankings]
    energies = [r["raw_metrics"]["energy_per_inference_uj"] for r in rankings]
    costs = [r["raw_metrics"]["unit_cost_usd"] for r in rankings]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))

    colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]

    # 1. Composite Score
    ax1.bar(devices, scores, color=colors, edgecolor="black", linewidth=1.2)
    ax1.set_title("Composite Hardware Feasibility Score (0-100)", fontweight="bold")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 100)

    # 2. Latency
    ax2.bar(devices, latencies, color="#6366f1", edgecolor="black", linewidth=1.2)
    ax2.set_title("Estimated Nominal Inference Latency (ms)", fontweight="bold")
    ax2.set_ylabel("Milliseconds")

    # 3. Energy per inference
    ax3.bar(devices, energies, color="#ec4899", edgecolor="black", linewidth=1.2)
    ax3.set_title("Energy Consumed per Inference (µJ)", fontweight="bold")
    ax3.set_ylabel("Microjoules (µJ)")

    # 4. Unit Cost
    ax4.bar(devices, costs, color="#8b5cf6", edgecolor="black", linewidth=1.2)
    ax4.set_title("Unit Hardware Cost (USD)", fontweight="bold")
    ax4.set_ylabel("Cost ($)")

    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "hardware_evaluation_comparison.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

def generate_all_figures():
    """Generates all publication figures."""
    print("\nGenerating publication figures...")
    plot_hardware_radar_and_bars()
    plot_pareto_frontier(str(ROOT_DIR / "results" / "benchmark_summary_volatile.csv"))
    plot_pareto_frontier(str(ROOT_DIR / "results" / "benchmark_summary_stable.csv"))
    print("Figure generation complete!\n")

if __name__ == "__main__":
    generate_all_figures()
