"""
Inference Latency Estimation Engine
Module C — TinyML Hardware Evaluation

Calculates estimated inference latency for each candidate microcontroller:
- Formula: t_latency = (Total_MACs * Cycles_per_MAC) / Clock_Frequency_Hz
- Reports Optimistic, Nominal, and Conservative latency bounds
- Cross-referenced against published CMSIS-NN and ESP-NN benchmark studies
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Cross-platform path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from hardware_eval.model_analysis import analyze_marl_policy

def estimate_inference_latencies(specs_path: str = "hardware_eval/device_specs.json") -> Dict[str, Any]:
    """
    Computes estimated inference latency with nominal, optimistic, and conservative bounds.
    """
    specs_file = Path(specs_path)
    with open(specs_file, "r") as f:
        db = json.load(f)

    devices = db["devices"]
    model_stats = analyze_marl_policy()
    total_macs = model_stats["total_macs_per_inference"]

    results = {}

    for dev_key, dev in devices.items():
        f_hz = dev["clock_speed_mhz"] * 1e6

        # Cycle bounds per Int8 MAC
        cycles_opt = dev["cycles_per_int8_mac_optimistic"]
        cycles_nom = dev["cycles_per_int8_mac_nominal"]
        cycles_cons = dev["cycles_per_int8_mac_conservative"]

        # Latency in milliseconds: (MACs * cycles / f_hz) * 1000
        latency_opt_ms = (total_macs * cycles_opt / f_hz) * 1000.0
        latency_nom_ms = (total_macs * cycles_nom / f_hz) * 1000.0
        latency_cons_ms = (total_macs * cycles_cons / f_hz) * 1000.0

        # Max inference throughput (inferences per second under nominal execution)
        throughput_hz = 1000.0 / max(0.001, latency_nom_ms)

        results[dev_key] = {
            "device_name": dev["name"],
            "clock_speed_mhz": dev["clock_speed_mhz"],
            "acceleration_kernel": dev["simd_library"],
            "estimated_latency_optimistic_ms": round(latency_opt_ms, 3),
            "estimated_latency_nominal_ms": round(latency_nom_ms, 3),
            "estimated_latency_conservative_ms": round(latency_cons_ms, 3),
            "nominal_throughput_inferences_per_sec": round(throughput_hz, 1),
            "methodology_note": "Estimated based on theoretical MAC instruction cycles cross-referenced with vendor DSP kernels"
        }

    return results

if __name__ == "__main__":
    lat_report = estimate_inference_latencies()
    print("=" * 65)
    print("ESTIMATED INFERENCE LATENCY REPORT ACROSS CANDIDATE MCUS")
    print("=" * 65)
    print(json.dumps(lat_report, indent=2))
