"""
Hardware Energy-Per-Inference and System Energy Budget Engine
Module C — TinyML Hardware Evaluation

Calculates electrical energy consumed per single MARL policy inference:
- Formula: E_inference = Voltage_V * Current_Active_A * Latency_Seconds
- Compares MARL decision energy against sensor acquisition and RF transmission costs
- Quantifies Energy Return on Investment (ROI): Energy saved by sleeping vs. decision cost
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Cross-platform path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from hardware_eval.latency_estimator import estimate_inference_latencies
from environment.energy_model import PhysicalEnergyProfile

def evaluate_hardware_energy(specs_path: str = "hardware_eval/device_specs.json") -> Dict[str, Any]:
    """
    Computes energy per inference across candidate microcontrollers and compares with system budget.
    """
    specs_file = Path(specs_path)
    with open(specs_file, "r") as f:
        db = json.load(f)

    devices = db["devices"]
    latency_report = estimate_inference_latencies(specs_path)
    profile = PhysicalEnergyProfile()

    # System benchmark costs (Joules)
    sensor_energy_joules = profile.voltage_volts * (profile.sensor_current_ma * 1e-3) * profile.sensor_duration_seconds # ~0.000173 J (0.173 mJ)
    radio_energy_joules = profile.voltage_volts * (profile.radio_tx_current_ma * 1e-3) * profile.radio_tx_duration_seconds # ~0.01386 J (13.86 mJ)
    total_sampling_action_energy_joules = sensor_energy_joules + radio_energy_joules # ~0.01403 J (14.03 mJ)

    results = {}

    for dev_key, dev in devices.items():
        v = dev["voltage_volts"]
        i_active_a = dev["active_current_ma"] * 1e-3
        t_nom_s = latency_report[dev_key]["estimated_latency_nominal_ms"] * 1e-3

        # E_inf = V * I * t (Joules)
        e_inf_joules = v * i_active_a * t_nom_s
        e_inf_uj = e_inf_joules * 1e6
        e_inf_mj = e_inf_joules * 1e3

        # Comparison ratios
        inf_to_sample_cost_pct = (e_inf_joules / total_sampling_action_energy_joules) * 100.0
        energy_roi_ratio = total_sampling_action_energy_joules / max(1e-9, e_inf_joules)

        results[dev_key] = {
            "device_name": dev["name"],
            "active_power_mw": round(v * dev["active_current_ma"], 2),
            "estimated_latency_nominal_ms": latency_report[dev_key]["estimated_latency_nominal_ms"],
            "energy_per_inference_microjoules": round(e_inf_uj, 2),
            "energy_per_inference_millijoules": round(e_inf_mj, 5),
            "inference_energy_as_pct_of_sample_action": round(inf_to_sample_cost_pct, 3),
            "energy_savings_roi_per_sleep_decision": f"{round(energy_roi_ratio, 1)}x net energy saved per avoided sample",
            "is_energy_negligible": (inf_to_sample_cost_pct < 2.0)
        }

    return results

if __name__ == "__main__":
    energy_report = evaluate_hardware_energy()
    print("=" * 65)
    print("ENERGY PER INFERENCE & SYSTEM BUDGET REPORT")
    print("=" * 65)
    print(json.dumps(energy_report, indent=2))
