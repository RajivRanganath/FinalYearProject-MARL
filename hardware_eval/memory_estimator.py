"""
Memory Footprint & SRAM/Flash Feasibility Estimator
Module C — TinyML Hardware Evaluation

Calculates static and dynamic memory requirements for each candidate microcontroller:
- Model Weights (Flash / ROM)
- Tensor Arena & Activation Buffers (SRAM)
- TFLite Micro Interpreter Runtime Overhead (Flash & SRAM)
- Feasibility Gate: Flags any device where memory is exceeded
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

# Planning assumptions only. A real converted graph/build must replace these.
TFLM_FLASH_RUNTIME_OVERHEAD_KB = 120.0  # Core TFLite Micro runtime code footprint
TFLM_SRAM_INTERPRETER_OVERHEAD_KB = 16.0 # Core tensor arena & interpreter working memory

def evaluate_memory_footprint(specs_path: str = "hardware_eval/device_specs.json") -> Dict[str, Any]:
    """
    Evaluates memory feasibility across all candidate microcontrollers.
    """
    specs_file = Path(specs_path)
    if not specs_file.exists():
        raise FileNotFoundError(f"Specs file not found at {specs_file}")

    with open(specs_file, "r") as f:
        db = json.load(f)

    devices = db["devices"]
    model_stats = analyze_marl_policy()
    int8_weights_kb = model_stats["projected_int8_weight_memory_kb"]

    results = {}

    for dev_key, dev in devices.items():
        avail_sram_kb = dev["sram_kb"]
        avail_flash_kb = dev["flash_mb"] * 1024.0

        # Required memory
        required_sram_kb = TFLM_SRAM_INTERPRETER_OVERHEAD_KB + (
            model_stats["largest_activation_tensor_int8_bytes_lower_bound"] / 1024.0
        ) + (model_stats["recurrent_state_fp32_bytes"] / 1024.0)
        required_flash_kb = TFLM_FLASH_RUNTIME_OVERHEAD_KB + int8_weights_kb

        # Feasibility check
        sram_feasible = (required_sram_kb <= avail_sram_kb)
        flash_feasible = (required_flash_kb <= avail_flash_kb)
        is_feasible = (sram_feasible and flash_feasible)

        sram_utilization_pct = (required_sram_kb / avail_sram_kb) * 100.0
        flash_utilization_pct = (required_flash_kb / avail_flash_kb) * 100.0
        sram_headroom_kb = avail_sram_kb - required_sram_kb

        results[dev_key] = {
            "device_name": dev["name"],
            "available_sram_kb": avail_sram_kb,
            "required_sram_kb": round(required_sram_kb, 2),
            "sram_utilization_pct": round(sram_utilization_pct, 2),
            "sram_headroom_kb": round(sram_headroom_kb, 2),
            "available_flash_kb": avail_flash_kb,
            "required_flash_kb": round(required_flash_kb, 2),
            "flash_utilization_pct": round(flash_utilization_pct, 2),
            "is_feasible_by_analytical_memory_bound": is_feasible,
            "is_feasible": is_feasible,
            "analytical_only": True,
            "feasibility_notes": (
                "Estimated memory fit; conversion, operators, tensor arena, and target build unverified"
                if is_feasible
                else "Analytical memory bound exceeds device capacity"
            ),
        }

    return results

if __name__ == "__main__":
    mem_report = evaluate_memory_footprint()
    print("=" * 65)
    print("MICROCONTROLLER MEMORY FOOTPRINT & FEASIBILITY REPORT")
    print("=" * 65)
    print(json.dumps(mem_report, indent=2))
