# MARL Adaptive IoT Sampling Project

This repository contains the codebase for a Multi-Agent Reinforcement Learning system for adaptive IoT sampling rate control under energy harvesting constraints.

## Modules

The project is split into three modules developed in parallel:
- **Module A (Environment and Simulation Core)**: `environment/`
- **Module B (MARL Algorithm and Training)**: `training/`
- **Module C (Hardware Profiling and Evaluation)**: `hardware_eval/`

## Shared Configuration

All modules must adhere to the interface contracts and parameters defined in `shared_config.py`. Do not hardcode state shapes, action shapes, episode lengths, or scenario parameters.

## Setup Instructions

This project supports cross-platform development across Windows and Mac. Follow these steps to set up your environment:

1. **Clone the repository:**
   ```bash
   git clone <repo_url>
   cd FinalYearProject-MARL
   ```

2. **Create a virtual environment:**
   - On Windows: `python -m venv venv`
   - On Mac: `python3 -m venv venv`

3. **Activate the virtual environment:**
   - On Windows: `venv\Scripts\activate`
   - On Mac: `source venv/bin/activate`

4. **Install shared dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
