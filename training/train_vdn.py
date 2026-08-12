import subprocess
import sys
import os

# Get path to epymarl's main.py
epymarl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epymarl", "src")
main_script = os.path.join(epymarl_dir, "main.py")

# Ensure DEVICE and SEED from shared config are used
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shared_config

def train_vdn():
    cmd = [
        sys.executable, main_script,
        "--config=vdn",
        "--env-config=iot",
        "with",
        "save_model=True",
        "save_model_interval=25000",
        "buffer_size=5000",
        "batch_size=16",
        "epsilon_anneal_time=25000",
        "epsilon_finish=0.10"
    ]
    print(f"Running VDN with command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    train_vdn()
