import subprocess
import sys
import os

# Get path to epymarl's main.py
epymarl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epymarl", "src")
main_script = os.path.join(epymarl_dir, "main.py")

# Ensure DEVICE and SEED from shared config are used
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shared_config

def train_qmix():
    cmd = [
        sys.executable, main_script,
        "--config=qmix",
        "--env-config=iot",
        f"device={shared_config.DEVICE}",
        f"seed={shared_config.SEED}"
    ]
    print(f"Running QMIX with command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    train_qmix()
