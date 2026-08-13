import subprocess
import sys
import os

# Get path to epymarl's main.py
epymarl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epymarl", "src")
main_script = os.path.join(epymarl_dir, "main.py")

# Ensure DEVICE and SEED from shared config are used
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shared_config

def train_marl():
    cmd = [
        sys.executable, main_script,
        "--config=iql",
        "--env-config=iot",
        "with",
        "t_max=1000000",
        "lr=0.001",
        "epsilon_anneal_time=100000",
        "common_reward=False",
        "reward_scalarisation=None",
        "standardise_rewards=False",
        "use_rnn=False",
        "save_model=True",
        "save_model_interval=200000"
    ]
    print(f"Running MARL with command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    train_marl()
