"""Compatibility entry point for the canonical, sanity-gated QMIX trainer."""

from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_ALL = ROOT_DIR / "training" / "train_all.py"

def train_marl():
    if any(arg == "--alg" or arg.startswith("--alg=") for arg in sys.argv[1:]):
        raise SystemExit("training/train_qmix.py fixes --alg=qmix; do not pass --alg")
    cmd = [
        sys.executable,
        str(TRAIN_ALL),
        "--alg=qmix",
        *sys.argv[1:],
    ]
    print(f"Running canonical QMIX training: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT_DIR, check=False).returncode

if __name__ == "__main__":
    raise SystemExit(train_marl())
