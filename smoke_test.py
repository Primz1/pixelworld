"""Run the whole pipeline at toy size in a temp folder (about a minute).
Use it first on a new machine to check everything is installed and works:

    python smoke_test.py            # uses the Mac GPU (mps) if available
    python smoke_test.py --device cpu
"""
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(args, cwd, device):
    env = dict(os.environ, PYTHONPATH=ROOT, PYTORCH_ENABLE_MPS_FALLBACK="1")
    cmd = [sys.executable, os.path.join(ROOT, args[0])] + args[1:] + ["--device", device]
    if args[0] == "collect.py":
        cmd = cmd[:-2]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main(device="auto"):
    with tempfile.TemporaryDirectory() as d:
        run(["collect.py", "--frames", "3000"], d, device)
        run(["train_tokenizer.py", "--steps", "40", "--batch", "16", "--ch", "16", "32", "32",
             "--eval-every", "40", "--eval-frames", "32"], d, device)
        run(["train_dynamics.py", "--steps", "40", "--batch", "16", "--d", "64", "--layers", "2",
             "--heads", "2", "--eval-every", "40"], d, device)
        run(["evaluate.py", "--episodes", "4", "--horizon", "6"], d, device)
        for f in ["checkpoints/tokenizer.pt", "checkpoints/dynamics.pt", "data/tokens.npy",
                  "runs/eval.json", "runs/dream_vs_real.gif"]:
            assert os.path.exists(os.path.join(d, f)), f"missing {f}"
    print("\nSMOKE TEST PASSED: the pipeline runs end to end on this machine.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    main(ap.parse_args().device)
