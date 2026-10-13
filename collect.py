"""Step 1: play the real game and record frames + actions.

    python collect.py                     # 200k frames (~2.4 GB, a few minutes)
    python collect.py --frames 50000      # smaller/faster
"""
import argparse

from pixelworld.data import collect

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--frames", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta = collect(args.out, n_frames=args.frames, seed=args.seed)
    print("done:", meta)
