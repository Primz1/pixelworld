"""Collect gameplay from the real game and serve training windows."""
import json
import os
import time

import numpy as np

from .env import CoinChase, ExplorerPolicy, N_ACTIONS


def collect(out_dir, n_frames=200_000, ep_len=(128, 384), seed=0, log_every=20_000):
    """Play the real game with the explorer policy and save every frame.

    Files written to out_dir:
      frames.npy   uint8 [N, 64, 64, 3]
      act_in.npy   int8  [N]  action that LED INTO frame i (0 = stay at episode start)
      ep_start.npy bool  [N]  True on the first frame of each episode
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames = np.lib.format.open_memmap(os.path.join(out_dir, "frames.npy"), mode="w+",
                                       dtype=np.uint8, shape=(n_frames, 64, 64, 3))
    act_in = np.zeros(n_frames, np.int8)
    ep_start = np.zeros(n_frames, bool)
    events = {"coin": 0, "hit": 0}
    i, n_eps, t0 = 0, 0, time.time()
    while i < n_frames:
        env = CoinChase(seed=int(rng.integers(1 << 31)))
        pol = ExplorerPolicy(rng)
        frames[i] = env.render()
        ep_start[i] = True
        i += 1
        n_eps += 1
        for _ in range(int(rng.integers(ep_len[0], ep_len[1] + 1))):
            if i >= n_frames:
                break
            a = pol(env)
            img, ev = env.step(a)
            if ev:
                events[ev] += 1
            frames[i] = img
            act_in[i] = a
            i += 1
            if i % log_every == 0:
                print(f"  collected {i:,}/{n_frames:,} frames ({time.time() - t0:.0f}s)", flush=True)
    frames.flush()
    np.save(os.path.join(out_dir, "act_in.npy"), act_in)
    np.save(os.path.join(out_dir, "ep_start.npy"), ep_start)
    meta = {"n_frames": n_frames, "episodes": n_eps, "n_actions": N_ACTIONS, **events}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load(data_dir, mmap=True):
    frames = np.load(os.path.join(data_dir, "frames.npy"), mmap_mode="r" if mmap else None)
    act_in = np.load(os.path.join(data_dir, "act_in.npy"))
    ep_start = np.load(os.path.join(data_dir, "ep_start.npy"))
    return frames, act_in, ep_start


def episode_index(ep_start):
    """For each frame, the index of the first frame of its episode."""
    starts = np.flatnonzero(ep_start)
    ep_id = np.cumsum(ep_start) - 1
    return starts[ep_id]


def split_episodes(ep_start, val_frac=0.05):
    """Split frame indices into train / val by whole episodes (last episodes go to val)."""
    starts = np.flatnonzero(ep_start)
    n_val = max(1, int(len(starts) * val_frac))
    cut = starts[-n_val]
    idx = np.arange(len(ep_start))
    return idx[:cut], idx[cut:]


def window_indices(targets, ep_first, k):
    """Context+target frame indices for each target frame.

    Returns int array [len(targets), k+1]: k context frames then the target.
    Context never crosses into a previous episode; near the start of an episode the
    first frame is repeated (exactly what happens at the start of a dream rollout).
    """
    offs = np.arange(-k, 1)
    idx = targets[:, None] + offs[None, :]
    return np.maximum(idx, ep_first[targets][:, None])


def valid_targets(frame_idx, ep_start):
    """Target frames must have a previous frame in the same episode."""
    return frame_idx[~ep_start[frame_idx]]
