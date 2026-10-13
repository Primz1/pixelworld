"""Step 4: measure the dream against the real game.

Plays fresh episodes of the real game, feeds the SAME actions to the learned world
(started from the same first frame), and checks, step by step, whether the dream
still has the player, the enemy and the score in the right place.

    python evaluate.py                     # 64 episodes x 64 steps
    python evaluate.py --gif-only

Writes runs/eval.json and runs/dream_vs_real.gif (top: real game, bottom: dream).
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from pixelworld import env as E
from pixelworld.utils import get_device, seed_all
from pixelworld.world import DreamWorld, load_dynamics, load_tokenizer


def run(world, n_eps, horizon, seed):
    rng = np.random.default_rng(seed)
    envs = [E.CoinChase(seed=int(rng.integers(1 << 31)) + 10_000_000) for _ in range(n_eps)]
    pols = [E.ExplorerPolicy(rng) for _ in envs]
    real = [np.stack([e.render() for e in envs])]
    dream = [world.reset(real[0])]
    for _ in range(horizon):
        acts = [p(e) for p, e in zip(pols, envs)]
        real.append(np.stack([e.step(a)[0] for e, a in zip(envs, acts)]))
        dream.append(world.step(acts))
    return np.stack(real, 1), np.stack(dream, 1)  # [N, T+1, 64, 64, 3]


def score(real, dream):
    N, T = real.shape[:2]
    out = {k: np.zeros(T) for k in ["player", "enemy", "score", "frozen_player", "frozen_enemy"]}
    for n in range(N):
        for t in range(T):
            r, d, f = real[n, t], dream[n, t], real[n, 0]
            rp, re = E.find_player(r), E.find_enemy(r)
            out["player"][t] += rp == E.find_player(d)
            out["enemy"][t] += re == E.find_enemy(d)
            out["score"][t] += E.count_pips(r) == E.count_pips(d)
            out["frozen_player"][t] += rp == E.find_player(f)
            out["frozen_enemy"][t] += re == E.find_enemy(f)
    return {k: (v / N).round(4).tolist() for k, v in out.items()}


def save_gif(real, dream, path, n=4, scale=4):
    frames = []
    for t in range(real.shape[1]):
        top = np.concatenate([real[i, t] for i in range(n)], 1)
        bot = np.concatenate([dream[i, t] for i in range(n)], 1)
        sep = np.full((2, top.shape[1], 3), 255, np.uint8)
        img = np.concatenate([top, sep, bot], 0)
        frames.append(Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.pt")
    ap.add_argument("--dynamics", default="checkpoints/dynamics_best.pt")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8, help="MaskGIT decoding steps per frame")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--gif-only", action="store_true")
    args = ap.parse_args()
    seed_all(args.seed)
    device = get_device(args.device)
    if not os.path.exists(args.dynamics):
        args.dynamics = args.dynamics.replace("_best", "")
    world = DreamWorld(load_tokenizer(args.tokenizer, device), load_dynamics(args.dynamics, device), device,
                       steps=args.steps, temperature=args.temperature)
    n = 4 if args.gif_only else args.episodes
    real, dream = run(world, n, args.horizon, args.seed)
    os.makedirs("runs", exist_ok=True)
    save_gif(real, dream, "runs/dream_vs_real.gif", n=min(4, n))
    print("wrote runs/dream_vs_real.gif")
    if args.gif_only:
        return
    s = score(real, dream)
    with open("runs/eval.json", "w") as f:
        json.dump({"episodes": n, "horizon": args.horizon, **s}, f, indent=1)
    print(f"\n{'step':>5} {'player':>8} {'enemy':>8} {'score':>8}   (frozen-frame baseline: player / enemy)")
    for t in [1, 2, 4, 8, 16, 32, 64]:
        if t < len(s["player"]):
            print(f"{t:5d} {s['player'][t]:8.3f} {s['enemy'][t]:8.3f} {s['score'][t]:8.3f}   "
                  f"{s['frozen_player'][t]:.3f} / {s['frozen_enemy'][t]:.3f}")
    print("\nplayer/enemy/score = fraction of episodes where the dream still matches the real game at that step.")


if __name__ == "__main__":
    main()
