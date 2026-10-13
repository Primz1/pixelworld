"""Step 2: train the tokenizer (64x64 frame <-> 8x8 grid of tokens), then tokenize the dataset.

    python train_tokenizer.py                    # defaults are sized for an M1 Max/Ultra
    python train_tokenizer.py --steps 2000       # quick look

Writes checkpoints/tokenizer.pt, data/tokens.npy, and sample images in runs/tokenizer/.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from pixelworld import env as E
from pixelworld.data import load, split_episodes
from pixelworld.tokenizer import Tokenizer, pixel_weights, to_tensor, to_uint8
from pixelworld.utils import Timer, count_params, get_device, load_ckpt, lr_at, save_ckpt, seed_all


@torch.no_grad()
def evaluate(tok, frames, idx, device, bs=256):
    """Reconstruction error and how often sprites / score survive the round trip."""
    tok.eval()
    mse, ok_p, ok_e, ok_s, n = 0.0, 0, 0, 0, 0
    for i in range(0, len(idx), bs):
        x = frames[np.sort(idx[i:i + bs])]
        rec, _ = tok(to_tensor(x, device))
        r = to_uint8(rec)
        mse += float(((r.astype(np.float32) - x) ** 2).mean()) * len(x)
        for a, b in zip(x, r):
            ok_p += E.find_player(a) == E.find_player(b)
            ok_e += E.find_enemy(a) == E.find_enemy(b)
            ok_s += E.count_pips(a) == E.count_pips(b)
        n += len(x)
    tok.train()
    return {"mse": mse / n, "player_ok": ok_p / n, "enemy_ok": ok_e / n, "score_ok": ok_s / n}


def save_samples(tok, frames, idx, device, path):
    with torch.no_grad():
        x = frames[np.sort(idx[:8])]
        r = to_uint8(tok(to_tensor(x, device))[0])
    grid = np.concatenate([np.concatenate(list(x), 1), np.concatenate(list(r), 1)], 0)
    Image.fromarray(grid).resize((grid.shape[1] * 4, grid.shape[0] * 4), Image.NEAREST).save(path)


@torch.no_grad()
def tokenize_all(tok, frames, device, out_path, bs=512):
    tok.eval()
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.int16, shape=(len(frames), 8, 8))
    for i in range(0, len(frames), bs):
        out[i:i + bs] = tok.encode(to_tensor(frames[i:i + bs], device)).cpu().numpy().astype(np.int16)
    out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="checkpoints/tokenizer.pt")
    ap.add_argument("--steps", type=int, default=15_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ch", type=int, nargs=3, default=[64, 128, 256])
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-frames", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)
    frames, _, ep_start = load(args.data)
    train_idx, val_idx = split_episodes(ep_start)
    rng = np.random.default_rng(args.seed)
    val_eval = rng.choice(val_idx, min(args.eval_frames, len(val_idx)), replace=False)

    config = {"ch": tuple(args.ch), "levels": (8, 5, 5, 5)}
    tok = Tokenizer(**config).to(device)
    opt = torch.optim.AdamW(tok.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))
    start = 0
    if args.resume and os.path.exists(args.out):
        ck = load_ckpt(args.out, device)
        tok.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start = ck.get("step", 0)
    print(f"device={device}  params={count_params(tok) / 1e6:.1f}M  train frames={len(train_idx):,}  "
          f"val frames={len(val_idx):,}")

    os.makedirs("runs/tokenizer", exist_ok=True)
    timer, run_loss = Timer(), 0.0
    for step in range(start, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr)
        b = np.sort(rng.choice(train_idx, args.batch, replace=False))
        x = to_tensor(frames[b], device)
        rec, _ = tok(x)
        w = pixel_weights(x, E.BACKGROUND_COLORS)
        loss = (w * (rec - x) ** 2).mean() + 0.1 * (w * (rec - x).abs()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
        opt.step()
        run_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f"[{timer}] step {step + 1:6d}  loss {run_loss / 100:.5f}", flush=True)
            run_loss = 0.0
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            m = evaluate(tok, frames, val_eval, device)
            print(f"   val: mse {m['mse']:.2f}  player {m['player_ok']:.3f}  enemy {m['enemy_ok']:.3f}  "
                  f"score {m['score_ok']:.3f}", flush=True)
            save_samples(tok, frames, val_eval, device, f"runs/tokenizer/step{step + 1:06d}.png")
            save_ckpt(args.out, tok, config, opt=opt.state_dict(), step=step + 1, val=m)

    with open("runs/tokenizer/final.json", "w") as f:
        json.dump(m, f, indent=1)
    print("final val:", m)
    print("tokenizing the whole dataset ...")
    tokenize_all(tok, frames, device, os.path.join(args.data, "tokens.npy"))
    print("wrote", os.path.join(args.data, "tokens.npy"))


if __name__ == "__main__":
    main()
