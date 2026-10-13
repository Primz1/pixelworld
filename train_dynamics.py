"""Step 3: train the action-conditioned dynamics model on the tokenized dataset.

    python train_dynamics.py                     # defaults sized for an M1 Max/Ultra
    python train_dynamics.py --steps 3000 --d 256 --layers 4   # quick look

Needs data/tokens.npy from train_tokenizer.py. Writes checkpoints/dynamics.pt.
"""
import argparse
import json
import os

import numpy as np
import torch

from pixelworld.data import episode_index, split_episodes, valid_targets, window_indices
from pixelworld.dynamics import Dynamics
from pixelworld.env import N_ACTIONS
from pixelworld.utils import Timer, count_params, get_device, load_ckpt, lr_at, save_ckpt, seed_all


def batch(tokens, act_in, windows, device):
    t = torch.from_numpy(tokens[windows].reshape(len(windows), windows.shape[1], -1).astype(np.int64))
    a = torch.from_numpy(act_in[windows].astype(np.int64))
    return t.to(device), a.to(device)


@torch.no_grad()
def evaluate(dyn, tokens, act_in, windows, device, bs=128, full_frames=256):
    dyn.eval()
    cpu_state = torch.get_rng_state()
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    tot_l, tot_a, n = 0.0, 0.0, 0
    for i in range(0, len(windows), bs):
        t, a = batch(tokens, act_in, windows[i:i + bs], device)
        torch.manual_seed(i)  # fixed masks, comparable across evals
        l, acc = dyn.loss(t, a, ctx_noise=0.0)
        tot_l += l.item() * len(t)
        tot_a += acc.item() * len(t)
        n += len(t)
    # full next-frame prediction (greedy): fraction of tokens and of whole frames exactly right
    w = windows[:full_frames]
    t, a = batch(tokens, act_in, w, device)
    pred = dyn.predict_next(t[:, :-1], a, steps=8, temperature=0)
    tok_ok = (pred == t[:, -1]).float().mean().item()
    frame_ok = (pred == t[:, -1]).all(1).float().mean().item()
    dyn.train()
    torch.set_rng_state(cpu_state)
    if mps_state is not None:
        torch.mps.set_rng_state(mps_state)
    return {"loss": tot_l / n, "masked_acc": tot_a / n, "next_token_acc": tok_ok, "next_frame_exact": frame_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.pt")
    ap.add_argument("--out", default="checkpoints/dynamics.pt")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k", type=int, default=4, help="context frames")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx-noise", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)
    tokens = np.load(os.path.join(args.data, "tokens.npy"))
    act_in = np.load(os.path.join(args.data, "act_in.npy"))
    ep_start = np.load(os.path.join(args.data, "ep_start.npy"))
    vocab = load_ckpt(args.tokenizer, "cpu")["config"]
    vocab = int(np.prod(vocab.get("levels", (8, 5, 5, 5))))
    ep_first = episode_index(ep_start)
    train_idx, val_idx = split_episodes(ep_start)
    train_t = valid_targets(train_idx, ep_start)
    val_t = valid_targets(val_idx, ep_start)
    rng = np.random.default_rng(args.seed)
    val_w = window_indices(rng.choice(val_t, min(2048, len(val_t)), replace=False), ep_first, args.k)

    config = {"vocab": vocab, "n_actions": N_ACTIONS, "k_ctx": args.k, "tokens_per_frame": 64,
              "d": args.d, "layers": args.layers, "heads": args.heads}
    dyn = Dynamics(**config).to(device)
    opt = torch.optim.AdamW(dyn.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    start, best_resume = 0, None
    if args.resume and os.path.exists(args.out):
        ck = load_ckpt(args.out, device)
        dyn.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start = ck.get("step", 0)
        best_path = args.out.replace(".pt", "_best.pt")
        if os.path.exists(best_path):
            best_resume = load_ckpt(best_path, "cpu").get("val", {}).get("loss")
    print(f"device={device}  params={count_params(dyn) / 1e6:.1f}M  vocab={vocab}  "
          f"train targets={len(train_t):,}  val targets={len(val_t):,}")

    os.makedirs("runs/dynamics", exist_ok=True)
    timer, run_l, run_a = Timer(), 0.0, 0.0
    best = best_resume if best_resume is not None else float("inf")
    for step in range(start, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, warmup=1000)
        w = window_indices(rng.choice(train_t, args.batch), ep_first, args.k)
        t, a = batch(tokens, act_in, w, device)
        loss, acc = dyn.loss(t, a, ctx_noise=args.ctx_noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt.step()
        run_l += loss.item()
        run_a += acc.item()
        if (step + 1) % 100 == 0:
            print(f"[{timer}] step {step + 1:6d}  loss {run_l / 100:.4f}  masked-acc {run_a / 100:.3f}", flush=True)
            run_l = run_a = 0.0
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            m = evaluate(dyn, tokens, act_in, val_w, device)
            print(f"   val: loss {m['loss']:.4f}  masked-acc {m['masked_acc']:.3f}  "
                  f"next-token {m['next_token_acc']:.4f}  exact-frame {m['next_frame_exact']:.3f}", flush=True)
            save_ckpt(args.out, dyn, config, opt=opt.state_dict(), step=step + 1, val=m)
            if m["loss"] < best:
                best = m["loss"]
                save_ckpt(args.out.replace(".pt", "_best.pt"), dyn, config, step=step + 1, val=m)
            with open("runs/dynamics/last_eval.json", "w") as f:
                json.dump({"step": step + 1, **m}, f, indent=1)


if __name__ == "__main__":
    main()
