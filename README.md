# PixelWorld

A small **world model that learns a game from pixels** and then lets you play inside it.

The model never sees the game's code or state. It watches 64x64 frames and the keys pressed, and learns
to produce the next frame. Once trained, `play.py` runs the game *entirely inside the model*: walls,
movement, the chasing enemy, coin pickups and the score counter are all generated, pixel by pixel.

It is the same two-stage recipe as DeepMind's Genie and IRIS, at a size that trains on an Apple-silicon Mac:

1. **Tokenizer** (`pixelworld/tokenizer.py`): a conv autoencoder with FSQ quantisation turns each
   64x64 frame into an 8x8 grid of tokens (1000 possible codes each) and back.
2. **Dynamics model** (`pixelworld/dynamics.py`): a transformer reads the last 4 frames' tokens plus the
   actions, and predicts the next frame's tokens. It is trained MaskGIT-style, and at play time each
   frame is filled in over 8 parallel steps (fast enough to play).

The "real world" is **CoinChase** (`pixelworld/env.py`): you (blue) collect coins (yellow) while an
enemy (red) chases you. Coins add a pip to the score bar; getting caught resets the score and the bar
flashes red.

## Setup (Mac with Apple silicon)

```bash
git clone <your repo url> pixelworld && cd pixelworld
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py        # ~1-2 min: runs the whole pipeline at toy size, checks the Mac GPU works
```

The scripts pick the Apple GPU (`mps`) automatically. Add `--device cpu` to any script to force the CPU.
If PyTorch says an op is not implemented on MPS, run with `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Run it

```bash
python collect.py                 # 1. play the real game, record 200k frames (~2.4 GB in data/)
python train_tokenizer.py         # 2. learn frames <-> tokens, then tokenize the dataset
python train_dynamics.py          # 3. learn what happens next, given an action
python evaluate.py                # 4. measure dream vs real; writes runs/dream_vs_real.gif
python play.py                    # 5. play inside the model (arrows/WASD, R = new world)
```

Or run `./run_all.sh` for steps 1-4. Every training script prints progress, saves checkpoints as it goes
(`checkpoints/`), and accepts `--resume`.

My rough guess for an M1 Max/Ultra is 30-60 min for the tokenizer and 1.5-3 h for the dynamics model at the
defaults. **These are estimates, not measurements.** For a quick first pass, use
`--steps 3000` on both.

## What to look at

* `runs/tokenizer/stepXXXXXX.png`: top row real frames, bottom row reconstructions. The validation line
  reports how often the player, enemy and score survive the round trip. Aim for >0.99 before moving on.
* The dynamics validation line: `next-token` (fraction of next-frame tokens exactly right) and
  `exact-frame` (fraction of next frames with every token right).
* `evaluate.py` prints, for steps 1..64 of a rollout, how often the dream still has the player, the enemy
  and the score where the real game has them. It also prints a "frozen frame" baseline (just repeating the first
  frame), so you can see the model is really simulating, not copying.
* `runs/dream_vs_real.gif`: top is the real game, bottom is the dream, same key presses.

Some divergence is expected and honest. Coin respawns and post-hit respawns are random, so the dream
picks its own (plausible) spot, and from then on the two worlds legitimately differ.

## Things to try next (where the bigger machine helps)

* **Bigger dynamics model**: `python train_dynamics.py --d 384 --layers 8 --heads 6 --steps 60000`.
* **More context**: `--k 8` so the model can remember things that are off-screen for a few frames.
* **Sampling**: `play.py --temperature 0.7` (steadier) or `--steps 4` (faster).
* **Harder world**: edit `env.py`, e.g. more enemies, doors and keys, scrolling. Re-collect and retrain.
* **Real game footage**: swap `collect.py` for frames from any game you can record at 64x64 with key logs.
* **Learn to act in the dream**: train an agent purely inside `DreamWorld` (the Dreamer/IRIS idea) and
  test it in the real game.

## Layout

```
pixelworld/env.py         the real game + sprite detectors used for evaluation
pixelworld/data.py        data collection, episode-safe context windows
pixelworld/tokenizer.py   FSQ autoencoder
pixelworld/dynamics.py    block-causal MaskGIT transformer
pixelworld/world.py       DreamWorld: run the learned simulator step by step
collect.py  train_tokenizer.py  train_dynamics.py  evaluate.py  play.py  smoke_test.py
tests/                    unit tests + full toy pipeline (run by GitHub Actions on every push)
```

## Status

The PyTorch code has not been run yet: it was written and reviewed in a sandbox without PyTorch. The game,
data collection and window logic were run and tested. GitHub Actions (`.github/workflows/ci.yml`) installs
CPU PyTorch and runs `pytest` on every push, including the full toy pipeline. After you push, check the Actions
tab, then run `python smoke_test.py` on the Mac.
