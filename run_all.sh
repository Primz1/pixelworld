#!/usr/bin/env bash
# Full pipeline. Roughly 2-4 hours on an M1 Max/Ultra (estimates, not measured).
set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1
python collect.py --frames 200000
python train_tokenizer.py
python train_dynamics.py
python evaluate.py
echo "Done. Open runs/dream_vs_real.gif, then: python play.py"
