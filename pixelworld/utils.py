import math
import os
import random
import time

import numpy as np
import torch


def get_device(name="auto"):
    """Apple GPU (mps) on a Mac, CUDA on NVIDIA, otherwise CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at(step, total, base, warmup=500, floor=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def save_ckpt(path, model, config, **extra):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "config": config, **extra}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __str__(self):
        s = int(time.time() - self.t0)
        return f"{s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}"


def count_params(m):
    return sum(p.numel() for p in m.parameters())
