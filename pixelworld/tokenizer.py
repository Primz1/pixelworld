"""Stage 1: an image tokenizer.

Turns a 64x64 RGB frame into an 8x8 grid of discrete tokens (and back).
Quantisation is FSQ (finite scalar quantisation, Mentzer et al. 2023): each of the
4 latent channels is squashed and rounded to a few levels; with levels (8,5,5,5)
that is 1000 possible codes per token. No codebook, no commitment loss, no collapse.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FSQ(nn.Module):
    def __init__(self, levels=(8, 5, 5, 5), eps=1e-3):
        super().__init__()
        levels = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("levels", levels, persistent=False)
        basis = torch.cumprod(torch.cat([torch.ones(1), levels[:-1]]), 0)
        self.register_buffer("basis", basis, persistent=False)
        self.eps = eps
        self.dim = len(levels)
        self.codebook_size = int(levels.prod().item())

    def _bound(self, z):
        lv = self.levels.view(1, -1, 1, 1)
        half_l = (lv - 1) * (1 + self.eps) / 2
        offset = torch.where(lv % 2 == 0, torch.full_like(lv, 0.5), torch.zeros_like(lv))
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def forward(self, z):
        """z: [B, dim, h, w] -> quantised zq (same shape, values in [-1, 1]), indices [B, h, w]."""
        b = self._bound(z)
        q = b + (torch.round(b) - b).detach()  # straight-through rounding
        half_w = (self.levels.view(1, -1, 1, 1) // 2)
        zq = q / half_w
        return zq, self.codes_to_indices(zq)

    def codes_to_indices(self, zq):
        half_w = (self.levels.view(1, -1, 1, 1) // 2)
        digits = torch.round(zq * half_w + half_w)
        return (digits * self.basis.view(1, -1, 1, 1)).sum(1).long()

    def indices_to_codes(self, idx):
        """idx: [B, h, w] long -> zq [B, dim, h, w]."""
        idx = idx.unsqueeze(1).long()
        lv = self.levels.view(1, -1, 1, 1)
        digits = ((idx // self.basis.long().view(1, -1, 1, 1)) % lv.long()).float()  # exact integer math
        half_w = lv // 2
        return (digits - half_w) / half_w


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class Tokenizer(nn.Module):
    def __init__(self, ch=(64, 128, 256), levels=(8, 5, 5, 5)):
        super().__init__()
        self.fsq = FSQ(levels)
        c0, c1, c2 = ch
        self.enc = nn.Sequential(
            nn.Conv2d(3, c0, 3, padding=1), ResBlock(c0),
            nn.Conv2d(c0, c1, 4, stride=2, padding=1), ResBlock(c1),   # 32x32
            nn.Conv2d(c1, c2, 4, stride=2, padding=1), ResBlock(c2),   # 16x16
            nn.Conv2d(c2, c2, 4, stride=2, padding=1), ResBlock(c2), ResBlock(c2),  # 8x8
            nn.GroupNorm(8, c2), nn.SiLU(), nn.Conv2d(c2, self.fsq.dim, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(self.fsq.dim, c2, 3, padding=1), ResBlock(c2), ResBlock(c2),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c2, c2, 3, padding=1), ResBlock(c2),  # 16
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c2, c1, 3, padding=1), ResBlock(c1),  # 32
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c1, c0, 3, padding=1), ResBlock(c0),  # 64
            nn.GroupNorm(8, c0), nn.SiLU(), nn.Conv2d(c0, 3, 3, padding=1),
        )

    @property
    def vocab(self):
        return self.fsq.codebook_size

    def encode(self, x):
        """x: float [B,3,64,64] in [0,1] -> token indices [B,8,8]."""
        _, idx = self.fsq(self.enc(x * 2 - 1))
        return idx

    def decode(self, idx):
        """idx: long [B,8,8] -> float image [B,3,64,64] in [0,1]."""
        return (self.dec(self.fsq.indices_to_codes(idx)) + 1) / 2

    def forward(self, x):
        zq, idx = self.fsq(self.enc(x * 2 - 1))
        return (self.dec(zq) + 1) / 2, idx


def to_tensor(frames_uint8, device):
    """uint8 numpy [B,64,64,3] -> float tensor [B,3,64,64] in [0,1]."""
    t = torch.from_numpy(np.array(frames_uint8, dtype=np.uint8)).to(device)
    return t.permute(0, 3, 1, 2).float() / 255.0


def to_uint8(img):
    """float tensor [B,3,64,64] -> uint8 numpy [B,64,64,3]."""
    return (img.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def pixel_weights(x, background_colors, rare_weight=20.0):
    """Per-pixel loss weight: 1 for background colours, rare_weight for everything else
    (sprites, coin, HUD pips), so the 4x4 sprites dominate the reconstruction loss."""
    bg = torch.as_tensor(background_colors, dtype=torch.float32, device=x.device) / 255.0  # [C,3]
    d = (x.unsqueeze(1) - bg.view(1, -1, 3, 1, 1)).abs().sum(2)  # [B,C,H,W]
    is_bg = (d.min(1).values < 0.06).float()  # [B,H,W]
    return (1 + (rare_weight - 1) * (1 - is_bg)).unsqueeze(1)
