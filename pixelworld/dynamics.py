"""Stage 2: the action-conditioned dynamics model (the 'world model' proper).

Input: the tokens of the last K frames, plus the action that led into each frame,
plus the action the player takes now. Output: the tokens of the next frame.

It is a transformer with block-causal attention over frames (a frame sees itself
and earlier frames), trained MaskGIT-style: the next frame's tokens are partially
replaced by [MASK] and the model predicts the hidden ones. At play time the next
frame starts fully masked and is filled in over a few confident-first steps.
This is the same recipe as Genie (DeepMind, 2024), at toy scale.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads, mlp=4, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))
        self.heads = heads
        self.dropout = dropout

    def forward(self, x, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.n2(x))


class Dynamics(nn.Module):
    def __init__(self, vocab, n_actions, k_ctx=4, tokens_per_frame=64,
                 d=256, layers=6, heads=4, dropout=0.0):
        super().__init__()
        self.vocab, self.n_actions, self.k = vocab, n_actions, k_ctx
        self.tpf = tokens_per_frame
        self.n_frames = k_ctx + 1
        self.mask_id = vocab
        self.tok = nn.Embedding(vocab + 1, d)
        self.pos = nn.Parameter(torch.zeros(1, 1, tokens_per_frame, d))
        self.frame = nn.Parameter(torch.zeros(1, self.n_frames, 1, d))
        self.act = nn.Embedding(n_actions, d)
        self.blocks = nn.ModuleList([Block(d, heads, dropout=dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.frame, std=0.02)
        f = torch.arange(self.n_frames).repeat_interleave(tokens_per_frame)
        self.register_buffer("attn_mask", f[:, None] >= f[None, :], persistent=False)  # True = may attend

    def forward(self, tokens, act_in):
        """tokens: long [B, K+1, 64] (last frame may contain mask_id).
        act_in:  long [B, K+1]      action that led into each frame (the last one is the action
                                    being taken now, i.e. what produces the predicted frame).
        Returns logits for the last frame: [B, 64, vocab]."""
        B, F_, T = tokens.shape
        x = self.tok(tokens) + self.pos + self.frame[:, :F_] + self.act(act_in).unsqueeze(2)
        x = x.reshape(B, F_ * T, -1)
        mask = self.attn_mask[:F_ * T, :F_ * T]
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x[:, -T:])
        return self.head(x)

    # ------------------------------------------------------------------ train
    def loss(self, tokens, act_in, ctx_noise=0.05):
        """MaskGIT loss on the last frame. Context tokens get light random corruption so the
        model learns to cope with its own small mistakes during long dream rollouts."""
        B, F_, T = tokens.shape
        tokens = tokens.clone()
        if ctx_noise > 0:
            ctx = tokens[:, :-1]
            p = torch.rand(B, 1, 1, device=tokens.device) * ctx_noise
            noise = torch.rand(ctx.shape, device=tokens.device) < p
            rnd = torch.randint(0, self.vocab, ctx.shape, device=tokens.device)
            tokens[:, :-1] = torch.where(noise, rnd, ctx)
        target = tokens[:, -1].clone()
        r = torch.cos(torch.rand(B, device=tokens.device) * math.pi / 2)  # mask ratio in (0, 1]
        n_mask = (r * T).ceil().clamp(min=1).long()
        scores = torch.rand(B, T, device=tokens.device)
        ranks = scores.argsort(1).argsort(1)
        masked = ranks < n_mask[:, None]
        tokens[:, -1] = torch.where(masked, torch.full_like(target, self.mask_id), target)
        logits = self(tokens, act_in)
        loss = F.cross_entropy(logits[masked], target[masked])
        with torch.no_grad():
            acc = (logits.argmax(-1)[masked] == target[masked]).float().mean()
        return loss, acc

    # --------------------------------------------------------------- generate
    @torch.no_grad()
    def predict_next(self, ctx_tokens, act_in, steps=8, temperature=1.0, top_k=None):
        """ctx_tokens: long [B, K, 64] last K frames. act_in: long [B, K+1].
        Returns long [B, 64] tokens of the next frame."""
        assert steps >= 1, "need at least one decoding step"
        B, K, T = ctx_tokens.shape
        cur = torch.full((B, 1, T), self.mask_id, dtype=torch.long, device=ctx_tokens.device)
        for s in range(steps):
            tokens = torch.cat([ctx_tokens, cur], 1)
            logits = self(tokens, act_in).float()
            if temperature <= 0:
                probs = logits.softmax(-1)
                conf, sample = probs.max(-1)
            else:
                logits = logits / temperature
                if top_k:
                    kth = logits.topk(top_k, -1).values[..., -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = logits.softmax(-1)
                sample = torch.multinomial(probs.view(-1, self.vocab), 1).view(B, T)
                conf = probs.gather(-1, sample.unsqueeze(-1)).squeeze(-1)
            still = cur[:, 0] == self.mask_id
            sample = torch.where(still, sample, cur[:, 0])
            # how many tokens remain masked after this step (cosine schedule)
            n_keep_masked = int(math.floor(T * math.cos(math.pi / 2 * (s + 1) / steps)))
            if n_keep_masked <= 0:
                cur = sample.unsqueeze(1)
                break
            conf = torch.where(still, conf, torch.full_like(conf, float("inf")))
            lowest = conf.argsort(1)[:, :n_keep_masked]  # exactly n least-confident stay masked
            remask = torch.zeros_like(still).scatter(1, lowest, True) & still
            cur = torch.where(remask, torch.full_like(sample, self.mask_id), sample).unsqueeze(1)
        return cur[:, 0]
