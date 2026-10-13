"""Load trained models and run the learned world ('the dream') one frame at a time."""
import torch

from .dynamics import Dynamics
from .tokenizer import Tokenizer, to_tensor, to_uint8
from .utils import load_ckpt


def load_tokenizer(path, device):
    ck = load_ckpt(path, device)
    tok = Tokenizer(**ck["config"]).to(device).eval()
    tok.load_state_dict(ck["model"])
    return tok


def load_dynamics(path, device):
    ck = load_ckpt(path, device)
    dyn = Dynamics(**ck["config"]).to(device).eval()
    dyn.load_state_dict(ck["model"])
    return dyn


class DreamWorld:
    """Batched learned simulator. Start it from real frame(s), then feed actions.

        world = DreamWorld(tok, dyn, device)
        frame = world.reset(first_frames)      # uint8 [B,64,64,3]
        frame = world.step(actions)            # actions: list/array of B ints
    """

    def __init__(self, tok, dyn, device, steps=8, temperature=1.0, top_k=None):
        self.tok, self.dyn, self.device = tok, dyn, device
        self.steps, self.temperature, self.top_k = steps, temperature, top_k
        self.k = dyn.k

    @torch.no_grad()
    def reset(self, frames_uint8):
        idx = self.tok.encode(to_tensor(frames_uint8, self.device)).flatten(1)  # [B,64]
        self.ctx = idx.unsqueeze(1).repeat(1, self.k, 1)  # repeat first frame K times
        self.acts = torch.zeros(idx.shape[0], self.k, dtype=torch.long, device=self.device)
        return to_uint8(self.tok.decode(idx.view(-1, 8, 8)))

    @torch.no_grad()
    def step(self, actions):
        a = torch.as_tensor(actions, dtype=torch.long, device=self.device).view(-1, 1)
        act_in = torch.cat([self.acts, a], 1)  # [B, K+1]
        nxt = self.dyn.predict_next(self.ctx, act_in, steps=self.steps,
                                    temperature=self.temperature, top_k=self.top_k)
        self.ctx = torch.cat([self.ctx[:, 1:], nxt.unsqueeze(1)], 1)
        self.acts = act_in[:, 1:]
        return to_uint8(self.tok.decode(nxt.view(-1, 8, 8)))
