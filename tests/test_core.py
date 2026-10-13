import numpy as np
import pytest
import torch

from pixelworld import env as E
from pixelworld.data import episode_index, window_indices
from pixelworld.dynamics import Dynamics
from pixelworld.tokenizer import FSQ, Tokenizer, pixel_weights, to_tensor


def test_env_detectors_track_state():
    rng = np.random.default_rng(0)
    for ep in range(5):
        env = E.CoinChase(seed=ep)
        pol = E.ExplorerPolicy(rng)
        for _ in range(100):
            img, _ = env.step(pol(env))
            assert img.shape == (64, 64, 3) and img.dtype == np.uint8
            assert E.find_player(img) == tuple(env.player)
            assert E.find_enemy(img) == tuple(env.enemy)
            assert E.count_pips(img) == env.score


def test_windows_never_cross_episodes():
    ep_start = np.array([1, 0, 0, 1, 0, 0, 0, 0], bool)
    w = window_indices(np.array([4, 7]), episode_index(ep_start), 4)
    assert w.tolist() == [[3, 3, 3, 3, 4], [3, 4, 5, 6, 7]]


def test_fsq_roundtrip():
    fsq = FSQ((8, 5, 5, 5))
    z = torch.randn(3, 4, 8, 8) * 3
    zq, idx = fsq(z)
    assert idx.min() >= 0 and idx.max() < 1000
    assert torch.allclose(fsq.indices_to_codes(idx), zq, atol=1e-6)
    all_idx = torch.arange(1000).view(1, 10, 100)
    assert torch.equal(fsq.codes_to_indices(fsq.indices_to_codes(all_idx)), all_idx)


def test_tokenizer_shapes_and_grad():
    tok = Tokenizer(ch=(16, 32, 32))
    frames = np.stack([E.CoinChase(seed=i).render() for i in range(2)])
    x = to_tensor(frames, "cpu")
    rec, idx = tok(x)
    assert rec.shape == x.shape and idx.shape == (2, 8, 8)
    (pixel_weights(x, E.BACKGROUND_COLORS) * (rec - x) ** 2).mean().backward()
    assert all(p.grad is not None for p in tok.parameters())
    assert tok.decode(tok.encode(x)).shape == x.shape


def test_dynamics_block_causal_and_generate():
    torch.manual_seed(0)
    dyn = Dynamics(vocab=1000, n_actions=5, k_ctx=3, d=32, layers=2, heads=2).eval()
    t = torch.randint(0, 1000, (2, 4, 64))
    a = torch.randint(0, 5, (2, 4))
    # the prediction must depend on the action taken now ...
    a2 = a.clone(); a2[:, -1] = (a[:, -1] + 1) % 5
    assert not torch.allclose(dyn(t, a), dyn(t, a2))
    # ... and context frames must not see the frame being predicted (block-causal)
    x = dyn.tok(t) + dyn.pos + dyn.frame + dyn.act(a).unsqueeze(2)
    x = x.reshape(2, 256, -1)
    t2 = t.clone(); t2[:, -1] = 7
    x2 = (dyn.tok(t2) + dyn.pos + dyn.frame + dyn.act(a).unsqueeze(2)).reshape(2, 256, -1)
    h, h2 = x, x2
    for blk in dyn.blocks:
        h, h2 = blk(h, dyn.attn_mask), blk(h2, dyn.attn_mask)
    assert torch.allclose(h[:, :192], h2[:, :192], atol=1e-5)
    loss, acc = dyn.train().loss(t, a)
    loss.backward()
    nxt = dyn.eval().predict_next(t[:, :-1], a, steps=4)
    assert nxt.shape == (2, 64) and (nxt < 1000).all()
    greedy = dyn.predict_next(t[:, :-1], a, steps=4, temperature=0)
    assert (greedy < 1000).all()


@pytest.mark.slow
def test_pipeline_end_to_end():
    import smoke_test
    smoke_test.main("cpu")
