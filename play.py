"""Step 5: play inside the learned world.

Left: the dream (every pixel produced by the model). Right: the real game, driven by
the same keys, so you can see where they agree and where the dream drifts.

    python play.py
    python play.py --dream-only --temperature 0.8

Keys: arrows / WASD move, nothing pressed = stay, R = new world, Esc = quit.
"""
import argparse
import os
import time

import numpy as np

from pixelworld import env as E
from pixelworld.utils import get_device
from pixelworld.world import DreamWorld, load_dynamics, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.pt")
    ap.add_argument("--dynamics", default="checkpoints/dynamics_best.pt")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--dream-only", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import pygame  # imported here so the rest of the repo works without it

    device = get_device(args.device)
    if not os.path.exists(args.dynamics):
        args.dynamics = args.dynamics.replace("_best", "")
    world = DreamWorld(load_tokenizer(args.tokenizer, device), load_dynamics(args.dynamics, device), device,
                       steps=args.steps, temperature=args.temperature)

    panes = 1 if args.dream_only else 2
    S = 64 * args.scale
    pygame.init()
    screen = pygame.display.set_mode((S * panes + 16 * (panes - 1), S + 36))
    pygame.display.set_caption("PixelWorld: playing inside a learned world model")
    font = pygame.font.SysFont("menlo,monospace", 16)

    def new_world():
        env = E.CoinChase(seed=int(time.time() * 1000) % (1 << 31))
        first = env.render()
        return env, world.reset(first[None])[0], first

    env, dream, real = new_world()
    clock, ms = pygame.time.Clock(), 0.0
    keymap = {pygame.K_UP: 1, pygame.K_w: 1, pygame.K_DOWN: 2, pygame.K_s: 2,
              pygame.K_LEFT: 3, pygame.K_a: 3, pygame.K_RIGHT: 4, pygame.K_d: 4}
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                env, dream, real = new_world()
        pressed = pygame.key.get_pressed()
        a = 0
        for k, v in keymap.items():
            if pressed[k]:
                a = v
        t0 = time.time()
        dream = world.step([a])[0]
        ms = 0.9 * ms + 0.1 * (time.time() - t0) * 1000 if ms else (time.time() - t0) * 1000
        real, _ = env.step(a)

        screen.fill((16, 18, 22))
        for i, img in enumerate([dream] if args.dream_only else [dream, real]):
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))
            screen.blit(pygame.transform.scale(surf, (S, S)), (i * (S + 16), 36))
        label = f"DREAM ({ms:.0f} ms/frame)   action: {E.ACTION_NAMES[a]}"
        screen.blit(font.render(label, True, (230, 230, 230)), (8, 9))
        if not args.dream_only:
            screen.blit(font.render("REAL GAME", True, (230, 230, 230)), (S + 24, 9))
        pygame.display.flip()
        clock.tick(args.fps)
    pygame.quit()


if __name__ == "__main__":
    main()
