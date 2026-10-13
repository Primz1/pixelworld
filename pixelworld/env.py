"""CoinChase: a tiny 64x64 pixel game used as the 'real world' the model learns.

Layout: 16x16 grid of 4px cells. The top 2 rows are a HUD (score pips).
The play area (rows 2..15) has a border wall, a few random interior walls,
the player (blue), an enemy (red) that chases the player, and one coin (yellow).

Rules (the model is never told these; it has to learn them from pixels):
  * actions: 0 stay, 1 up, 2 down, 3 left, 4 right. Walls block movement.
  * touching the coin: score +1 (shown as a pip in the HUD), coin respawns at a random free cell.
  * the enemy steps toward the player every second tick.
  * enemy touching the player: score resets to 0, HUD flashes red for one frame,
    player respawns at a random free cell.
"""
import numpy as np

H = W = 64
CELL = 4
GRID = 16
HUD_ROWS = 2
N_ACTIONS = 5
ACTION_NAMES = ["stay", "up", "down", "left", "right"]
MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # (drow, dcol)
MAX_PIPS = 12

FLOOR = np.array([28, 44, 52], np.uint8)
WALL = np.array([120, 128, 136], np.uint8)
WALL_EDGE = np.array([84, 90, 98], np.uint8)
PLAYER = np.array([60, 140, 255], np.uint8)
PLAYER_CORE = np.array([200, 230, 255], np.uint8)
ENEMY = np.array([230, 60, 60], np.uint8)
ENEMY_CORE = np.array([255, 190, 120], np.uint8)
COIN = np.array([255, 210, 40], np.uint8)
HUD_BG = np.array([10, 12, 16], np.uint8)
HUD_HIT = np.array([150, 20, 20], np.uint8)

# Colours that cover most of the screen. The tokenizer up-weights every other pixel,
# so tiny sprites are not ignored by the reconstruction loss.
BACKGROUND_COLORS = np.stack([FLOOR, WALL, WALL_EDGE, HUD_BG])


class CoinChase:
    def __init__(self, seed=None, n_walls=(8, 16)):
        self.rng = np.random.default_rng(seed)
        self.n_walls = n_walls
        self.reset()

    # ------------------------------------------------------------------ state
    def _random_free(self, exclude=()):
        free = [c for c in np.argwhere(~self.walls).tolist() if tuple(c) not in exclude]
        return np.array(free[self.rng.integers(len(free))])

    def reset(self):
        walls = np.zeros((GRID, GRID), bool)
        walls[:HUD_ROWS] = True  # HUD rows are not walkable
        walls[HUD_ROWS, :] = True
        walls[-1, :] = True
        walls[:, 0] = True
        walls[:, -1] = True
        n = self.rng.integers(self.n_walls[0], self.n_walls[1] + 1)
        for _ in range(n):  # short wall segments
            r = self.rng.integers(HUD_ROWS + 2, GRID - 2)
            c = self.rng.integers(2, GRID - 2)
            if self.rng.random() < 0.5:
                walls[r, c:c + self.rng.integers(1, 4)] = True
            else:
                walls[r:r + self.rng.integers(1, 4), c] = True
        walls[:, -1] = True
        walls[-1, :] = True
        self.walls = walls
        self.player = self._random_free()
        self.enemy = self._random_free(exclude={tuple(self.player)})
        while np.abs(self.enemy - self.player).sum() < 6:
            self.enemy = self._random_free(exclude={tuple(self.player)})
        self.coin = self._random_free(exclude={tuple(self.player), tuple(self.enemy)})
        self.score = 0
        self.t = 0
        self.hit = False
        return self.render()

    def _try_move(self, pos, a):
        dr, dc = MOVES[a]
        nxt = pos + np.array([dr, dc])
        if self.walls[nxt[0], nxt[1]]:
            return pos
        return nxt

    def _enemy_step(self):
        d = self.player - self.enemy
        order = [0, 1] if abs(d[0]) >= abs(d[1]) else [1, 0]
        for axis in order:
            if d[axis] == 0:
                continue
            step = np.zeros(2, int)
            step[axis] = np.sign(d[axis])
            nxt = self.enemy + step
            if not self.walls[nxt[0], nxt[1]]:
                self.enemy = nxt
                return

    def step(self, a):
        """Apply action a. Returns (frame, event) where event is 'coin', 'hit' or ''."""
        self.t += 1
        self.hit = False
        event = ""
        self.player = self._try_move(self.player, int(a))
        if self.t % 2 == 0:
            self._enemy_step()
        if (self.player == self.coin).all():
            self.score = min(self.score + 1, MAX_PIPS)
            self.coin = self._random_free(exclude={tuple(self.player), tuple(self.enemy)})
            event = "coin"
        if (self.player == self.enemy).all():
            self.score = 0
            self.hit = True
            self.player = self._random_free(exclude={tuple(self.enemy), tuple(self.coin)})
            event = "hit"
        return self.render(), event

    # ----------------------------------------------------------------- render
    def render(self):
        img = np.empty((H, W, 3), np.uint8)
        img[:] = FLOOR
        hud_h = HUD_ROWS * CELL
        img[:hud_h] = HUD_HIT if self.hit else HUD_BG
        for i in range(self.score):
            x = 2 + i * 5
            img[2:6, x:x + 4] = COIN
        for r, c in np.argwhere(self.walls[HUD_ROWS:]) + [HUD_ROWS, 0]:
            y, x = r * CELL, c * CELL
            img[y:y + CELL, x:x + CELL] = WALL
            img[y + CELL - 1, x:x + CELL] = WALL_EDGE
        cy, cx = self.coin * CELL
        img[cy + 1:cy + 3, cx + 1:cx + 3] = COIN
        ey, ex = self.enemy * CELL
        img[ey:ey + 4, ex:ex + 4] = ENEMY
        img[ey + 1:ey + 3, ex + 1:ex + 3] = ENEMY_CORE
        py, px = self.player * CELL
        img[py:py + 4, px:px + 4] = PLAYER
        img[py + 1:py + 3, px + 1:px + 3] = PLAYER_CORE
        return img


# --------------------------------------------------------------- measurement
def find_sprite(img, color, core):
    """Return (row, col) grid cell of a 4x4 sprite with the given outer/core colours, or None."""
    for gy in range(HUD_ROWS, GRID):
        for gx in range(GRID):
            y, x = gy * CELL, gx * CELL
            blk = img[y:y + 4, x:x + 4].astype(int)
            if np.abs(blk[0, 0] - color).sum() < 90 and np.abs(blk[1, 1] - core).sum() < 120:
                return gy, gx
    return None


def find_player(img):
    return find_sprite(img, PLAYER, PLAYER_CORE)


def find_enemy(img):
    return find_sprite(img, ENEMY, ENEMY_CORE)


def count_pips(img):
    row = img[3, :, :].astype(int)
    return int(sum(np.abs(row[2 + i * 5 + 1] - COIN).sum() < 120 for i in range(MAX_PIPS)))


# ------------------------------------------------------------------ policies
class ExplorerPolicy:
    """Data-collection policy: a mix of sticky random moves and coin seeking,
    so that coin pickups and enemy hits both show up often in the data."""

    def __init__(self, rng, greedy_prob=0.5, sticky=0.75):
        self.rng = rng
        self.greedy = rng.random() < greedy_prob
        self.sticky = sticky
        self.prev = int(rng.integers(N_ACTIONS))

    def __call__(self, env):
        if self.greedy and self.rng.random() < 0.7:
            d = env.coin - env.player
            options = []
            if d[0] < 0: options.append(1)
            if d[0] > 0: options.append(2)
            if d[1] < 0: options.append(3)
            if d[1] > 0: options.append(4)
            if options:
                a = int(self.rng.choice(options))
                if not (env._try_move(env.player, a) == env.player).all():
                    self.prev = a
                    return a
        if self.rng.random() > self.sticky:
            self.prev = int(self.rng.integers(N_ACTIONS))
        return self.prev
