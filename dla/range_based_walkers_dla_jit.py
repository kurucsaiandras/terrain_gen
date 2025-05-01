import numpy as np
import numba
from numba import njit, prange
import os
from PIL import Image

@njit
def is_in_range(pos, start_pos, current_range, threshold=0):
    p0 = pos[0] - start_pos[0]
    p1 = pos[1] - start_pos[1]
    r = current_range + threshold
    return (p0 * p0 + p1 * p1) <= (r * r)

@njit
def move(pos, directions, size, start_pos, current_range):
    valid_moves = np.empty((len(directions), 2), dtype=np.int32)
    count = 0
    for d in directions:
        new_pos = pos + d
        if 0 <= new_pos[0] < size and 0 <= new_pos[1] < size:
            if is_in_range(new_pos, start_pos, current_range):
                valid_moves[count] = new_pos
                count += 1
    if count == 0:
        raise ValueError("No valid moves available")
    idx = np.random.randint(0, count)
    return valid_moves[idx]

@njit
def get_neighbors(pos, grid, directions, size):
    neighbors = np.empty((len(directions), 2), dtype=np.int32)
    count = 0
    for d in directions:
        neighbor = pos + d
        if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
            if grid[neighbor[0], neighbor[1]] == 1:
                neighbors[count] = neighbor
                count += 1
    return neighbors[:count]

@njit(parallel=True)
def step(grid, walkers, directions, start_pos, size, current_range, range_step, connected_pairs, connected_index):
    n_walkers = walkers.shape[0]
    spawn_radius = current_range
    updated = False

    for i in prange(n_walkers):
        tile = walkers[i]
        neighbors = get_neighbors(tile, grid, directions, size)
        if len(neighbors) == 0:
            walkers[i] = move(tile, directions, size, start_pos, current_range)
        else:
            idx = np.random.randint(0, len(neighbors))
            neighbor = neighbors[idx]
            connected_pairs[connected_index[0], 0] = neighbor
            connected_pairs[connected_index[0], 1] = tile
            connected_index[0] += 1

            grid[tile[0], tile[1]] = 1

            if not is_in_range(tile, start_pos, current_range, threshold=2):
                updated = True

            # spawn new walker
            angle = np.random.uniform(0, 2 * np.pi)
            tile = np.array([np.cos(angle), np.sin(angle)]) * spawn_radius + start_pos
            walkers[i] = np.clip(np.round(tile).astype(np.int32), 0, size - 1)

    return updated

class RangeBasedWalkersDlaJit:
    def __init__(self, size=64, start_pos=(31, 31), allow_diagonals=False, range_step=2, num_of_walkers=1, max_pairs=10000):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int32)
        self.start_pos = np.array(start_pos, dtype=np.int32)
        self.grid[start_pos] = 1
        if allow_diagonals:
            self.directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0],
                                        [1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=np.int32)
        else:
            self.directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int32)

        self.range_step = range_step
        self.current_range = range_step

        angles = np.random.uniform(0, 2 * np.pi, num_of_walkers)
        self.walkers = np.array([np.cos(angles), np.sin(angles)]).T * self.current_range + self.start_pos
        self.walkers = np.clip(np.round(self.walkers).astype(np.int32), 0, self.size - 1)

        self.max_pairs = max_pairs
        self.connected_pairs = np.zeros((max_pairs, 2, 2), dtype=np.int32)
        self.connected_index = np.array([0], dtype=np.int32)  # mutable scalar

    def expand_range(self):
        num_of_new_walkers = int(self.walkers.shape[0] * (self.range_step / self.current_range))
        self.current_range += self.range_step
        angles = np.random.uniform(0, 2 * np.pi, num_of_new_walkers)
        new_walkers = np.array([np.cos(angles), np.sin(angles)]).T * self.current_range + self.start_pos
        new_walkers = np.clip(np.round(new_walkers).astype(np.int32), 0, self.size - 1)
        self.walkers = np.vstack([self.walkers, new_walkers])

    def generate(self, filename=None):
        while True:
            updated = step(self.grid, self.walkers, self.directions, self.start_pos, self.size,
                           self.current_range, self.range_step, self.connected_pairs, self.connected_index)

            if updated:
                self.expand_range()
            if self.current_range > self.size // 2:
                break

        self.grid *= 255
        self.grid = self.grid.astype(np.uint8)
        if filename:
            output_dir = f"results/dla"
            os.makedirs(output_dir, exist_ok=True)
            img = Image.fromarray(self.grid, mode='L')
            img.save(f"{output_dir}/{filename}.png")
        return self.grid
    