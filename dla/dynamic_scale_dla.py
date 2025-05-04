import numpy as np
import matplotlib.pyplot as plt

class DynamicScaleDla:
    def __init__(self, size=4, start_pos=(1, 1), max_filled=0.5, max_size=64, allow_diagonals=False):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.grid[start_pos] = 1
        self.start_pos = np.array(start_pos)
        if allow_diagonals:
            self.directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]])
        else:
            self.directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        self.max_filled = max_filled
        self.connected_pairs = [] # Tree structure to store connected pairs
        self.max_size = max_size

    def is_in_range(self, pos, threshold=0):
        # Check if the position is within the current range
        #return np.linalg.norm(pos - self.start_pos) <= self.current_range - threshold
        r = self.size // 2 - threshold
        p = pos - self.start_pos
        return np.dot(p, p) <= r * r

    def move(self, pos):
        # Move to a random adjacent tile
        valid_moves = []
        for d in self.directions:
            new_pos = pos + d
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            #if not self.is_in_range(pos):
                valid_moves.append(new_pos)
        if valid_moves:
            return valid_moves[np.random.choice(len(valid_moves))]
        else:
            raise ValueError("No valid moves available")
    
    def get_neighbors(self, pos):
        # Get indices of neighbors
        neighbors = []
        for d in self.directions:
            neighbor = pos + d
            if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size:
                if self.grid[tuple(neighbor)] == 1:
                    neighbors.append(neighbor)
        return np.array(neighbors)
    
    def upscale_grid(self):
        # Upscale the grid and recalculate connected pairs
        new_connected_pairs = []
        self.size *= 2
        self.start_pos *= 2
        self.grid = np.zeros((self.size, self.size), dtype=int)
        for (i1, j1), (i2, j2) in self.connected_pairs:
            i1_up, i2_up = i1 * 2, i2 * 2
            j1_up, j2_up = j1 * 2, j2 * 2

            i_diff = i2 - i1
            j_diff = j2 - j1
            for n in range(3):
                self.grid[i1_up + n * i_diff, j1_up + n * j_diff] = 1

            mid_i = (i1_up + i2_up) // 2
            mid_j = (j1_up + j2_up) // 2
            new_connected_pairs.append(((i1_up, j1_up), (mid_i, mid_j)))
            new_connected_pairs.append(((mid_i, mid_j), (i2_up, j2_up)))
        self.connected_pairs = new_connected_pairs

    def add_tile(self):
        # Choose random empty tile
        empty_tiles = np.argwhere(self.grid == 0)
        if empty_tiles.shape[0] == 0:
            raise ValueError("No empty tiles available")
        # Randomly select an empty tile
        tile = empty_tiles[np.random.choice(empty_tiles.shape[0])]
        # Move until it has a neighbor
        neighbors = self.get_neighbors(tile)
        while len(neighbors) == 0:
            tile = self.move(tile)
            neighbors = self.get_neighbors(tile)
        # Randomly select a neighbor
        neighbor = neighbors[np.random.choice(len(neighbors))]
        # Connect the tile to the neighbor
        self.connected_pairs.append((tuple(neighbor), tuple(tile)))
        # Add the tile to the grid
        self.grid[tuple(tile)] = 1
        # Check if we need to upscale the grid
        #if (empty_tiles.shape[0] - 1) <= self.grid.size * (1 - self.max_filled):
        if not self.is_in_range(tile, threshold=2):
            if self.size == self.max_size:
                return False
            self.upscale_grid()
        return True

