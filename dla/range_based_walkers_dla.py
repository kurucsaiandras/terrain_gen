import numpy as np
import os
from PIL import Image

class RangeBasedWalkersDla:
    def __init__(self, size=64, start_pos=(31, 31), allow_diagonals=False, range_step=2, num_of_walkers=1):
        self.size = size
        self.grid = np.zeros((size, size), dtype=float)
        self.start_pos = np.array(start_pos)
        self.grid[start_pos] = 1
        if allow_diagonals:
            self.directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]])
        else:
            self.directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        #self.connected_pairs = [] # Tree structure to store connected pairs
        self.current_range = range_step
        self.range_step = range_step
        angles = np.random.uniform(0, 2 * np.pi, num_of_walkers)
        self.walkers = np.array([np.cos(angles), np.sin(angles)]).T * self.current_range + self.start_pos
        self.walkers = np.clip(np.round(self.walkers).astype(int), 0, self.size - 1)

    def is_in_range(self, pos, threshold=0):
        # Check if the position is within the current range
        #return np.linalg.norm(pos - self.start_pos) <= self.current_range - threshold
        r = self.current_range - threshold
        p = pos - self.start_pos
        return np.dot(p, p) <= r * r
    
    def expand_range(self):
        num_of_new_walkers = int(self.walkers.shape[0] * (self.range_step / self.current_range))
        self.current_range += self.range_step
        angles = np.random.uniform(0, 2 * np.pi, num_of_new_walkers)
        new_walkers = np.array([np.cos(angles), np.sin(angles)]).T * self.current_range + self.start_pos
        new_walkers = np.clip(np.round(new_walkers).astype(int), 0, self.size - 1)
        self.walkers = np.vstack([self.walkers, new_walkers])

    def move(self, pos):
        # Move to a random adjacent tile
        valid_moves = []
        for d in self.directions:
            new_pos = pos + d
            #if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            if self.is_in_range(new_pos):
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
                if self.grid[tuple(neighbor)] != 0:
                    neighbors.append(neighbor)
        return np.array(neighbors)

    def step(self):
        for i, tile in enumerate(self.walkers):
            #if self.grid[tuple(tile)] == 1:
                #raise ValueError("Tile already occupied") # TODO Just move instead
            # Move if it does not have a neighbor
            neighbors = self.get_neighbors(tile)
            if len(neighbors) == 0:
                self.walkers[i] = self.move(tile)
            else:
                # Randomly select a neighbor
                neighbor = neighbors[np.random.choice(len(neighbors))]
                # Connect the tile to the neighbor
                #self.connected_pairs.append((tuple(neighbor), tuple(tile)))
                # Add the tile to the grid
                self.grid[tuple(tile)] = self.grid[tuple(neighbor)] + 1
                # Check if we need to increase the range
                if not self.is_in_range(tile, threshold=2):
                    self.expand_range()
                if self.current_range > self.size // 2:
                    return False
                # Choose new point on the spawn circle
                angle = np.random.uniform(0, 2 * np.pi)
                tile = np.array([np.cos(angle), np.sin(angle)]) * self.current_range + self.start_pos
                self.walkers[i] = np.clip(np.round(tile).astype(int), 0, self.size - 1)
        return True
    
    def generate(self, filename=None):
        while self.step():
            pass
        mask = self.grid > 0
        self.grid[mask] = (self.grid.max() + 1 - self.grid[mask]) / self.grid.max() # values between 0 and 1
        # smooth falloff
        rate = 5
        self.grid[mask] = (1 - 1 / (1 + rate * self.grid[mask])) / self.grid.max() # values between 0 and 1
        self.grid = np.round(self.grid * 255).astype(np.uint8)
        if filename:
            output_dir = f"results/dla"
            os.makedirs(output_dir, exist_ok=True)
            img = Image.fromarray(self.grid, mode='L')
            img.save(f"{output_dir}/{filename}.png")
        return self.grid
