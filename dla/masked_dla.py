import numpy as np
import os
from PIL import Image

class MaskedDla:
    def __init__(self, size=64, start_pos=(31, 31), max_filled=0.5, allow_diagonals=False, num_of_walkers=1, mask=None):
        self.size = size
        self.grid = np.zeros((size, size), dtype=float)
        if mask is not None:
            self.mask = mask
        else:
            self.mask = np.zeros((size, size), dtype=int) # zeros are the available tiles
        self.start_pos = np.array(start_pos)
        # Check if start is in the mask
        if self.mask[tuple(start_pos)] != 0:
            raise ValueError("Start position is not in the mask")
        self.grid[start_pos] = 1
        if allow_diagonals:
            self.directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]])
        else:
            self.directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        #self.connected_pairs = [] # Tree structure to store connected pairs
        empty_tiles = np.argwhere(self.mask == 0)
        self.min_empty_tiles = empty_tiles.shape[0] * (1 - max_filled)
        self.walkers = np.zeros((num_of_walkers, 2), dtype=int)
        for i in range(num_of_walkers):
            self.walkers[i] = empty_tiles[np.random.choice(empty_tiles.shape[0])]

    def move(self, pos):
        # Move to a random adjacent tile
        valid_moves = []
        for d in self.directions:
            new_pos = pos + d
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and self.mask[tuple(new_pos)] == 0:
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
                # Add the tile to the grid if inside the mask
                if self.mask[tuple(tile)] == 0:
                    self.grid[tuple(tile)] = self.grid[tuple(neighbor)] + 1
                    self.mask[tuple(tile)] = 1 # mark as filled
                # Choose new point in the mask
                empty_tiles = np.argwhere(self.mask == 0)
                self.walkers[i] = empty_tiles[np.random.choice(empty_tiles.shape[0])]
                if (empty_tiles.shape[0] - 1) <= self.min_empty_tiles:
                    return False # stop if we are close to the max filled
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
