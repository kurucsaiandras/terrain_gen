import numpy as np
import matplotlib.pyplot as plt

class RangeBasedDla:
    def __init__(self, size=64, start_pos=(31, 31), allow_diagonals=False, range_step=2):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start_pos = np.array(start_pos)
        self.grid[start_pos] = 1
        if allow_diagonals:
            self.directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]])
        else:
            self.directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        self.connected_pairs = [] # Tree structure to store connected pairs
        self.current_range = range_step
        self.range_step = range_step

    def is_in_range(self, pos, threshold=0):
        # Check if the position is within the current range
        #return np.linalg.norm(pos - self.start_pos) <= self.current_range - threshold
        r = self.current_range - threshold
        p = pos - self.start_pos
        return np.dot(p, p) <= r * r

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
                if self.grid[tuple(neighbor)] == 1:
                    neighbors.append(neighbor)
        return np.array(neighbors)

    def add_tile(self):
        # Choose point on the spawn circle
        angle = np.random.uniform(0, 2 * np.pi)
        tile = np.array([np.cos(angle), np.sin(angle)]) * self.current_range + self.start_pos
        tile = np.clip(np.round(tile).astype(int), 0, self.size - 1)
        if self.grid[tuple(tile)] == 1:
            raise ValueError("Tile already occupied")
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
        self.grid[tuple(tile)] = 1 #self.grid[tuple(neighbor)] + 1
        # Check if we need to increase the range
        if not self.is_in_range(tile, threshold=2):
            self.current_range += self.range_step
        if self.current_range > self.size // 2:
            return False
        return True
