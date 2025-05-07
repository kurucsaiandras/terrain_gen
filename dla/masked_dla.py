import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
from scipy.ndimage import label

class MaskedDla:
    def __init__(self, size=64, start_pos=np.array([[31, 31]]), max_filled=0.5, allow_diagonals=False, num_of_walkers=1, base_mask=None, save_vid=False):
        self.save_vid = save_vid
        if save_vid:
            self.video_writer = None
            self.video_initialized = False
            self.video_path = "output_video.avi"
            self.fps = 30
        self.DBG_START_TIME = time.time()
        self.range_step = 2 # How much to increase the range of the mountain and also the width of the spawn rings
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start_pos = start_pos
        if base_mask is not None:
            self.base_mask = base_mask
        else:
            self.base_mask = np.zeros((size, size), dtype=int) # Zeros are the available tiles
        self.mask = self.base_mask.copy()
        self.ranges = np.full(start_pos.shape[0], self.range_step) # Keeping track of spawn ranges for each mountain
        # Check if start is in the mask
        if not np.all(self.mask[start_pos[:, 0], start_pos[:, 1]] == 0):
            raise ValueError("Start position is not in the mask")
        self.grid[start_pos[:, 0], start_pos[:, 1]] = np.arange(1, len(start_pos)+1)
        self.mask[start_pos[:, 0], start_pos[:, 1]] = 1
        self.init_dynamic_masks()
        if allow_diagonals:
            self.directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]])
        else:
            self.directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        #self.connected_pairs = [] # Tree structure to store connected pairs
        empty_tiles = np.argwhere(self.mask == 0)
        self.min_empty_tiles = empty_tiles.shape[0] * (1 - max_filled)
        self.to_fill = empty_tiles.shape[0] - self.min_empty_tiles # Just for progress bar
        self.walkers = np.zeros((num_of_walkers, 2), dtype=int)
        empty_tiles_in_range = np.argwhere(self.spawn_mask == 0)
        for i in range(num_of_walkers):
            self.walkers[i] = empty_tiles_in_range[np.random.choice(empty_tiles_in_range.shape[0])]
    
########################## PLOTTING ##########################

    def init_video_writer(self, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Use 'mp4v' for .mp4
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (width, height))
        self.video_initialized = True
    def finalize_video(self):
        if self.video_writer:
            self.video_writer.release()
            print("Video saved to", self.video_path)
    def plot_all_maps(self):
        # Plot all maps
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(self.grid, cmap='gray')
        axs[0].set_title('Grid')
        axs[1].imshow(self.mask, cmap='gray')
        axs[1].set_title('Mask')
        axs[2].imshow(self.movement_mask, cmap='gray')
        axs[2].set_title('Movement Mask')
        axs[3].imshow(self.spawn_mask, cmap='gray')
        axs[3].set_title('Spawn Mask')
        for walker in self.walkers:
            for ax in axs:
                ax.axis('off')
                ax.scatter(walker[1], walker[0], c='red', s=1)
        plt.show()
    def write_frame(self):
        h, w = self.grid.shape
        pad = 10  # Padding width in pixels
        # List of maps
        maps = [self.grid, self.mask, self.movement_mask, self.spawn_mask]
        num_maps = len(maps)
        # Create canvas with white padding between maps
        canvas_width = w * num_maps + pad * (num_maps - 1)
        canvas = np.zeros((h, canvas_width), dtype=np.uint8)
        if not self.video_initialized:
            self.init_video_writer(canvas_width, h)
        # Place each map with padding
        for i, m in enumerate(maps):
            m = (m > 0) * 255
            x_start = i * (w + pad)
            canvas[:, x_start:x_start + w] = m
        # Draw white padding between maps
        for i in range(1, num_maps):
            x_pad = i * w + (i - 1) * pad
            canvas[:, x_pad:x_pad + pad] = 255  # White padding
        # Convert to color image
        canvas_color = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        # Draw red walkers
        for walker in self.walkers:
            for i in range(num_maps):
                x = walker[1] + i * (w + pad)
                y = walker[0]
                if 0 <= x < canvas_color.shape[1] and 0 <= y < canvas_color.shape[0]:
                    #cv2.circle(canvas_color, (x, y), 1, (0, 0, 255), -1)
                    canvas_color[y, x] = (0, 0, 255)
        # Resize to output size and write frame
        #resized = cv2.resize(canvas_color, self.output_size)
        self.video_writer.write(canvas_color)

############################ FUNCTIONALITY ##########################

    def is_in_range(self, idx, pos):
        # Check if the position is within the current range
        #return np.linalg.norm(pos - self.start_pos) <= self.current_range - threshold
        r = self.ranges[idx] - self.range_step
        p = pos - self.start_pos[idx]
        return np.dot(p, p) <= r * r

    def init_dynamic_masks(self):
        self.movement_mask = np.ones_like(self.mask, dtype=int)
        # Initialize the movement mask with the current ranges
        for i in range(len(self.start_pos)):
            circle_mask = np.ones_like(self.mask, dtype=int)
            height, width = circle_mask.shape
            yy, xx = np.ogrid[:height, :width]
            y, x = self.start_pos[i]
            # Create a circular mask
            distance_squared = (xx - x)**2 + (yy - y)**2
            circle_mask[distance_squared <= self.ranges[i]**2] = 0
            # Keep center of the circle as occupied (1)
            circle_mask[y, x] = 1
            # Update the movement mask with the part of the new circle that lies in the mask
            self.movement_mask = (circle_mask | self.mask) & self.movement_mask
        self.spawn_mask = self.movement_mask.copy() # First, simply spawn in the movement mask

    def update_movement_mask(self, idx):
        circle_mask = np.ones_like(self.mask)
        height, width = circle_mask.shape
        yy, xx = np.ogrid[:height, :width]
        y, x = self.start_pos[idx]
        self.ranges[idx] += self.range_step
        # Create a circular mask
        distance_squared = (xx - x)**2 + (yy - y)**2
        circle_mask[distance_squared <= self.ranges[idx]**2] = 0
        # Update the movement mask with the part of the new circle that lies in the mask
        self.movement_mask = (circle_mask | self.mask) & self.movement_mask
        #plt.imshow(self.mask, cmap='gray')
        #plt.show()
        #plt.imshow(circle_mask, cmap='gray')
        #plt.show()
        #plt.imshow(self.movement_mask, cmap='gray')
        #plt.show()
    
    def update_spawn_mask(self):
        # Recalculate the spawn mask based on the current ranges
        self.spawn_mask = np.ones_like(self.mask, dtype=int)
        for i in range(len(self.start_pos)):
            circle_mask = np.ones_like(self.mask, dtype=int)
            height, width = circle_mask.shape
            yy, xx = np.ogrid[:height, :width]
            y, x = self.start_pos[i]
            # Create a circular mask
            distance_squared = (xx - x)**2 + (yy - y)**2
            own_land_mask = np.zeros_like(self.mask, dtype=int)
            own_land_mask[(distance_squared <= self.ranges[i]**2) & (self.base_mask == 0)] = 1
            labeled, _ = label(own_land_mask)
            parent_land_id = labeled[tuple(self.start_pos[i])]
            circle_mask[(distance_squared >= (self.ranges[i] - self.range_step)**2) & 
                        (distance_squared <= self.ranges[i]**2) & (labeled == parent_land_id)] = 0
            # Keep center of the circle as occupied (1)
            circle_mask[y, x] = 1
            # Update the spawn mask with the part of the new circle that lies in the mask
            self.spawn_mask = (circle_mask | self.mask) & self.spawn_mask
            
    def move(self, pos):
        # Move to a random adjacent tile
        valid_moves = []
        for d in self.directions:
            new_pos = pos + d
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and self.movement_mask[tuple(new_pos)] == 0:
                valid_moves.append(new_pos)
        if valid_moves:
            return valid_moves[np.random.choice(len(valid_moves))]
        else: # HACK: Just respawn in the spawn mask
            empty_tiles_in_range = np.argwhere(self.spawn_mask == 0)
            #self.plot_all_maps()
            return empty_tiles_in_range[np.random.choice(empty_tiles_in_range.shape[0])]
            #print("No valid moves available")
            #raise ValueError("No valid moves available")
    
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
                #raise ValueError("Tile already occupied")
            # Move if it does not have a neighbor
            neighbors = self.get_neighbors(tile)
            if len(neighbors) == 0:
                if time.time() - self.DBG_START_TIME > 0.5:
                    self.plot_all_maps()
                self.walkers[i] = self.move(tile)
            else:
                self.DBG_START_TIME = time.time()
                # Randomly select a neighbor
                neighbor = neighbors[np.random.choice(len(neighbors))]
                # Connect the tile to the neighbor
                #self.connected_pairs.append((tuple(neighbor), tuple(tile)))
                # Add the tile to the grid if inside the mask
                if self.movement_mask[tuple(tile)] == 0:
                    mountain_id = self.grid[tuple(neighbor)]
                    self.grid[tuple(tile)] = mountain_id
                    self.movement_mask[tuple(tile)] = 1 # mark as filled
                    self.spawn_mask[tuple(tile)] = 1
                    self.mask[tuple(tile)] = 1 # mark as filled
                    if not self.is_in_range(mountain_id-1, tile): # IDs start from 1 so subtract for indexing
                        self.update_movement_mask(mountain_id-1) # update the range of the mountain
                        self.update_spawn_mask()
                if self.save_vid:
                    self.write_frame()
                # Choose new point in the mask
                empty_tiles_in_range = np.argwhere(self.spawn_mask == 0)
                empty_tiles = np.argwhere(self.mask == 0)
                self.walkers[i] = empty_tiles_in_range[np.random.choice(empty_tiles_in_range.shape[0])]
                tiles_left = empty_tiles.shape[0] - self.min_empty_tiles
                print(f"Progress: {(1.0 - tiles_left / self.to_fill) * 100.0:.3f} %", end="\r")
                if (empty_tiles.shape[0] - 1) <= self.min_empty_tiles:
                    return False # stop if we are close to the max filled
        return True
    
    def generate(self, filename=None):
        iter = 0
        while self.step():
            iter += 1
            if self.save_vid:
                if iter % 100 == 0:
                    self.write_frame()
            pass
        # grid was used for the mountain IDs, so converting into a mask
        self.grid[self.grid > 0] = 255
        self.grid = self.grid.astype(np.uint8)
        if self.save_vid:
            self.finalize_video()
        if filename:
            output_dir = f"results/dla"
            os.makedirs(output_dir, exist_ok=True)
            img = Image.fromarray(self.grid, mode='L')
            img.save(f"{output_dir}/{filename}.png")
        return self.grid
