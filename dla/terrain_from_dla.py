from scipy.ndimage import gaussian_filter
from range_based_walkers_dla import RangeBasedWalkersDla
import numpy as np
from PIL import Image
import os

def multi_scale_blur(input, iter, step):
    result = np.zeros_like(input)
    for i in range(iter):
        input = gaussian_filter(input, step)
        result += input
    return result / np.max(result)

def save_terrain(terrain, filename):
    terrain *= 255
    terrain = terrain.astype(np.uint8)
    output_dir = f"results/terrain"
    os.makedirs(output_dir, exist_ok=True)
    img = Image.fromarray(terrain, mode='L')
    img.save(f"{output_dir}/{filename}.png")

generate = True
if generate:
    dla = RangeBasedWalkersDla(size=512, start_pos=(255, 255), allow_diagonals=False)
    dla_map = dla.generate("raw_grid_large")
else:
    dla_map = np.array(Image.open('results/dla/raw_grid_large.png').convert('L'))

terrain = multi_scale_blur(dla_map.astype(np.float64), 20, 1)
save_terrain(terrain, "terrain_large")
