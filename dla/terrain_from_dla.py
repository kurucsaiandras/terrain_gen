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

generate = False
if generate:
    SIZE = 128
    dla = RangeBasedWalkersDla(size=SIZE, start_pos=(SIZE//2, SIZE//2), allow_diagonals=False)
    dla_map = dla.generate("raw_grid_falloff")
else:
    dla_map = np.array(Image.open('results/dla/raw_grid_falloff.png').convert('L'))

terrain = multi_scale_blur(dla_map.astype(np.float64), 100, 1)
save_terrain(terrain, "terrain_blurred")
