from scipy.ndimage import gaussian_filter
from masked_dla import MaskedDla
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
    SIZE = 128
    mask = np.array(Image.open('data/mask.png').convert('L'))
    dla = MaskedDla(size=SIZE, start_pos=(100, 50), mask=mask, allow_diagonals=False, num_of_walkers=10, max_filled=0.25)
    dla_map = dla.generate("raw_grid_masked")
else:
    dla_map = np.array(Image.open('results/dla/raw_grid_masked.png').convert('L'))

terrain = multi_scale_blur(dla_map.astype(np.float64), 50, 1)
save_terrain(terrain, "masked_terrain_blurred")
