from scipy.ndimage import gaussian_filter
from masked_dla import MaskedDla
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt

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
SIZE = 128
if generate:
    dla = MaskedDla(size=SIZE,
                    start_pos=np.array([[SIZE//2, SIZE//2]]),
                    allow_diagonals=False,
                    num_of_walkers=10,
                    max_filled=0.15)
    dla_map = dla.generate(f"raw_grid_{SIZE}")
else:
    dla_map = np.array(Image.open(f"results/dla/raw_grid_{SIZE}.png").convert('L')) / 255
dla_map = dla_map.astype(np.float64)
terrain = multi_scale_blur(dla_map, 50, 1)
save_terrain(terrain, f"dla_terrain_{SIZE}")
