from scipy.ndimage import gaussian_filter
from masked_dla import MaskedDla
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import label

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

def preprocess_gramgan_mask(gramgan):
    threshold = 0.5
    gramgan = gramgan.astype(np.float64) / 255.0
    gramgan_small_blur = gaussian_filter(gramgan, sigma=1.0)
    mask = (gramgan_small_blur > threshold).astype(int)
    peaks = peak_local_max(gramgan_small_blur, min_distance=25, threshold_abs=threshold, exclude_border=False)
    # Label connected black regions
    labeled, _ = label(mask)
    # Map which labels contain dots
    #labeled_mask = np.zeros_like(labeled)
    #for y, x in peaks:
    #    label_val = labeled[int(y), int(x)]
    #    if label_val != 0:
    #        labeled_mask[labeled == label_val] = label_val

    # Create a mask keeping only regions with red dots
    labels_with_dots = set()
    for y, x in peaks:
        label_val = labeled[int(y), int(x)]
        if label_val != 0:
            labels_with_dots.add(label_val)
    cleaned_mask = (~np.isin(labeled, list(labels_with_dots))).astype(int)
    # swap x y
    #peaks = np.flip(peaks, axis=1)
    #gramgan_masked = gramgan * mask
    #plt.imshow(gramgan_masked, cmap='gray')
    #plt.scatter(peaks[:, 1], peaks[:, 0], c='red', s=5)  # Scatter plot of peaks
    #plt.show()
    return cleaned_mask, peaks

generate = True
mask = np.array(Image.open('dla/data/gramgan_raw.png').convert('L'))
if generate:
    SIZE = 256
    mask, peaks = preprocess_gramgan_mask(mask)
    #plt.imshow(mask, cmap='gray')
    #plt.scatter(peaks[:, 1], peaks[:, 0], c='red', s=5)  # Scatter plot of peaks
    #plt.show()
    dla = MaskedDla(size=SIZE, start_pos=peaks, base_mask=mask, allow_diagonals=False, num_of_walkers=10, max_filled=0.25, save_vid=True)
    dla_map = dla.generate()
else:
    dla_map = np.array(Image.open('dla/results/dla/raw_grid_gramgan_masked_fast_rings.png').convert('L')) / 255
#dla_map *= mask
#terrain = multi_scale_blur(dla_map.astype(np.float64), 50, 1)
#save_terrain(terrain, "masked_terrain_blurred_gramgan")
