from scipy.ndimage import gaussian_filter
from masked_dla import MaskedDla
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import label

def save_for_ppt(img, filename, peaks=None):
    plt.figure(figsize=(10, 10))  # Set figure size in inches
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Hide axes
    if peaks is not None:
        plt.scatter(peaks[:, 0], peaks[:, 1], c='red', s=5)

    # Save at high resolution (e.g., 300 DPI)
    plt.savefig(f"results/{filename}_plot.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

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
    save_for_ppt(gramgan_small_blur, "2_gramgan_blurred")
    peaks = peak_local_max(gramgan_small_blur, min_distance=25, threshold_abs=threshold, exclude_border=False)
    # Filter out peaks not in the mask
    peaks = peaks[mask[peaks[:, 0], peaks[:, 1]] == 1]
    peaks_plot = np.flip(peaks, axis=1)
    save_for_ppt(gramgan_small_blur, "3_gramgan_peaks", peaks_plot)
    save_for_ppt(mask, "4_gramgan_masked", peaks_plot)

    # Label connected black regions
    labeled, _ = label(mask)
    labels_with_dots = set()
    for y, x in peaks:
        label_val = labeled[int(y), int(x)]
        if label_val != 0:
            labels_with_dots.add(label_val)
    cleaned_mask = (~np.isin(labeled, list(labels_with_dots))).astype(int)
    save_for_ppt(cleaned_mask < 0.5, "5_gramgan_cleaned_mask", peaks_plot)
    # swap x y
    #peaks = np.flip(peaks, axis=1)
    #gramgan_masked = gramgan * mask
    #plt.imshow(gramgan_masked, cmap='gray')
    #plt.scatter(peaks[:, 1], peaks[:, 0], c='red', s=5)  # Scatter plot of peaks
    #plt.show()
    return cleaned_mask, peaks

generate = True
mask = np.array(Image.open('dla/data/gramgan_raw.png').convert('L'))
save_for_ppt(mask, "1_gramgan_input")
if generate:
    SIZE = 256
    base_mask, peaks = preprocess_gramgan_mask(mask)
    #plt.imshow(mask, cmap='gray')
    #plt.scatter(peaks[:, 1], peaks[:, 0], c='red', s=5)  # Scatter plot of peaks
    #plt.show()
    dla = MaskedDla(size=SIZE, start_pos=peaks, base_mask=base_mask, allow_diagonals=False, num_of_walkers=10, max_filled=0.25, save_vid=True)
    dla_map = dla.generate()
else:
    dla_map = np.array(Image.open('dla/results/dla/raw_grid_gramgan_masked_fast_rings.png').convert('L')) / 255
save_for_ppt(dla_map, "6_dla_map")
dla_map = dla_map.astype(np.float64)
dla_map *= mask
save_for_ppt(dla_map, "7_dla_map_blended")
terrain = multi_scale_blur(dla_map, 50, 1)
save_for_ppt(terrain, "8_dla_map_blurred")
#save_terrain(terrain, "9_masked_terrain_ppt")
