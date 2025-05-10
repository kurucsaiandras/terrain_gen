import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from PIL import Image

def save_for_ppt(img, filename, peaks=None):
    plt.figure(figsize=(10, 10))  # Set figure size in inches
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Hide axes
    if peaks is not None:
        plt.scatter(peaks[:, 0], peaks[:, 1], c='red', s=5)

    # Save at high resolution (e.g., 300 DPI)
    plt.savefig(f"results/{filename}_plot.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# Load data
dla = Image.open('results/terrain/9_masked_terrain_ppt.png').convert('L')
gramgan = Image.open('dla/data/gramgan_raw.png').convert('L')
# Normalize both images to the range [0, 1]
dla = (dla - np.min(dla)) / (np.max(dla) - np.min(dla))
gramgan = (gramgan - np.min(gramgan)) / (np.max(gramgan) - np.min(gramgan))
# Simple blending
terrain = 0.8 * dla + 0.2 * gramgan
# Height based weights blending
#terrain = dla * (1.0 + 0.2 * gramgan)
save_for_ppt(terrain, "10_terrain_blended")

Z = np.array(terrain)
# Flip the image vertically to match the coordinate system
Z = np.flipud(Z)

# Create x and y coordinates
x = np.linspace(0, Z.shape[1]-1, Z.shape[1])
y = np.linspace(0, Z.shape[0]-1, Z.shape[0])
X, Y = np.meshgrid(x, y)

# Create the surface plot
colorscale = [[0, 'rgb(128,128,128)'], [1, 'rgb(128,128,128)']]  # Uniform grey
# Terrain colorscale
colorscale_t = [
    # Ocean blue
    [0.0, 'rgb(0, 0, 255)'],      # Blue (low elevation)
    [0.2, 'rgb(0, 128, 0)'],      # Dark Green (low elevation)
    [0.5, 'rgb(210, 180, 70)'],  # Tan (light brown)
    [0.7, 'rgb(139, 69, 19)'],    # SaddleBrown (darker brown)
    [1.0, 'rgb(255, 250, 250)']   # Snow white (high elevation)
]
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale=colorscale_t)])

# Customize the layout
fig.update_layout(
    title='Terrain Surface',
    autosize=True,
    width=800,
    height=800,
    margin=dict(l=0, r=0, b=0, t=30),
    scene=dict(
        zaxis=dict(title='Height'),
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        aspectratio=dict(x=1, y=1, z=0.1)
    )
)

# Save the figure
fig.show()