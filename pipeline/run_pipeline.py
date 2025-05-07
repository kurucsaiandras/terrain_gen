import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from PIL import Image
# Add the path to gramgan/src/models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gramgan', 'src')))

from models import Noise, Generator
#from gramgan.src.models import Noise, Generator

def generate_grid_coords(resolution, size, device = None) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0.0, size, resolution, device=device),
        torch.linspace(0.0, size, resolution, device=device),
        indexing='xy',
    )).permute([1, 2, 0])
    
    return coords

def generate_gramgan(resolution, size, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise = Noise(16, 64, device=device)
    model = Generator(128, 16, 1)

    model_path = f"gramgan/models/{model_name}"
    dict = torch.load(model_path, map_location=device)
    model.load_state_dict(dict["generator"])

    model.eval()

    # Generate grid of coordinates
    coords = generate_grid_coords(resolution, size, device=device)
    patches = model(noise, coords) # res x res x 1
    return patches.detach().cpu().squeeze().numpy()

# Load data
RESOLUTION = 256
dla = Image.open('dla/results/terrain/terrain_blurred.png').convert('L')
dla = np.array(dla.resize((RESOLUTION, RESOLUTION), Image.LANCZOS))
gamgan = generate_gramgan(RESOLUTION, 3, "train_state_24000.pt")
plt.imshow(gamgan)
plt.axis('off')
plt.show()
plt.imsave('gamgan.png', gamgan, cmap='gray')
# Normalize both images to the range [0, 1]
dla = (dla - np.min(dla)) / (np.max(dla) - np.min(dla))
gamgan = (gamgan - np.min(gamgan)) / (np.max(gamgan) - np.min(gamgan))
# Simple blending
#terrain = 0.8 * dla + 0.2 * gamgan
# Height based weights blending
terrain = dla * (1.0 + 0.2 * gamgan)

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
    [0.0, 'rgb(0, 128, 0)'],      # Dark Green (low elevation)
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
        aspectratio=dict(x=1, y=1, z=0.2)
    )
)

# Save the figure
fig.show()
