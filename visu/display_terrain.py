import numpy as np
import plotly.graph_objects as go
from PIL import Image

# Load image
img = Image.open('dla/results/terrain/dla_terrain_128.png').convert('L')
Z = np.array(img)
# Flip the image vertically to match the coordinate system
Z = np.flipud(Z)

# Create x and y coordinates
x = np.linspace(0, Z.shape[1]-1, Z.shape[1])
y = np.linspace(0, Z.shape[0]-1, Z.shape[0])
X, Y = np.meshgrid(x, y)

# Create the surface plot
colorscale = [[0, 'rgb(128,128,128)'], [1, 'rgb(128,128,128)']]  # Grey to Grey
# Terrain colorscale
colorscale_t = [
    [0.0, 'rgb(0, 128, 0)'],      # Dark Green (low elevation)
    [0.5, 'rgb(210, 180, 70)'],  # Tan (light brown)
    [0.7, 'rgb(139, 69, 19)'],    # SaddleBrown (darker brown)
    [1.0, 'rgb(255, 250, 250)']   # Snow white (high elevation)
]
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale=colorscale_t)])

# Customize the layout a bit (optional)
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
