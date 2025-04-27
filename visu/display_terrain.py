import numpy as np
import plotly.graph_objects as go
from PIL import Image

# Load image
img = Image.open('../dla/results/terrain/terrain_blurred.png').convert('L')
Z = np.array(img)

# Create x and y coordinates
x = np.linspace(0, Z.shape[1]-1, Z.shape[1])
y = np.linspace(0, Z.shape[0]-1, Z.shape[0])
X, Y = np.meshgrid(x, y)

# Create the surface plot
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='gray')])

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
