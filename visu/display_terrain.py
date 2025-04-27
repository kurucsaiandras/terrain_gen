import numpy as np
import plotly.graph_objects as go
from PIL import Image

# Load image
img = Image.open('../dla/results/terrain/terrain.png').convert('L')
Z = np.array(img)

# Create x and y coordinates
x = np.linspace(0, Z.shape[1]-1, Z.shape[1])
y = np.linspace(0, Z.shape[0]-1, Z.shape[0])
X, Y = np.meshgrid(x, y)

# Create the surface plot
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='gray')])

# Customize the layout a bit (optional)
fig.update_layout(
    title='Grayscale Image as 3D Surface',
    autosize=True,
    width=800,
    height=800,
    margin=dict(l=0, r=0, b=0, t=30),
    scene=dict(
        zaxis=dict(title='Pixel Intensity'),
        xaxis=dict(title='X'),
        yaxis=dict(title='Y')
    )
)

# Save the figure
fig.write_html('surface_plot.html')
