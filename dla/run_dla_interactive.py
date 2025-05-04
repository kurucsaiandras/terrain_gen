
# %%
# Interactive plotting setup for CodeTunnel
%matplotlib inline
from dynamic_scale_dla import DynamicScaleDla
from range_based_dla import RangeBasedDla
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
# %%

terrain = DynamicScaleDla(size=4, start_pos=(1, 1), max_filled=0.2, max_size=64, allow_diagonals=True)

fig, ax = plt.subplots()
img = ax.imshow(terrain.grid, cmap="gray", vmin=0, vmax=1)
ax.axis('off')
plt.title("Terrain Growth")

# Run the expansion loop with updates
while terrain.add_tile():
    img.set_data(terrain.grid)
    ax.set_title(f"Grid size: {terrain.size}x{terrain.size}")
    clear_output(wait=True)
    display(fig)
    #time.sleep(0.01)  # delay to see the update

# %%
terrain = RangeBasedDla(size=128, start_pos=(63, 63), allow_diagonals=False)

fig, ax = plt.subplots()
img = ax.imshow(terrain.grid, cmap="gray", vmin=0, vmax=1)
ax.axis('off')
plt.title("Terrain Growth")

# Run the expansion loop with updates
while terrain.add_tile():
    img.set_data(terrain.grid)
    ax.set_title(f"Grid size: {terrain.size}x{terrain.size}")
    clear_output(wait=True)
    display(fig)
    #time.sleep(0.01)  # delay to see the update
# %%
