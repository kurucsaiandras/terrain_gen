from range_based_walkers_dla import RangeBasedWalkersDla
import matplotlib.pyplot as plt
import time

terrain = RangeBasedWalkersDla()

fig, ax = plt.subplots()
img = ax.imshow(terrain.grid, cmap="gray", vmin=0, vmax=1)
ax.axis('off')
plt.title("Terrain Growth")

# Run the expansion loop with updates
while terrain.step():
    img.set_data(terrain.grid)
    ax.set_title(f"Grid size: {terrain.size}x{terrain.size}")
    # save to file
    plt.savefig(f"dla.png")

