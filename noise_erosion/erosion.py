import numpy as np
from noise import pnoise2
import sys
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#plot the heightmap using matplotlib
def plot_3d(heightmap, seed):
    h, w = heightmap.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, heightmap, cmap='terrain', linewidth=0, antialiased=True)

    ax.set_title("Result Terrain")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")

    plt.tight_layout()
    plt.show()

    filename = f"eroded_terrain_{seed}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')


#standard perlin noise generator
def generate_perlin_map(width=100, height=100, scale=20.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=42):
    noise_map = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            noise_val = pnoise2(x / scale,
                                y / scale,
                                octaves=octaves,
                                persistence=persistence,
                                lacunarity=lacunarity,
                                repeatx=1024,
                                repeaty=1024,
                                base=seed)
            noise_map[y][x] = noise_val
    return noise_map

def generate_map_for_erosion(seed=42):
    low = generate_perlin_map(scale=100.0, seed=seed)
    high = generate_perlin_map(scale=20.0, seed=seed)
    map = 0.7 * low + 0.3 * high
    map = (map - map.min()) / (map.max() - map.min())
    return map

#used to apply a change to the map at a given position using bilinear interpolation
#this is done by taking the four nearest grid points and interpolating between them based on the fractional part of the drops position
def apply_bilinear_change(map, x, y, delta):
    h, w = map.shape

    #clamp coordinates to stay within valid range
    x = np.clip(x, 0, w - 2)
    y = np.clip(y, 0, h - 2)

    #get the integer part of the coordinates
    x0 = int(x)
    y0 = int(y)

    #get the fractional part of the coordinates
    fx = x - x0
    fy = y - y0

    #compute bilinear weights based on the distance from the drop to each corner.
    #each weight reflects how close the drop is to a corner
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy

    #apply the change with the weights calculated before
    map[y0, x0] += delta * w00
    map[y0,     x0 + 1] += delta * w10
    map[y0 + 1, x0    ] += delta * w01
    map[y0 + 1, x0 + 1] += delta * w11


#we use bilinear interpolation to get the height of the terrain at a given the drops position which is floating point
#this is done by taking the four nearest grid points and interpolating between them based on the fractional part of the drops position
#this allows us to get a smooth height value even if the droplet is not exactly on a grid point
def bilinear(map, x, y):
    #get map height and width
    h, w = map.shape

    #clamp coordinates to stay within valid range for indexing
    x, y = np.clip(x, 0, w - 2), np.clip(y, 0, h - 2)

    #integer part of the coordinates
    x0, y0 = int(x), int(y)

    #fractional part of the coordinates
    fx, fy = x - x0, y - y0

    #get the four nearest grid points
    map00 = map[y0, x0]
    map01 = map[y0, x0 + 1]
    map10 = map[y0 + 1, x0]
    map11 = map[y0 + 1, x0 + 1]

    #interpolate between the top and bottom edges
    top = map00 * (1 - fx) + map01 * fx
    bottom = map10 * (1 - fx) + map11 * fx

    #interpolate between the left and right edges and return the final height
    #this gives us a smooth height value even if the droplet is not exactly on a grid point
    return top * (1 - fy) + bottom * fy

#used to calculate the gradient of the terrain at a given position
#this is done by taking the height of the terrain at the position and at the 
#four nearest grid points and using central differences to calculate the slope in both directions
def get_gradient(map, x, y):
    dx = bilinear(map, x + 1, y) - bilinear(map, x - 1, y)
    dy = bilinear(map, x, y + 1) - bilinear(map, x, y - 1)
    return np.array([dx, dy])

    
#simulates the droplet movement and erosion process in the map using the parameters given in 'p'
def simulate_droplet(map, p):
    #load the initial parameters
    inertia = p["inertia"]          #inertia
    capacity_factor = p["capacity"] #capacity of the droplet
    min_slope = p["min_slope"]      #minimum slope needed to erode
    evaporation = p["evaporation"]  #evaporation factor	
    deposition = p["deposition"]    #deposition factor
    erosion = p["erosion"]          #erosion factor
    gravity = p["gravity"]          #gravity factor
    volume = p["initial_water"]     #initial volume
    speed = p["initial_speed"]      #initial speed
    max_iter = p["max_iterations"]  #maximum number of iterations

    #initialize the droplet
    sediment = 0.0                   #sediment levels
    direction = np.array([0.0, 0.0]) #direction
    x = np.random.uniform(0, map.shape[1] - 1) #randomly select a position for the droplet
    y = np.random.uniform(0, map.shape[0] - 1)

    for _ in range(max_iter):
        #calculate height and gradient
        height = bilinear(map, x, y)
        gradient = get_gradient(map, x, y)

        #update diretion based on the gradient and inertia
        new_dir = direction * inertia - gradient * (1 - inertia)
        norm = np.linalg.norm(new_dir)
        if norm < 1e-6:
            break #exit if the droplet is stuck
        #normalize the direction vector
        direction = new_dir / norm

        #move drop
        x = x + direction[0] * speed
        y = y + direction[1] * speed

        #if the droplet is out of bounds, break the loop
        if not (1 <= x < map.shape[1] - 2 and 1 <= y < map.shape[0] - 2):
            break

        #calculate the new height using bilinear interpolation
        new_height = bilinear(map, x, y)
        delta_h = height - new_height

        #calculate capacity based based on the slop and speed
        capacity = max(-delta_h * speed * volume * capacity_factor, min_slope)

        if sediment > capacity:
            #deposit excess sediment if its over capacity
            deposit = (sediment - capacity) * deposition
            deposit = min((sediment - capacity) * deposition, 0.01)
            apply_bilinear_change(map, x, y, deposit)
            sediment -= deposit
        else:
            #erode the terrain if the sediment is under capacity
            erode = (capacity - sediment) * erosion
            erode = min((capacity - sediment) * erosion, 0.01, height)
            apply_bilinear_change(map, x, y, -erode)
            sediment += erode

        #update speed based on the height difference and gravity
        speed = np.sqrt(max(speed**2 + delta_h * gravity, 0))

        #update volume
        volume *= (1 - evaporation)

        #break when volume or speed is too low
        if volume < 0.01 or speed < 0.01:
            break

def simulate_erosion(map, params, iterations=50000, seed=42):
    #simulate erosion on the map given map using the parameters in 'params'

    np.random.seed(seed)
    for _ in range(iterations):
        simulate_droplet(map, params)

    map = gaussian_filter(map, sigma=1.0)
    return map

params = {
    #droplet inertia
    #high = smooth curves
    #low = sharp turns
    "inertia": 0.2,

    #sediment capacity factor
    #high = more sediment capacity, less erosion
    #low = less sediment capacity, more erosion
    "capacity": 1.0,

    #minimum slope to erode
    #high = flat terrain erodes less
    #low = flat terrain erodes more
    "min_slope": 0.001,

    #water evaporation factor
    #high = more evaporation, less erosion
    #low = less evaporation, more erosion
    "evaporation": 0.02,

    #sediment deposition factor
    #high = more deposition, less erosion
    #low = less deposition, more erosion
    "deposition": 0.05,

    #sediment erosion factor
    #high = more erosion, less deposition
    #low = less erosion, more deposition
    "erosion": 0.15,

    #gravity factor
    #high = more gravity, water moves faster downhill
    #low = less gravity, water moves slower downhill
    "gravity": 9.81,

    #initial water factor
    #high = more water per droplet, more erosion
    #low = less water per droplet, less erosion
    "initial_water": 1.0,

    #initial droplet speed
    "initial_speed": 1.0,

    #maximum number of iterations for each droplet
    #high = more iterations, more erosion
    #low = less iterations, less erosion
    "max_iterations": 100
}

if __name__ == "__main__":

    #check for an argument to set the seed or use the default seed (42)
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 42

    #create a map using perlin noise to simulate the erosion in
    map = generate_map_for_erosion(seed=seed)

    #simulate erosion on the map using the parameters in 'params'
    map = simulate_erosion(map, params, iterations=50000, seed=seed)
    
    #plot the final map using matplotlib
    plot_3d(map, seed)