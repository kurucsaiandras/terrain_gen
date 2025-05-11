#for arguments
import sys

#for calculations and stuff
import numpy as np

#numba stuff for speeding up the erosion simulation
from numba import njit
from numba.typed import Dict
from numba import types

#for generating the map
from noise import pnoise2
from scipy.ndimage import gaussian_filter

#for the biome map
import matplotlib.pyplot as plt

#for saving the timelapse video
import cv2
import pyvista as pv


#standard perlin noise generator
def generate_perlin_map(width, height, scale=20.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=42):
    seed %= 512
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

#generates a map for erosion using perlin noise
def generate_map_for_erosion(seed, width, height):
    scale_factor = width / 100.0
    low = generate_perlin_map(width, height, scale=100.0*scale_factor, seed=seed)
    high = generate_perlin_map(width, height, scale=20.0*scale_factor, seed=seed)
    map = 1.0 * low + 0.0 * high
    map = (map - map.min()) / (map.max() - map.min())
    map = gaussian_filter(map, sigma=1.2)
    return map

#used to apply a change to the map at a given position using bilinear interpolation
#this is done by taking the four nearest grid points and interpolating between them based on the fractional part of the drops position
@njit
def apply_bilinear_change(map, x, y, delta):
    h, w = map.shape

    #get the integer part of the coordinates
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

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
    map[y0, x1] += delta * w10
    map[y1, x0] += delta * w01
    map[y1, x1] += delta * w11


#we use bilinear interpolation to get the height of the terrain at a given the drops position which is floating point
#this is done by taking the four nearest grid points and interpolating between them based on the fractional part of the drops position
#this allows us to get a smooth height value even if the droplet is not exactly on a grid point
@njit
def bilinear(map, x, y):
    #get map height and width
    h, w = map.shape

    #integer part of the coordinates
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    #fractional part of the coordinates
    fx, fy = x - x0, y - y0

    #get the four nearest grid points
    map00 = map[y0, x0]
    map01 = map[y0, x1]
    map10 = map[y1, x0]
    map11 = map[y1, x1]

    #interpolate between the top and bottom edges
    top = map00 * (1 - fx) + map01 * fx
    bottom = map10 * (1 - fx) + map11 * fx

    #interpolate between the left and right edges and return the final height
    #this gives us a smooth height value even if the droplet is not exactly on a grid point
    return top * (1 - fy) + bottom * fy

#used to calculate the gradient of the terrain at a given position
#this is done by taking the height of the terrain at the position and at the 
#four nearest grid points and using central differences to calculate the slope in both directions
@njit
def get_gradient(map, x, y):

    w = map.shape[1]
    h = map.shape[0]

    #check if the droplet is out of bounds
    if x == w - 1 or y == h - 1:
        return 0.0, 0.0

    #get the grid points around the droplet
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0+1
    y1 = y0+1

    #fractional part
    fx = x - x0
    fy = y - y0

    #get the values at the four nearest grid points
    map00 = map[y0, x0]
    map01 = map[y0, x1]
    map10 = map[y1, x0]
    map11 = map[y1, x1]

    #now we calculate the gradient in the droplet position
    grad = ((map01 - map00)*(1-fy) + (map11 - map10)*fy,
            (map10 - map00)*(1-fx) + (map11 - map01)*fx)

    dx, dy = grad[0], grad[1]

    return dx, dy

#simulates the droplet movement and erosion process in the map using the parameters given in 'p'
@njit
def simulate_droplet(map, p, rng):
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
    sediment = 0.0                  #sediment levels
    x_dir, y_dir = 0.0, 0.0         #direction of the droplet

    #get random position for the droplet
    x = rng[0] * (map.shape[1] - 1)
    y = rng[1] * (map.shape[0] - 1)

    for _ in range(max_iter):
        #get droplet height and direction of flow
        height = bilinear(map, x, y)
        dx, dy = get_gradient(map, x, y)

        #update the droplets position
        # Update direction with inertia and gradient
        x_dir = x_dir * inertia - dx * (1 - inertia)
        y_dir = y_dir * inertia - dy * (1 - inertia)

        #normalize the direction
        norm = np.sqrt(x_dir * x_dir + y_dir * y_dir)

        #if the droplet is not moving, we set a random direction
        #this happens mostly when the droplet spawns on a flat area
        if norm < 1e-5:
            x_dir = rng[0] * 2 - 1
            y_dir = rng[1] * 2 - 1
            norm = np.sqrt(x_dir * x_dir + y_dir * y_dir)
        x_dir /= norm
        y_dir /= norm

        #move the droplet
        x_new = x + x_dir
        y_new = y + y_dir

        #stop if the droplet is out of bounds
        if x < 1 or x >= map.shape[1] - 2 or y < 1 or y >= map.shape[0] - 2:
            break

        #get new height
        new_height = bilinear(map, x_new, y_new)
        delta_height = new_height - height

        #calculate capacity
        capacity = max(-delta_height, min_slope) * speed * volume * capacity_factor

        if capacity < sediment:
            #more sediment than capacity, so we deposit
            depos = (sediment - capacity) * deposition
            apply_bilinear_change(map, x, y, depos)
            sediment -= depos
        if capacity > sediment:
            #more capacity than sediment, so we erode
            eros = min((capacity - sediment)*erosion, -delta_height)
            apply_bilinear_change(map, x, y, -eros)
            sediment += eros

        #update the droplet position
        x = x_new
        y = y_new

        #update speed
        speed = np.sqrt(speed * speed + gravity * delta_height)

        volume = volume * (1 - evaporation)
        
        #if no more water, break
        if volume < 0.01:
            break

#simulates the erosion process on the map using the parameters given in 'params'
def simulate_erosion(map, params, iterations=50000):
    #simulate erosion on the map given map using the parameters in 'params'

    snapshots = []
    save_interval = np.floor(iterations / 240)

    rng_seeds = np.random.random((iterations, 2))
    for i in range(iterations):
        simulate_droplet(map, params, rng_seeds[i])

        if (i % save_interval == 0):
            #make a copy of that map and apply some filters to it to smooth it out
            snap = map.copy()
            snap = np.clip(snap, 0.0, 1.0)
            snap = (snap - snap.min()) / (snap.max() - snap.min())
            snapshots.append(snap)

    map = np.clip(map, 0.0, 1.0)
    map = (map - map.min()) / (map.max() - map.min())
    return map, snapshots

#saves a timelapse video of the erosion process using opencv
def save_timelapse_video(snapshots, seed, fps=30, scale=400.0, resolution=(800, 600)):
    filename = f"timelapses/timelapse_3d_{seed}.mp4"
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)

    h, w = snapshots[0].shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    plotter = pv.Plotter(off_screen=True, window_size=resolution)

    for i, snap in enumerate(snapshots):
        Z = snap * scale
        points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (w, h, 1)

        grid["elevation"] = Z.flatten()

        plotter.clear()
        plotter.add_mesh(
            grid,
            scalars="elevation",
            cmap="gist_earth",
            show_edges=False,
            show_scalar_bar=False
        )
        plotter.enable_eye_dome_lighting()

        center = (w // 2, h // 2, 0)
        zoom_factor = 1.2 - (i / len(snapshots)) * 0.5

        cam_x = center[0] + w * zoom_factor
        cam_y = center[1] + h * zoom_factor
        cam_z = scale * 1.5

        plotter.camera_position = [
            (cam_x, cam_y, cam_z),
            center,
            (0, 0, 1)
        ]

        img = plotter.screenshot(return_img=True)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f"Saved 3D timelapse to {filename}")

#creates an interactive 3D plot of the heightmap using pyvista
def preview_3d(heightmap):
    import pyvista as pv
    import numpy as np

    h, w = heightmap.shape

    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    Z = heightmap * 400

    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (w, h, 1)

    grid["elevation"] = Z.flatten()

    plotter = pv.Plotter()
    plotter.add_mesh(
        grid,
        scalars="elevation",
        cmap="gist_earth",
        show_edges=False,
        show_scalar_bar=False
    )
    plotter.enable_eye_dome_lighting()
    plotter.show()

#creates a biome map using the heightmap
def biome_colormap(heightmap):
    h, w = heightmap.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            val = heightmap[y, x]
            if val < 0.2:
                rgb[y, x] = [0.2, 0.4, 1.0]  #water
            elif val < 0.35:
                rgb[y, x] = [0.9, 0.8, 0.6]  #beach
            elif val < 0.6:
                rgb[y, x] = [0.1, 0.6, 0.2]  #grass
            elif val < 0.8:
                rgb[y, x] = [0.4, 0.3, 0.2]  #rock
            else:
                rgb[y, x] = [1.0, 1.0, 1.0]  #snow
    filename = f"biomes/biome_map_{seed}.png"
    plt.imsave(filename, rgb)
    print(f"Biome map saved as biome_map_{seed}.png")

#exports the heightmap to an obj file
def export_obj(heightmap, filename="terrain.obj", scale=1.0, height_scale=50.0):
    h, w = heightmap.shape
    vertices = []
    faces = []

    #vertices
    for y in range(h):
        for x in range(w):
            z = heightmap[y, x] * height_scale
            vertices.append(f"v {x * scale} {y * scale} {z}")

    #faces (two triangles per grid square)
    for y in range(h - 1):
        for x in range(w - 1):
            i = y * w + x + 1
            faces.append(f"f {i} {i + 1} {i + w}")
            faces.append(f"f {i + 1} {i + w + 1} {i + w}")

    with open(filename, "w") as f:
        f.write("\n".join(vertices) + "\n")
        f.write("\n".join(faces))

    print(f"Exported OBJ mesh to {filename}")

#parameters for the erosion simulation
params = {
    #droplet inertia
    #high = smooth curves
    #low = sharp turns
    "inertia": 0.4,

    #sediment capacity factor
    #high = more sediment capacity, less erosion
    #low = less sediment capacity, more erosion
    "capacity": 0.08,

    #minimum slope to erode
    #high = flat terrain erodes less
    #low = flat terrain erodes more
    "min_slope": 0.05,

    #water evaporation factor
    #high = more evaporation, less erosion
    #low = less evaporation, more erosion
    "evaporation": 0.05,

    #sediment deposition factor
    #high = more deposition, less erosion
    #low = less deposition, more erosion
    "deposition": 0.1,

    #sediment erosion factor
    #high = more erosion, less deposition
    #low = less erosion, more deposition
    "erosion": 0.1,

    #gravity factor
    #high = more gravity, water moves faster downhill
    #low = less gravity, water moves slower downhill
    "gravity": 9.81,

    #initial water factor
    #high = more water per droplet, more erosion
    #low = less water per droplet, less erosion
    "initial_water": 1.0,

    #initial droplet speed
    "initial_speed": 0.8,

    #maximum number of iterations for each droplet
    #high = more iterations, more erosion
    #low = less iterations, less erosion
    "max_iterations": 75
}

if __name__ == "__main__":
    
    #start with default values
    seed = 42
    width = 512
    height = 512
    save_biome_map = False
    save_timelapse = False
    save_obj = False
    save_npy = False
    preview = False

    #check for an argument to set some parameters
    args = sys.argv[1:]
    for arg in args:
        if arg.startswith("--seed="):
            seed = int(arg.split("=")[1])
        elif arg.startswith("--size="):
            width = int(arg.split("=")[1])
            height = width
        elif arg.startswith("--biome-map"):
            save_biome_map = True
        elif arg.startswith("--timelapse"):
            save_timelapse = True
        elif arg.startswith("--obj"):
            save_obj = True
        elif arg.startswith("--npy"):
            save_npy = True
        elif arg.startswith("--preview"):
            preview = True
        elif arg.startswith("--help"):
            print("Usage: python simulate_erosion.py [--seed=SEED] [--size=SIZE] [--biome-map] [--timelapse] [--obj] [--npy]")
            print("Options:")
            print("  --seed=SEED       Seed for the map generation (default: 42)")
            print("  --size=SIZE       Size of the map (default: 100)")
            print("  --biome-map       Save the biome map (default: False)")
            print("  --timelapse       Save the timelapse video (default: False)")
            print("  --obj             Save the map as an OBJ file (default: False)")
            print("  --npy             Save the map as a NPY file (default: False)")
            print("  --preview         Preview the map in 3D (default: False)")
            print("  --help            Show this help message and exit")
            sys.exit(0)
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for more information.")
            sys.exit(1)

    #generate the map with perlin noise
    map = generate_map_for_erosion(seed, width, height)

    #how many droplets to simulate
    iterations = width * height

    #create a numba typed dictionary to hold the parameters for the erosion simulation
    param_typed = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64
    )

    for key, value in params.items():
        param_typed[key] = value
        
    #simulate erosion on the map using the parameters in 'params'
    map, snapshots = simulate_erosion(map, param_typed, iterations)

    #save the timelapse video of the erosion process
    if save_timelapse:
        save_timelapse_video(snapshots, seed)
    
    #save the biome map
    if save_biome_map:
        biome_colormap(map)

    #export the map to an obj and npy file
    if save_obj:
        export_obj(map, f"obj/terrain_{seed}.obj")
    if save_npy:
        np.save(f"npy/terrain_{seed}.npy", map)
        print(f"Saved map as terrain_{seed}.npy")

    #preview the map in 3D
    if preview:
        preview_3d(map)

    

