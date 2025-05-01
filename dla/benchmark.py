from dynamic_scale_dla import DynamicScaleDla
from range_based_dla import RangeBasedDla
from range_based_walkers_dla_jit import RangeBasedWalkersDlaJit
from range_based_walkers_dla import RangeBasedWalkersDla
import time
import matplotlib.pyplot as plt
import os

def run_dynamic_scale_dla(size=4, start_pos=(1, 1), max_filled=0.25, max_size=64, allow_diagonals=False):
    terrain = DynamicScaleDla(size=size, start_pos=start_pos, max_filled=max_filled, max_size=max_size, allow_diagonals=allow_diagonals)
    # measure time
    start_time = time.time()
    while terrain.add_tile():
        pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    return terrain.grid, elapsed_time

def run_range_based_dla(size=64, start_pos=(31, 31), allow_diagonals=False, range_step=2):
    terrain = RangeBasedDla(size=size, start_pos=start_pos, allow_diagonals=allow_diagonals, range_step=range_step)
    # measure time
    start_time = time.time()
    while terrain.add_tile():
        pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    return terrain.grid, elapsed_time

def run_range_based_walkers_dla(size=64, start_pos=(31, 31), allow_diagonals=False, range_step=2, num_of_walkers=1):
    terrain = RangeBasedWalkersDla(size=size, start_pos=start_pos, allow_diagonals=allow_diagonals, range_step=range_step, num_of_walkers=num_of_walkers)
    # measure time
    start_time = time.time()
    terrain.generate()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return terrain.grid, elapsed_time

def run_range_based_walkers_dla_jit(size=64, start_pos=(31, 31), allow_diagonals=False, range_step=2, num_of_walkers=1):
    terrain = RangeBasedWalkersDlaJit(size=size, start_pos=start_pos, allow_diagonals=allow_diagonals, range_step=range_step, num_of_walkers=num_of_walkers)
    # measure time
    start_time = time.time()
    terrain.generate()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return terrain.grid, elapsed_time

def plot_results(results, elapsed_times, name, size):
    num_results = len(results)
    fig, axes = plt.subplots(1, num_results, figsize=(4*num_results, 5))

    if num_results == 1:
        axes = [axes]

    for i, (result, time) in enumerate(zip(results, elapsed_times)):
        ax = axes[i]
        im = ax.imshow(result, cmap='gray')
        ax.set_title(f"Time: {time:.2f}s")
        ax.axis('off')
        #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_dir = f"results/benchmark/size_{size}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{name}.png")
    plt.close()

    # Print avg time
    avg_time = sum(elapsed_times) / num_results
    print()
    print("#" * 20)
    print(f"Average time for {name}: {avg_time:.2f}s")
    # Print max time
    max_time = max(elapsed_times)
    print(f"Max time for {name}: {max_time:.2f}s")
    print("#" * 20)
    print()

def main():
    results, elapsed_times = [], []
    END_SIZE = 128
    ALLOW_DIAGONALS = False
    NUM_OF_RUNS = 5
    #for i in range(NUM_OF_RUNS):
    #    grid, elapsed_time = run_dynamic_scale_dla(size=4, start_pos=(1, 1), max_filled=0.25, max_size=END_SIZE, allow_diagonals=ALLOW_DIAGONALS)
    #    results.append(grid)
    #    elapsed_times.append(elapsed_time)

    #plot_results(results, elapsed_times, "dynamic_scale_dla")

    #results, elapsed_times = [], []

    for i in range(NUM_OF_RUNS):
        grid, elapsed_time = run_range_based_walkers_dla(size=END_SIZE, start_pos=(END_SIZE // 2, END_SIZE // 2), allow_diagonals=ALLOW_DIAGONALS, range_step=2, num_of_walkers=1)
        results.append(grid)
        elapsed_times.append(elapsed_time)

    plot_results(results, elapsed_times, "range_based_walkers_dla", END_SIZE)

    results, elapsed_times = [], []

    for i in range(NUM_OF_RUNS):
        grid, elapsed_time = run_range_based_walkers_dla_jit(size=END_SIZE, start_pos=(END_SIZE // 2, END_SIZE // 2), allow_diagonals=ALLOW_DIAGONALS, range_step=2, num_of_walkers=1)
        results.append(grid)
        elapsed_times.append(elapsed_time)

    plot_results(results, elapsed_times, "range_based_walkers_dla_jit", END_SIZE)

if __name__ == "__main__":
    main()