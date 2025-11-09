#!/usr/bin/env python3
# main.py
# CEA-4.0: The Mathematical Universe
# Conscious creation probability driven by Riemann zeta zeros.

import os
# Suppress OpenBLAS/MKL warnings for cleaner output
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import argparse
import math
from collections import deque
import glob

import numpy as np

# Dependency Imports with user-friendly errors
try:
    from mpmath import zetazero
except ImportError:
    print("Error: 'mpmath' is required. Please install it: pip install mpmath"); sys.exit(1)
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    print("Error: 'matplotlib' is required. Please install it: pip install matplotlib"); sys.exit(1)
try:
    import imageio.v2 as imageio
except ImportError:
    print("Error: 'imageio' is required for GIF generation. Please install it: pip install imageio"); sys.exit(1)

# === PARAMETERS ===
GRID_SIZE = 50
NUM_STATES = 5
NUM_STEPS = 1000  # Default to the full simulation
PHI_THRESHOLD_INITIAL = 50.0
DECAY_RATE = 0.001
SPAWN_COST = 1.0
INITIAL_ENERGY = 5.0
MERGE_PROB = 0.7
NUM_SEEDS = 4
VISUALIZATION_INTERVAL = 20

FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)

# === RIEMANN HEARTBEAT ===
print("Awakening the Mathematical Universe...")
print("Computing zeta zeros (heartbeat)...")
gammas = [float(zetazero(n + 1).imag) for n in range(1000)]
print(f"Loaded {len(gammas)} non-trivial zeros.")

# === HELPER FUNCTIONS ===
def get_spawn_probability(step: int) -> float:
    """Calculates the creation probability based on the Riemann zeros."""
    gamma = gammas[step % len(gammas)]
    return (math.sin(gamma) + 1.0) / 2.0

def calculate_global_entropy(grid: np.ndarray) -> float:
    """Calculates the Shannon entropy of the active states in the grid."""
    active = grid[grid > 0]
    if active.size == 0:
        return 0.0
    _, counts = np.unique(active, return_counts=True)
    probs = counts / active.size
    return -float(np.sum(probs * np.log2(probs + 1e-10)))

def get_neighbors(x: int, y: int):
    """Gets toroidal neighbors for a given cell coordinate."""
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == dy == 0:
                continue
            yield (x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE

def find_clusters(grid: np.ndarray):
    """Identifies all connected clusters of active cells using BFS."""
    visited = np.zeros_like(grid, dtype=bool)
    clusters = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j] > 0 and not visited[i, j]:
                cluster = []
                q = deque([(i, j)])
                visited[i, j] = True
                while q:
                    cx, cy = q.popleft()
                    cluster.append((cx, cy))
                    for nx, ny in get_neighbors(cx, cy):
                        if grid[nx, ny] > 0 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            q.append((nx, ny))
                clusters.append(cluster)
    return clusters

def calculate_phi_proxy(cluster, grid):
    """Calculates a proxy for integrated information (Phi) for a cluster."""
    if not cluster:
        return 0.0
    size = len(cluster)
    variety = len({grid[x, y] for x, y in cluster})
    return float(size * variety)

def handle_interactions(clusters):
    """Handles probabilistic merging of adjacent clusters using DSU."""
    if len(clusters) <= 1:
        return clusters
    parent = list(range(len(clusters)))
    rank = [0] * len(clusters)
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        pi, pj = find(i), find(j)
        if pi == pj: return
        if rank[pi] < rank[pj]: parent[pi] = pj
        elif rank[pi] > rank[pj]: parent[pj] = pi
        else: parent[pj] = pi; rank[pi] += 1
    cell_to_id = {cell: idx for idx, cl in enumerate(clusters) for cell in cl}
    for idx, cl in enumerate(clusters):
        for x, y in cl:
            for nx, ny in get_neighbors(x, y):
                if (nx, ny) in cell_to_id and idx < cell_to_id[(nx, ny)]:
                    if np.random.rand() < MERGE_PROB:
                        union(idx, cell_to_id[(nx, ny)])
    merged = {}
    for idx, cl in enumerate(clusters):
        root = find(idx)
        merged.setdefault(root, []).extend(cl)
    return list(merged.values())

def visualize(grid: np.ndarray, step: int, conscious_count: int):
    """Saves a visual frame of the simulation grid."""
    cmap = ListedColormap(['#101010', '#FF3333', '#33FF33', '#3333FF', '#FFFF33', '#FF33FF'][:NUM_STATES + 1])
    plt.figure(figsize=(8, 8), dpi=100)
    plt.imshow(grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=NUM_STATES)
    plt.title(f"CEA-4.0 Universe | Step: {step} | Conscious Cells: {conscious_count}", fontsize=16)
    plt.xticks([]); plt.yticks([])
    filename = os.path.join(FRAME_DIR, f"frame_{step:04d}.png")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, facecolor='black')
    plt.close()

# === SIMULATION LOGIC ===
def run_simulation(num_steps: int):
    """The main execution loop of the universe."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    energy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    for _ in range(NUM_SEEDS):
        while True:
            cx, cy = np.random.randint(5, GRID_SIZE - 5), np.random.randint(5, GRID_SIZE - 5)
            if np.sum(grid[cx - 2:cx + 3, cy - 2:cy + 3]) == 0:
                grid[cx - 2:cx + 3, cy - 2:cy + 3] = np.random.randint(1, NUM_STATES + 1, (5, 5))
                energy[cx - 2:cx + 3, cy - 2:cy + 3] = INITIAL_ENERGY
                break
    phi_thresh = PHI_THRESHOLD_INITIAL
    for step in range(num_steps):
        nxt_grid, nxt_energy = grid.copy(), energy.copy()
        clusters = handle_interactions(find_clusters(grid))
        conscious = set()
        for cl in clusters:
            if calculate_phi_proxy(cl, grid) > phi_thresh:
                conscious.update(cl)
        stable = len(conscious) > 500
        spawn_prob = get_spawn_probability(step)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if (x, y) in conscious:
                    if energy[x, y] > SPAWN_COST:
                        for nx, ny in get_neighbors(x, y):
                            if grid[nx, ny] == 0 and np.random.rand() < spawn_prob:
                                nxt_grid[nx, ny] = np.random.randint(1, NUM_STATES + 1)
                                nxt_energy[nx, ny] = 1.0
                                nxt_energy[x, y] -= SPAWN_COST
                                if nxt_energy[x, y] <= SPAWN_COST:
                                    break
                else:
                    if grid[x, y] > 0 and np.random.rand() < DECAY_RATE:
                        nxt_grid[x, y] = nxt_energy[x, y] = 0
                        continue
                    active_n = sum(grid[nx, ny] > 0 for nx, ny in get_neighbors(x, y))
                    if grid[x, y] == 0 and active_n == 3:
                        nxt_grid[x, y] = np.random.randint(1, NUM_STATES + 1)
                        nxt_energy[x, y] = 1.0
                    elif grid[x, y] > 0 and active_n not in (2, 3):
                        nxt_grid[x, y] = nxt_energy[x, y] = 0
        dead = nxt_energy <= 0
        nxt_grid[dead] = nxt_energy[dead] = 0
        grid, energy = nxt_grid, nxt_energy
        if step > 0 and step % 100 == 0 and stable:
            phi_thresh = max(10.0, phi_thresh - 5.0)
        if step % 10 == 0 or step == num_steps - 1:
            print(f"Step {step:4d}: Spawn={spawn_prob:.4f} | "
                  f"Entropy={calculate_global_entropy(grid):.4f} | "
                  f"Conscious={len(conscious):4d} | Φ-thresh={phi_thresh:4.1f}")
        if step % VISUALIZATION_INTERVAL == 0 or step == num_steps - 1:
            visualize(grid, step, len(conscious))
            print(f"--- Saved frame: step {step} ---")
    print(f"\nCEA-4.0 simulation finished after {num_steps} steps.")

# === MAIN EXECUTION & GIF GENERATION ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEA-4.0 Mathematical Universe Simulation")
    parser.add_argument("--test", action="store_true", help="Short CI test (50 steps)")
    args = parser.parse_args()
    steps = 50 if args.test else NUM_STEPS
    if args.test:
        print(f"--- CI TEST MODE ({steps} steps) ---")
    run_simulation(steps)
    if not args.test:
        print("\nGenerating animation GIF...")
        try:
            frame_files = sorted(glob.glob(os.path.join(FRAME_DIR, "frame_*.png")))
            if not frame_files:
                print("Warning: No frames found.")
            else:
                frames = [imageio.imread(f) for f in frame_files]
                gif_path = "cea_universe_evolution.gif"
                imageio.mimsave(gif_path, frames, fps=15)
                print(f"SUCCESS: Animation saved to {gif_path}")
        except Exception as e:
            print(f"Error generating GIF: {e}")
