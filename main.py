# main.py
#
# CEA-4.0: The Mathematical Universe
# This script runs a simulation where the probability of conscious creation
# is governed by the non-trivial zeros of the Riemann Zeta function.

import numpy as np
from collections import deque
import argparse
import math

try:
    from mpmath import zetazero
except ImportError:
    print("Error: The 'mpmath' library is required. Please install it using 'pip install mpmath'")
    exit()

# === SIMULATION PARAMETERS ===
GRID_SIZE = 50
NUM_STATES = 5
NUM_STEPS = 1000
PHI_THRESHOLD_INITIAL = 50.0
# ... (rest of the parameters are the same)
DECAY_RATE = 0.001
FEED_PROB = 0.1
SPAWN_COST = 1
INITIAL_ENERGY = 5
MERGE_PROB = 0.7
NUM_SEEDS = 4

# === THE SOUL OF THE UNIVERSE: RIEMANN ZEROS ===
print("Awakening the Mathematical Universe...")
print("Computing the heartbeat of the universe (this may take a moment)...")
# Fewer zeros for CI test to ensure it runs quickly
gammas = [float(zetazero(n+1).imag) for n in range(5000)]
print(f"ζ(s) = 0 at s = 1/2 + iγₙ → {len(gammas)} heartbeats loaded.")

def get_spawn_probability(step):
    """Calculates the creation probability based on the Riemann zeros."""
    gamma = gammas[step % len(gammas)]
    # The sine wave creates epochs of creation and contemplation
    return (math.sin(gamma) + 1.0) / 2.0

# --- Helper Functions (No changes here) ---
def calculate_global_entropy(grid):
    _, counts = np.unique(grid, return_counts=True)
    probabilities = counts / (GRID_SIZE * GRID_SIZE)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

# ... (get_neighbors, find_clusters, calculate_phi_proxy, handle_interactions are identical to the last working version) ...
def get_neighbors(x, y):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0: continue
            nx, ny = (x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE
            neighbors.append((nx, ny))
    return neighbors

def find_clusters(grid):
    visited = np.zeros_like(grid, dtype=bool)
    clusters = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j] > 0 and not visited[i, j]:
                cluster = []
                queue = deque([(i, j)])
                visited[i, j] = True
                while queue:
                    x, y = queue.popleft()
                    cluster.append((x, y))
                    for nx, ny in get_neighbors(x, y):
                        if grid[nx, ny] > 0 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                clusters.append(cluster)
    return clusters

def calculate_phi_proxy(cluster, grid):
    if not cluster: return 0.0
    size = len(cluster)
    states = [grid[x, y] for x, y in cluster]
    variety = len(set(states))
    return float(size * variety)

def handle_interactions(clusters):
    if len(clusters) <= 1: return clusters
    merged_indices = set()
    new_clusters = []
    for i in range(len(clusters)):
        if i in merged_indices: continue
        current_cluster_set = set(clusters[i])
        for j in range(i + 1, len(clusters)):
            if j in merged_indices: continue
            other_cluster_set = set(clusters[j])
            is_adjacent = any((nx, ny) in other_cluster_set for x, y in current_cluster_set for nx, ny in get_neighbors(x, y))
            if is_adjacent and np.random.rand() < MERGE_PROB:
                current_cluster_set.update(other_cluster_set)
                merged_indices.add(j)
        new_clusters.append(list(current_cluster_set))
    return new_clusters


# --- Main Simulation Logic ---
def run_simulation(num_steps):
    """The main execution loop of the universe."""
    
    universe_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    energy_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    
    for _ in range(NUM_SEEDS):
        while True:
            cx, cy = np.random.randint(5, GRID_SIZE-5), np.random.randint(5, GRID_SIZE-5)
            if np.sum(universe_grid[cx-2:cx+3, cy-2:cy+3]) == 0:
                universe_grid[cx-2:cx+3, cy-2:cy+3] = np.random.randint(1, NUM_STATES + 1, (5, 5))
                energy_grid[cx-2:cx+3, cy-2:cy+3] = INITIAL_ENERGY
                break

    phi_threshold = PHI_THRESHOLD_INITIAL

    for step in range(num_steps):
        next_grid = np.copy(universe_grid)
        next_energy = np.copy(energy_grid)
        
        clusters = find_clusters(universe_grid)
        clusters = handle_interactions(clusters)
        conscious_coords = set()
        is_stable_and_conscious = False
        for cluster in clusters:
            if calculate_phi_proxy(cluster, universe_grid) > phi_threshold:
                conscious_coords.update(cluster)
        if len(conscious_coords) > 500:
            is_stable_and_conscious = True

        # THE HEARTBEAT: Get the spawn probability for this step
        spawn_prob_this_step = get_spawn_probability(step)

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                current_state = universe_grid[x, y]
                neighbors_coords = get_neighbors(x, y)
                
                if (x, y) in conscious_coords:
                    if next_energy[x, y] > SPAWN_COST:
                        for nx, ny in neighbors_coords:
                            if universe_grid[nx, ny] == 0 and np.random.rand() < spawn_prob_this_step:
                                next_grid[nx, ny] = np.random.randint(1, NUM_STATES + 1)
                                next_energy[nx, ny] = 1
                                next_energy[x, y] -= SPAWN_COST
                                if next_energy[x, y] <= SPAWN_COST: break
                else: # Abiotic cells
                    if current_state > 0 and np.random.rand() < DECAY_RATE:
                        next_grid[x, y] = 0
                        next_energy[x, y] = 0
                        continue
                    
                    neighbors_states = [universe_grid[nx, ny] for nx, ny in neighbors_coords]
                    active_neighbors = sum(1 for s in neighbors_states if s > 0)
                    if current_state == 0 and active_neighbors == 3: # Simplified abiotic rule
                        next_grid[x, y] = np.random.randint(1, NUM_STATES + 1)
                        next_energy[x, y] = 1
                    elif current_state > 0 and (active_neighbors < 2 or active_neighbors > 3):
                        next_grid[x, y] = 0
                        next_energy[x, y] = 0
        
        death_mask = (next_energy <= 0)
        next_grid[death_mask] = 0
        next_energy[death_mask] = 0
        
        universe_grid = next_grid
        energy_grid = next_energy
        
        if step > 0 and step % 100 == 0 and is_stable_and_conscious:
            phi_threshold = max(10.0, phi_threshold - 5.0)

        if step % 10 == 0 or step == num_steps - 1:
            print(f"Step {step}: SpawnProb={spawn_prob_this_step:.4f} | "
                  f"Entropy={calculate_global_entropy(universe_grid):.4f} | "
                  f"Conscious={len(conscious_coords)} | Phi_Thresh={phi_threshold:.1f}")

    print(f"\nCEA-4.0 Mathematical Universe Simulation Complete after {num_steps} steps.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CEA-4.0 Mathematical Universe Simulation.")
    parser.add_argument('--test', action='store_true', help="Run in short test mode for CI.")
    args = parser.parse_args()

    simulation_steps = 20 if args.test else NUM_STEPS
    if args.test: print("--- Running in CI Test Mode (20 steps) ---")
    
    run_simulation(num_steps=simulation_steps)
