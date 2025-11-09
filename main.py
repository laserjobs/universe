# main.py
#
# This script runs a simulation of the Cosmological Evolution Algorithm (CEA-3.0),
# modeling a "Living Universe" that achieves a dynamic, sustainable equilibrium.
# Version 2.0: Corrected main loop logic for stable execution.

import numpy as np
from collections import deque
import argparse

# === SIMULATION PARAMETERS ===
GRID_SIZE = 50
NUM_STATES = 5
NUM_STEPS = 1000
PHI_THRESHOLD_INITIAL = 50.0
NEIGHBOR_MIN_ALIVE = 2
NEIGHBOR_MAX_ALIVE = 3
NUM_SEEDS = 4
DECAY_RATE = 0.001
FEED_PROB = 0.1
SPAWN_PROB = 0.5
SPAWN_COST = 1
INITIAL_ENERGY = 5
MERGE_PROB = 0.7

# Grids (will be initialized in main)
universe_grid = None
energy_grid = None
conscious_clusters_coords = set()

def calculate_global_entropy(grid):
    """Calculates the Shannon entropy of the grid."""
    _, counts = np.unique(grid, return_counts=True)
    probabilities = counts / (GRID_SIZE * GRID_SIZE)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def get_neighbors(x, y):
    """Gets the coordinates of the 8 neighbors of a cell with toroidal wrapping."""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0: continue
            nx, ny = (x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE
            neighbors.append((nx, ny))
    return neighbors

def find_clusters(grid):
    """Finds all contiguous clusters of non-vacuum cells using BFS."""
    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
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
    """Calculates a simplified proxy for Integrated Information (Φ)."""
    if not cluster: return 0.0
    size = len(cluster)
    states = [grid[x, y] for x, y in cluster]
    variety = len(set(states))
    return float(size * variety)

def handle_interactions(clusters):
    """Manages sociological interactions between conscious clusters (merge/compete)."""
    if len(clusters) <= 1: return clusters
    # This is a simplified merge logic for stability
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

def run_simulation(num_steps):
    """The main execution loop of the universe."""
    global conscious_clusters_coords, universe_grid, energy_grid
    
    # INIT Grids
    universe_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    energy_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # INIT Seeds
    for _ in range(NUM_SEEDS):
        while True:
            cx, cy = np.random.randint(5, GRID_SIZE-5), np.random.randint(5, GRID_SIZE-5)
            if np.sum(universe_grid[cx-2:cx+3, cy-2:cy+3]) == 0:
                universe_grid[cx-2:cx+3, cy-2:cy+3] = np.random.randint(1, NUM_STATES + 1, (5, 5))
                energy_grid[cx-2:cx+3, cy-2:cy+3] = INITIAL_ENERGY
                break

    phi_threshold = PHI_THRESHOLD_INITIAL

    # MAIN LOOP
    for step in range(num_steps):
        next_grid = np.copy(universe_grid)
        next_energy = np.copy(energy_grid)

        # 1. EMERGENCE & SOCIOLOGY
        clusters = find_clusters(universe_grid)
        clusters = handle_interactions(clusters)
        
        conscious_clusters_coords.clear()
        is_stable_and_conscious = False
        for cluster in clusters:
            if calculate_phi_proxy(cluster, universe_grid) > phi_threshold:
                conscious_clusters_coords.update(cluster)
        if len(conscious_clusters_coords) > 500:
            is_stable_and_conscious = True

        # 2. APPLY BACKGROUND PHYSICS (ABIOTIC RULES)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if (x, y) in conscious_clusters_coords: continue # Skip conscious cells for now
                
                # Spontaneous Decay
                if universe_grid[x, y] > 0 and np.random.rand() < DECAY_RATE:
                    next_grid[x, y] = 0
                    next_energy[x, y] = 0
                    continue
                
                # Normal Update Rule
                neighbors_states = [universe_grid[nx, ny] for nx, ny in get_neighbors(x, y)]
                next_grid[x, y] = normal_update_rule(universe_grid[x, y], neighbors_states)
        
        # 3. APPLY CONSCIOUS ACTIONS (METABOLISM & TERRAFORMING)
        for x, y in list(conscious_clusters_coords):
            # Metabolism (Feed on neighbors)
            for nx, ny in get_neighbors(x, y):
                if universe_grid[nx, ny] > 0 and (nx, ny) not in conscious_clusters_coords and np.random.rand() < FEED_PROB:
                    if next_energy[nx, ny] > 0:
                        next_energy[nx, ny] -= 1
                        next_energy[x, y] += 1

            # Terraforming (Spawn new matter)
            for nx, ny in get_neighbors(x, y):
                if universe_grid[nx, ny] == 0 and np.random.rand() < SPAWN_PROB and next_energy[x, y] > SPAWN_COST:
                    next_grid[nx, ny] = np.random.randint(1, NUM_STATES + 1)
                    next_energy[nx, ny] = 1
                    next_energy[x, y] -= SPAWN_COST

        # 4. FINAL STATE UPDATE (ENFORCE DEATH, UPDATE GRIDS)
        energy_grid = next_energy
        death_mask = (energy_grid <= 0)
        next_grid[death_mask] = 0
        universe_grid = next_grid
        
        # 5. SELF-TUNING UNIVERSE
        if step > 0 and step % 100 == 0 and is_stable_and_conscious:
            phi_threshold = max(10.0, phi_threshold - 5.0)

        # 6. REPORTING
        if step % 10 == 0 or step == num_steps - 1:
            print(f"Step {step}: Entropy={calculate_global_entropy(universe_grid):.4f} | "
                  f"Conscious={len(conscious_clusters_coords)} | Phi_Thresh={phi_threshold:.1f}")

    print(f"\nCEA-3.0 Living Universe Simulation Complete after {num_steps} steps.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CEA-3.0 Living Universe Simulation.")
    parser.add_argument('--test', action='store_true', help="Run in short test mode for CI.")
    args = parser.parse_args()

    if args.test:
        print("--- Running in CI Test Mode (20 steps) ---")
        simulation_steps = 20
    else:
        simulation_steps = NUM_STEPS

    run_simulation(num_steps=simulation_steps)
