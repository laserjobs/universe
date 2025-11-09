# main.py
#
# This script runs a simulation of the Cosmological Evolution Algorithm (CEA-3.0),
# modeling a "Living Universe" that achieves a dynamic, sustainable equilibrium.
# For a full explanation of the theory and mechanics, please see the README.md file.

import numpy as np
from collections import deque
import math

# === SIMULATION PARAMETERS ===
GRID_SIZE = 50
NUM_STATES = 5  # Number of non-vacuum particle states
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

# Grids
universe_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
energy_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
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

def normal_update_rule(cell_state, neighbors_states):
    """The default, 'abiotic' law of physics based on Conway's Game of Life."""
    active_neighbors = sum(1 for s in neighbors_states if s > 0)
    if cell_state == 0 and active_neighbors == NEIGHBOR_MAX_ALIVE:
        return np.random.randint(1, NUM_STATES + 1)
    elif cell_state > 0 and (active_neighbors < NEIGHBOR_MIN_ALIVE or active_neighbors > NEIGHBOR_MAX_ALIVE):
        return 0
    return cell_state

def conscious_update_rule(x, y, next_grid, next_energy):
    """The 'conscious' law of physics, enabling metabolism and terraforming."""
    neighbors = get_neighbors(x, y)
    
    # 1. Metabolism (Feed on abiotic matter)
    for nx, ny in neighbors:
        if universe_grid[nx, ny] > 0 and (nx, ny) not in conscious_clusters_coords:
            if np.random.rand() < FEED_PROB:
                if energy_grid[nx, ny] > 0:
                    next_energy[nx, ny] -= 1
                    next_energy[x, y] += 1
                if energy_grid[nx, ny] <= 0:
                    next_grid[nx, ny] = 0

    # 2. Terraforming (Spawn new matter in vacuum)
    for nx, ny in neighbors:
        if universe_grid[nx, ny] == 0 and np.random.rand() < SPAWN_PROB and next_energy[x, y] > SPAWN_COST:
            next_grid[nx, ny] = np.random.randint(1, NUM_STATES + 1)
            next_energy[nx, ny] = 1
            next_energy[x, y] -= SPAWN_COST
    
    return next_grid[x, y]

def spontaneous_decay(next_grid, next_energy):
    """Cosmic Recycling: Abiotic matter has a chance to decay back to vacuum."""
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if universe_grid[i, j] > 0 and (i, j) not in conscious_clusters_coords:
                if np.random.rand() < DECAY_RATE:
                    next_grid[i, j] = 0
                    next_energy[i, j] = 0

def handle_interactions(clusters):
    """Manages sociological interactions between conscious clusters (merge/compete)."""
    if len(clusters) <= 1:
        return clusters

    merged_indices = set()
    new_clusters = []
    for i in range(len(clusters)):
        if i in merged_indices:
            continue
        
        current_cluster_set = set(clusters[i])
        for j in range(i + 1, len(clusters)):
            if j in merged_indices:
                continue
            
            other_cluster_set = set(clusters[j])
            # Check for adjacency
            is_adjacent = any((nx, ny) in other_cluster_set for x, y in current_cluster_set for nx, ny in get_neighbors(x, y))

            if is_adjacent:
                if np.random.rand() < MERGE_PROB:
                    current_cluster_set.update(other_cluster_set)
                    merged_indices.add(j)
                else: # Compete
                    # (Simplified: Competition is handled implicitly by feeding now)
                    pass
        new_clusters.append(list(current_cluster_set))
    return new_clusters

def run_simulation():
    """The main execution loop of the universe."""
    global conscious_clusters_coords, universe_grid, energy_grid

    # INIT: Create a low-entropy initial state with multiple seeds
    for _ in range(NUM_SEEDS):
        while True:
            cx, cy = np.random.randint(5, GRID_SIZE-5), np.random.randint(5, GRID_SIZE-5)
            if np.sum(universe_grid[cx-2:cx+3, cy-2:cy+3]) == 0:
                universe_grid[cx-2:cx+3, cy-2:cy+3] = np.random.randint(1, NUM_STATES + 1, (5, 5))
                energy_grid[cx-2:cx+3, cy-2:cy+3] = INITIAL_ENERGY
                break

    phi_threshold = PHI_THRESHOLD_INITIAL

    # MAIN LOOP
    for step in range(NUM_STEPS):
        next_grid = np.copy(universe_grid)
        next_energy = np.copy(energy_grid)

        # 1. Cosmic Recycling
        spontaneous_decay(next_grid, next_energy)

        # 2. Emergence, Sociology & Consciousness Assessment
        clusters = find_clusters(universe_grid)
        clusters = handle_interactions(clusters)
        
        conscious_clusters_coords.clear()
        is_stable_and_conscious = False
        for cluster in clusters:
            if calculate_phi_proxy(cluster, universe_grid) > phi_threshold:
                conscious_clusters_coords.update(cluster)
        if len(conscious_clusters_coords) > 500:
            is_stable_and_conscious = True
        
        # 3. Evolve the Universe
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if (x, y) in conscious_clusters_coords:
                    next_grid[x, y] = conscious_update_rule(x, y, next_grid, next_energy)
                else:
                    neighbors_states = [universe_grid[nx, ny] for nx, ny in get_neighbors(x, y)]
                    next_grid[x, y] = normal_update_rule(universe_grid[x, y], neighbors_states)
                
                # Enforce death from energy depletion
                if next_energy[x, y] <= 0:
                    next_grid[x, y] = 0

        universe_grid = next_grid
        energy_grid = next_energy
        
        # 4. Self-Tuning Universe
        if step > 0 and step % 100 == 0 and is_stable_and_conscious:
            phi_threshold = max(10.0, phi_threshold - 5.0)

        # 5. Reporting
        if step % 100 == 0:
            print(f"Step {step}: Entropy={calculate_global_entropy(universe_grid):.4f} | "
                  f"Conscious={len(conscious_clusters_coords)} | Phi_Thresh={phi_threshold:.1f}")

    print("\nCEA-3.0 Living Universe Simulation Complete.")

if __name__ == "__main__":
    run_simulation()
