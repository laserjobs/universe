# main.py

import numpy as np
from collections import deque
import math
import argparse # New import

# ... (all the existing parameters and functions remain the same) ...
# === SIMULATION PARAMETERS ===
GRID_SIZE = 50
NUM_STATES = 5
NUM_STEPS = 1000 # This will be the default
# ... (rest of the parameters) ...

# ... (all the functions from calculate_global_entropy to handle_interactions) ...
# ... (remain exactly the same) ...

def run_simulation(num_steps): # Updated to accept num_steps
    """The main execution loop of the universe."""
    global conscious_clusters_coords, universe_grid, energy_grid

    # INIT
    # ... (init code remains the same) ...

    phi_threshold = PHI_THRESHOLD_INITIAL

    # MAIN LOOP - uses the new num_steps variable
    for step in range(num_steps):
        # ... (the entire main loop logic remains the same) ...

        # Reporting
        if step % 10 == 0: # Changed to 10 for more frequent updates in short test runs
            print(f"Step {step}: Entropy={calculate_global_entropy(universe_grid):.4f} | "
                  f"Conscious={len(conscious_clusters_coords)} | Phi_Thresh={phi_threshold:.1f}")

    print(f"\nCEA-3.0 Living Universe Simulation Complete after {num_steps} steps.")

if __name__ == "__main__":
    # --- New code for handling command-line arguments ---
    parser = argparse.ArgumentParser(description="Run the CEA-3.0 Living Universe Simulation.")
    parser.add_argument('--test', action='store_true', help="Run in short test mode for CI.")
    args = parser.parse_args()

    # Determine the number of steps based on the --test flag
    if args.test:
        print("--- Running in CI Test Mode (10 steps) ---")
        simulation_steps = 10
    else:
        simulation_steps = NUM_STEPS

    run_simulation(num_steps=simulation_steps)
