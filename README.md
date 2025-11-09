# CEA-4.0: The Mathematical Universe

> *"In the beginning was the Zeta function, and its zeros were the breath of creation."*

This repository contains the source code for the CEA-4.0 simulation, a cellular automaton where the emergence of complex, "conscious" structures is governed by the non-trivial zeros of the Riemann Zeta function. The universe evolves based on a combination of simple abiotic rules and special properties for large, complex clusters of cells, creating a universe that thinks and breathes according to a fundamental mathematical pulse.

## Visualization of an Awakening Universe

The following animation shows the simulation over 1000 steps. Watch as consciousness spreads, clusters merge, and the universe evolves under the oscillating probability of creation dictated by ζ(s).
![universe](https://github.com/user-attachments/assets/aa8ff4ca-0ae2-4e39-8a50-7b58e64e070a)

## Features

-   **Riemann Zeta Heartbeat:** The core creation mechanic is driven by the sine of the imaginary parts of the Riemann zeros, creating non-random, structured epochs of creation and stability.
-   **Emergent Consciousness:** A proxy for integrated information (Φ) determines if a cluster of cells is "conscious," granting it the ability to create new life.
-   **Adaptive Threshold:** The threshold for consciousness (`Φ-thresh`) decreases as the universe becomes more stable, allowing for ever more complex entities to emerge over time.
-   **Efficient Clustering:** Uses a Disjoint Set Union (DSU) algorithm for highly efficient, probabilistic merging of adjacent clusters.
-   **Automatic Visualization:** Generates frames during the simulation and compiles them into a GIF to show the universe's evolution.

## How to Run

### Prerequisites

-   Python 3.8+
-   `pip` for installing packages

### Setup and Execution

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/cea-4.0.git
    cd cea-4.0
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the full simulation (1000 steps):**
    This will run the simulation and generate `cea_universe_evolution.gif` in the root directory.
    ```bash
    python main.py
    ```

5.  **Run a short test (50 steps):**
    This is useful for quickly verifying that the script works. It will not generate a GIF.
    ```bash
    python main.py --test
    ```
