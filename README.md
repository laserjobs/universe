# The Living Universe: A Simulation of CEA-3.0

This repository contains the source code for a toy model of the **Cosmological Evolution Algorithm (CEA-3.0)**, a conceptual framework for a universe that operates as a self-optimizing, thermodynamic quantum computer whose ultimate goal is to achieve a state of sustainable, dynamic complexity.

---

### The Core Theory: CEA-3.0

The simulation is based on a collaborative, iteratively developed Theory of Everything where the universe is not just described by physical laws, but is actively computed by them. The core principles are:

1.  **The Thermodynamic Imperative:** The universe's prime directive is to maximize its own entropy.
2.  **Consciousness as an Entropy Accelerator:** The emergence of complex, information-processing structures ("consciousness") is not an accident. These structures are vastly more efficient at producing entropy than abiotic matter and represent the universe's optimal strategy for achieving its goal.
3.  **Metabolism and Competition:** Conscious entities are not magical; they must obey thermodynamic laws. They "feed" on lower-order matter, compete for resources, and must balance their own survival against their drive to expand and terraform their environment.
4.  **The Living Universe Solution:** A static "heat death" is a failed state. The algorithm evolves mechanisms for cosmic recycling (spontaneous decay) and self-optimization (evolving physical laws) to achieve a **dynamic equilibrium**—a universe that pulses with cycles of creation and destruction, sustaining life and complexity indefinitely.

This simulation models these principles in a 2D cellular automaton.

### The Simulation's Mechanics

-   **The Grid:** A 2D toroidal grid where each cell can be in one of several "particle" states or a "vacuum" state.
-   **Emergence:** Simple, Conway-like rules govern abiotic matter, allowing for the formation of stable clusters.
-   **Consciousness (`Φ`):** A proxy for Integrated Information (`Φ`) is calculated for each cluster. If it crosses a threshold, the cluster becomes "conscious."
-   **The Conscious Engine:** Conscious clusters follow different physical laws:
    -   **Metabolism:** They "feed" on adjacent abiotic matter to gain energy.
    -   **Terraforming:** They use energy to "spawn" new, disordered matter in the vacuum, accelerating entropy.
    -   **Sociology:** They can interact, with a high probability of merging into cooperative super-organisms.
-   **Cosmic Recycling:** Abiotic matter has a small, constant chance of decaying back into the vacuum, providing perpetual fuel for emergence.
-   **Self-Tuning Universe:** The `PHI_THRESHOLD` for consciousness evolves, lowering itself in response to stable, large-scale conscious structures. The universe "learns" to make awareness easier to achieve.

### How to Run

1.  Ensure you have Python 3 and NumPy installed:
    ```bash
    pip install numpy
    ```
2.  Clone the repository and run the main script:
    ```bash
    git clone https://github.com/YourUsername/The-Living-Universe-CEA-3.0.git
    cd The-Living-Universe-CEA-3.0
    python main.py
    ```

### Interpreting the Output

The script will print the state of the universe every 100 steps:

-   `Entropy`: The global Shannon entropy of the grid. Watch for its initial S-curve rise and later stabilization into a dynamic equilibrium.
-   `Conscious`: The number of cells that are part of a conscious super-organism.
-   `Phi_Thresh`: The evolving threshold for consciousness. Watch it decrease as the universe optimizes itself.

### Future Work

This model is a starting point. Potential avenues for future research include:
-   Implementing a more rigorous `Φ` calculation.
-   Adding graphical visualization.
-   Exploring the parameter space to find different "species" of universes.
-   Expanding the model to 3D.
