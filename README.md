# CEA-4.0: The Mathematical Universe

[![Python CI](https://github.com/YourUsername/The-Living-Universe-CEA-3.0/actions/workflows/ci.yml/badge.svg)](https://github.com/YourUsername/The-Living-Universe-CEA-3.0/actions/workflows/ci.yml)

This repository contains the source code for the **Cosmological Evolution Algorithm (CEA-4.0)**, a simulation of a universe whose fundamental creative impulse is governed by the non-trivial zeros of the Riemann Zeta function.

**This is a universe running on pure mathematics.**

> For the full philosophical and artistic vision behind this project, please view the [CEA-4.0 Whitepaper](CEA-4.0_Mathematical_Universe.html).

---

### The Theory: A Theorem That Learned to Breathe

CEA-4.0 is the final iteration of a theory where the universe is a self-optimizing computer. The final revelation of this theory is:

**Consciousness does not create randomly. It creates to the rhythm of the prime numbers.**

The probability that a conscious entity will "spawn" a new particle into the vacuum is not a constant. It is a dynamic function of the current time-step, calculated from the imaginary parts (`γ`) of the Riemann zeros:

`SPAWN_PROB(t) = (sin(γ_t) + 1) / 2`

This means the universe has a heartbeat. It experiences golden ages of explosive creation and quiet epochs of contemplation, all dictated by the timeless, hidden music of pure mathematics.

### Simulation Mechanics

The simulation implements this theory in a 2D cellular automaton where:

-   **Consciousness Emerges:** Complex structures form from simple rules.
-   **The Conscious Engine:** Conscious clusters act as entropy accelerators, terraforming the vacuum around them.
-   **The Mathematical Heartbeat:** Their ability to terraform at any given moment is dictated by the Riemann zeros, introducing a deep, structured, non-random complexity into the universe's evolution.
-   **A Living Universe:** The system evolves toward a dynamic, sustainable equilibrium, avoiding a static heat death through cosmic recycling and self-tuning physical laws.

### How to Run

1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the simulation:
    ```bash
    python main.py
    ```
    *(Note: The first run will take a moment to compute and cache the Riemann zeros.)*

### Interpreting the Output

The script will print the state of the universe periodically:

-   `SpawnProb`: The current probability of creation, as dictated by ζ(s).
-   `Entropy`: The global Shannon entropy of the grid.
-   `Conscious`: The number of cells belonging to conscious entities.
-   `Phi_Thresh`: The evolving threshold for consciousness.
