# Project Overview

This project is a computational physics engine written in Python for calculating the photoionization cross-sections of atoms (H, He, Ne, Ar). It supports modeling atoms under two types of confinement: Endohedral Confinement (C60) using the Gaussian Annular Square Well (GASW) model, and Debye Plasma Screening using a Yukawa-type potential.

The core of the simulation is based on the Single Active Electron (SAE) approximation. It uses effective model potentials (Tong-Lin) and a high-precision, Numba-accelerated Numerov solver to compute the wavefunctions of bound and continuum states.

## Building and Running

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/literate-robot.git
    cd literate-robot
    ```

2.  **Install dependencies:**
    The project requires the following Python libraries: `numpy`, `scipy`, `matplotlib`, `pandas`, `numba`, and `mpmath`. These can be installed using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Running Simulations

The primary way to run simulations and verification is by executing the scripts in the `verification/` directory.

*   **Master Verification:** To run a comprehensive verification suite that compares the simulation results against established experimental and theoretical data for H, He, and Ar, run:
    ```bash
    python verification/master_verification.py
    ```
    This script will generate a 4-panel summary plot in the `results/` directory.

*   **Individual Verification Scripts:** You can also run individual verification scripts for specific atoms:
    ```bash
    python verification/verify_hydrogen.py
    python verification/verify_helium.py
    python verification/verify_argon_cooper.py
    ```

## Development Conventions

*   **Code Style:** The code generally follows PEP 8 standards, with a focus on readability and performance.
*   **Modularity:** The codebase is organized into a `src/` directory with distinct modules for different parts of the physics engine (e.g., `potential.py`, `bound.py`, `continuum.py`).
*   **Performance:** Computationally intensive loops are optimized using the `@njit` decorator from the `numba` library.
*   **Caching:** The `functools.lru_cache` is used to cache results of expensive function calls.
*   **Testing:** The `verification/` directory serves as the primary testing suite, ensuring the physical accuracy of the simulations.

## Theoretical Background

### Saha et al. Reference

*(TODO: Add a summary of the theoretical model from the "Saha et al." reference here. This should include the key equations and assumptions of the model.)*
