"""
MAIN SCRIPT — Validate Free Hydrogen Photoionization Cross Section
Updated for Single Active Electron (SAE) Library.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import solvers from the updated library
from potential import VGASW_total_debye
from bound import solve_ground_u
from cross_section import compute_cross_section_spectrum

def main():
    # =====================================================================
    # 0. Configuration
    # =====================================================================
    SPECIES = 'H'  # Can be changed to 'He', 'Ne', 'Ar' (but validation val below is for H)
    
    # =====================================================================
    # 1. Bound State Calculation (SAE)
    # =====================================================================

    print(f"\n================ FREE {SPECIES} BOUND STATE ================\n")

    # Updated signature: passes 'species' and returns 'l_initial'
    r_bound, u_bound, E_bound, l_initial, norm = solve_ground_u(
        VGASW_total_debye,
        species=SPECIES,
        R_max=60.0,
        N=6000,
        A=0.0,  # Free atom (no cage)
        U=0.0,
        mu=0.0  # No plasma screening
    )

    print(f"Computed E_0   = {E_bound:.6f} a.u.")
    print(f"Angular Mom l  = {l_initial} ({'s' if l_initial==0 else 'p'})")
    print(f"Normalization  = {norm:.6f}")
    print(f"Peak location  = {r_bound[np.argmax(u_bound)]:.3f} a.u.\n")


    # =====================================================================
    # 2. Energy Grid for Continuum
    # =====================================================================

    E_min = 0.0001
    E_max = 2.0
    N_points = 80

    # Photoelectron energies
    E_pe = np.linspace(E_min, E_max, N_points)

    # =====================================================================
    # 3. Compute Numerical Cross Section (Parallel)
    # =====================================================================

    print("================ COMPUTING NUMERICAL σ(E) ================\n")
    
    # Updated signature: passes 'l_initial' and 'species'.
    # The solver automatically determines final channels (e.g., s->p, or p->s+d).
    sigma_num, diag = compute_cross_section_spectrum(
        E_pe,
        r_bound,
        u_bound,
        E_bound,
        l_initial,      # Passed from bound state solver
        species=SPECIES,
        A=0.0,
        U=0.0,
        mu=0.0,
        n_workers=None  # Auto-detect CPUs
    )

    print("\nCompleted numerical calculation.")
    print(f"Numeric peak σ = {sigma_num.max():.6f} a.u.")
    print(f"At E_pe        = {E_pe[np.argmax(sigma_num)]:.3f} a.u.\n")


    # =====================================================================
    # 4. VALIDATION (Specific to Hydrogen)
    # =====================================================================

    if SPECIES == 'H':
        sigma_lit = 0.22225  # Exact H(1s) threshold cross section (a.u.)
        sigma_num_threshold = sigma_num[0]

        print("========= LITERATURE VALIDATION (Hydrogen Only) =========\n")
        print(f"Exact literature σ(0)  = {sigma_lit:.5f} a.u.")
        print(f"Your numeric σ(E_min)  = {sigma_num_threshold:.5f} a.u.")
        print(f"Absolute deviation     = {abs(sigma_num_threshold - sigma_lit):.5f} a.u.\n")

        if abs(sigma_num_threshold - sigma_lit) < 0.01:
            print("✔ SUCCESS: Numerical cross section matches literature.\n")
        else:
            print("✘ WARNING: Significant deviation. Check normalization.\n")
    else:
        print(f"Skipping literature validation (Value 0.22225 is for H, you ran {SPECIES}).")


    # =====================================================================
    # 5. Plot Numerical σ(E)
    # =====================================================================

    plt.figure(figsize=(9, 5))
    plt.plot(E_pe, sigma_num, 'b-', lw=2, label=f'{SPECIES} SAE')

    plt.title(f"Photoionization Cross Section: {SPECIES} (SAE Model)", fontsize=14)
    plt.xlabel("Photoelectron Energy  $E_{pe}$  (a.u.)")
    plt.ylabel("Cross Section  σ  (a.u.)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# =====================================================================
# Multiprocessing guard (ESSENTIAL FOR WINDOWS)
# =====================================================================
if __name__ == "__main__":
    main()