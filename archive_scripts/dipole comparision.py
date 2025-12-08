import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.linalg import eigh_tridiagonal

# Import from your uploaded files
from potential import VGASW_total_debye, solve_gasw_parameters
from continuum import compute_continuum_state

# ==============================================================================
# 1. FIX: Redefine Dipole Integration (Corrects the Grid Bug)
# ==============================================================================
def correct_dipole_element(r_cont, u_cont, r_bound, u_bound):
    """
    Calculates dipole matrix element with CORRECT interpolation order.
    Fixes the bug in the uploaded cross_section.py
    """
    # Fix: Interpolate SMOOTH continuum onto DENSE bound grid
    # (The uploaded code did the reverse, which loses atomic details)
    u_cont_interp = np.interp(r_bound, r_cont, u_cont, left=0.0, right=0.0)
    
    # Integrate on the dense bound grid
    integrand = u_cont_interp * r_bound * u_bound
    D = trapezoid(integrand, r_bound)
    
    return D

# ==============================================================================
# 2. Solver for Argon 3p
# ==============================================================================
def solve_ar_3p(A, U, mu):
    R_max = 60.0
    N = 6000
    r = np.linspace(1e-5, R_max, N)
    dr = r[1] - r[0]
    
    # Argon Potential + Centrifugal (l=1)
    V = VGASW_total_debye(r, A, U, mu, species='Ar')
    V_eff = V + 1.0 * (1.0 + 1.0) / (2.0 * r**2)
    
    # Diagonalize
    k_const = 1.0 / (2.0 * dr**2)
    d = 2.0 * k_const + V_eff[1:-1]
    e = -k_const * np.ones(len(d) - 1)
    
    w, v = eigh_tridiagonal(d, e, select='i', select_range=(0, 2))
    
    # Argon 3p is Index 1 (2p is Index 0)
    idx = 1
    u = np.zeros_like(r)
    u[1:-1] = v[:, idx]
    u /= np.sqrt(trapezoid(u**2, r))
    
    return r, u, w[idx]

# ==============================================================================
# 3. Main Plotting Routine
# ==============================================================================
def plot_saha_reproduction():
    print(">>> Reproducing Saha Figure 11 (Argon Dipole Matrix Elements)...")
    
    # Parameters from Saha Paper (Fig 11)
    DEPTH = 0.56  # Intermediate confinement
    MU = 0.0      # Pure atomic potential (no plasma)
    
    # Get A/U parameters
    A_val, U_val = solve_gasw_parameters(DEPTH, sigma=1.70)
    
    # Solve Bound State
    r_b, u_b, E_b = solve_ar_3p(A_val, U_val, MU)
    print(f"    Argon 3p Energy: {E_b:.4f} a.u.")
    
    # Energy Grid (0.02 - 5.0 a.u.)
    energies = np.linspace(0.02, 5.0, 150)
    
    saha_s = []
    saha_d = []
    
    print("    Computing Dipoles...")
    for E_pe in energies:
        k = np.sqrt(2 * E_pe)
        
        # s-channel (l=0)
        r_s, u_s, _ = compute_continuum_state(E_pe, 0, 'Ar', A_val, U_val, MU)
        D_s = abs(correct_dipole_element(r_s, u_s, r_b, u_b))
        
        # d-channel (l=2)
        r_d, u_d, _ = compute_continuum_state(E_pe, 2, 'Ar', A_val, U_val, MU)
        D_d = abs(correct_dipole_element(r_d, u_d, r_b, u_b))
        
        # --- SAHA CONVENTION APPLIED HERE ---
        # 1. Kinematic Scaling: 1/k (from density of states/amplitude sq)
        # 2. Angular Weighting: s=1/4, d=1 (Differential slice)
        
        val_s = (D_s**2) / (4.0 * k)
        val_d = (D_d**2) / k
        
        saha_s.append(val_s)
        saha_d.append(val_d)
        
    # Plotting
    plt.figure(figsize=(8, 6), dpi=120)
    
    plt.plot(energies, saha_s, 'k--', linewidth=1.5, label=r's-wave ($|D_s|^2/4k$)')
    plt.plot(energies, saha_d, 'r-', linewidth=1.5, label=r'd-wave ($|D_d|^2/k$)')
    plt.plot(energies, np.array(saha_s)+np.array(saha_d), 'g-', linewidth=2, alpha=0.5, label='Total')
    
    plt.title(f"Reproduction of Saha Fig. 11\nAr @ GASW Depth {DEPTH} a.u.", fontsize=14)
    plt.xlabel("Photoelectron Energy (a.u.)", fontsize=12)
    plt.ylabel(r"Dipole Strength $|d_{if}|^2$ (arb.)", fontsize=12)
    
    plt.xlim(0, 5.0)
    plt.ylim(0, 0.12) # Matches Saha scale
    
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Saha_Fig11_Reproduction.png')
    print(">>> Plot saved to Saha_Fig11_Reproduction.png")

if __name__ == "__main__":
    plot_saha_reproduction()