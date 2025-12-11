"""
Diagnostic: Argon Bound State Spectrum
Checks if we can find the 2p and 3p states in the Tong-Lin potential.
FIXED: Corrected tridiagonal matrix sizes.
"""
import numpy as np
import matplotlib.pyplot as plt
from potential import VGASW_total_debye, SAE_PARAMS
from scipy.linalg import eigh_tridiagonal

def diagnose_argon():
    print("="*60)
    print("DIAGNOSTIC: Argon 3p Bound State")
    print("="*60)
    
    # 1. Inspect Parameters
    if 'Ar' in SAE_PARAMS:
        params = SAE_PARAMS['Ar'].get(1, SAE_PARAMS['Ar'].get('default'))
        print(f"Target: Argon l=1 (p-wave)")
        print(f"Params: Zc={params.Z_c}, a1={params.a1}...")
        print(f"Looking for state index: {params.ground_n_idx} (0=2p, 1=3p ?)")
    
    # 2. Run High-Resolution Solver
    # N=20000 is crucial for the deep Argon potential
    print("\nRunning High-Resolution Solver (N=20000)...")
    
    R_max = 50.0
    N = 20000
    r = np.linspace(1e-6, R_max, N)
    dr = r[1] - r[0]
    
    # Potential
    V = VGASW_total_debye(r, species='Ar', l_wave=1, A=0, U=0)
    V_eff = V + 1 * (1 + 1) / (2 * r**2) # l=1 centrifugal term
    
    # Finite Difference Matrix Construction
    # Inner points only: indices 1 to N-2
    
    # Diagonal elements (d): Size M
    k = 1.0 / (2.0 * dr**2)
    d = 2.0 * k + V_eff[1:-1]
    
    # Off-diagonal elements (e): Size M-1
    # BUG FIX: Size must be len(d) - 1
    e = -k * np.ones(len(d) - 1)
    
    # Solve
    try:
        # Calculate lowest 5 states
        vals, vecs = eigh_tridiagonal(d, e, select='i', select_range=(0, 4))
    except Exception as err:
        print(f"\n❌ Solver failed: {err}")
        return
    
    print("\n--- FOUND EIGENSTATES (l=1) ---")
    for i, E in enumerate(vals):
        print(f"  n_idx={i}: E = {E:.6f} a.u.")
        
    # Identify 3p
    # Theoretical 3p energy is approx -0.579 a.u.
    target_E = -0.579
    best_idx = np.argmin(np.abs(vals - target_E))
    
    print(f"\nTarget Energy (Exp): ~ {target_E:.5f} a.u.")
    print(f"Closest Match:       n_idx={best_idx} (E={vals[best_idx]:.5f} a.u.)")
    
    if abs(vals[best_idx] - target_E) > 0.1:
        print("⚠️ WARNING: Calculated energy is far from experimental value.")
        print("   Check if potential parameters are correct for Ar l=1.")
    
    # 3. Plot the wavefunction
    u_best = vecs[:, best_idx]
    # Simple normalization for plotting
    u_best /= np.sqrt(np.sum(u_best**2) * dr)
    
    # Sign convention
    if u_best[np.argmax(np.abs(u_best))] < 0: 
        u_best = -u_best
    
    plt.figure(figsize=(8, 5))
    plt.plot(r[1:-1], u_best, label=f'Eigenstate {best_idx} (3p)', color='blue')
    plt.axhline(0, color='black', lw=0.5)
    plt.title(f"Argon 3p Wavefunction (E={vals[best_idx]:.4f} a.u.)")
    plt.xlabel("r (a.u.)")
    plt.xlim(0, 15)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    diagnose_argon()