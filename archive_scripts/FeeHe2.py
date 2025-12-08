import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.linalg import eigh_tridiagonal

# Import your stack
from potential import VGASW_total_debye
from continuum import compute_continuum_state
from cross_section import dipole_matrix_element

# Constants
Ha_to_eV = 27.211386
Bohr_to_Ang = 0.529177
Ry_to_eV = 13.60569

# ==============================================================================
# 1. SPECIALIZED SOLVER (To find 2s and 2p specifically)
# ==============================================================================
def solve_specific_state(species, l_target, n_target_nodes, R_max=60.0, N=4000):
    """
    Solves for a specific angular momentum (l) and radial node count (n_r).
    Neon 2p: l=1, nodes=0 (in SAE potential)
    Neon 2s: l=0, nodes=1 (assuming 1s is present deep in the well) 
             OR nodes=0 (if potential is pseudized).
    For Tong-Lin Neon, 2s is usually the excited l=0 state or ground l=0.
    Let's search the first few states.
    """
    r = np.linspace(1e-5, R_max, N)
    dr = r[1] - r[0]
    
    # Potential
    V = VGASW_total_debye(r, 0, 0, 0, species=species)
    V_eff = V + l_target*(l_target+1)/(2*r**2)
    
    # Hamiltonian
    k = 1.0/(2*dr**2)
    d = 2.0*k + V_eff[1:-1]
    e = -k * np.ones(len(d)-1)
    
    # Get first 3 eigenvalues for this l
    w, v = eigh_tridiagonal(d, e, select='i', select_range=(0, 2))
    
    # Find the one with correct binding energy
    # Ne 2p ~ 21.6 eV (0.8 a.u.)
    # Ne 2s ~ 48.5 eV (1.78 a.u.)
    
    chosen_idx = 0
    best_diff = 1e9
    
    target_E = -0.79 if l_target == 1 else -1.78
    
    for i, en in enumerate(w):
        if abs(en - target_E) < best_diff:
            best_diff = abs(en - target_E)
            chosen_idx = i
            
    E_bound = w[chosen_idx]
    u = np.zeros_like(r)
    u[1:-1] = v[:, chosen_idx]
    u /= np.sqrt(trapezoid(u**2, r))
    
    return r, u, E_bound

# ==============================================================================
# 2. CALCULATION ROUTINE
# ==============================================================================
def reproduce_sewell_table():
    print(">>> Generating Neon 2p + 2s Cross Section Table...")
    
    # 1. Solve Bound States
    # Neon 2p (l=1)
    r_2p, u_2p, E_2p = solve_specific_state('Ne', l_target=1, n_target_nodes=0)
    print(f"    Found Ne 2p Energy: {E_2p:.4f} a.u. ({abs(E_2p)*Ha_to_eV:.2f} eV)")
    
    # Neon 2s (l=0)
    r_2s, u_2s, E_2s = solve_specific_state('Ne', l_target=0, n_target_nodes=0)
    print(f"    Found Ne 2s Energy: {E_2s:.4f} a.u. ({abs(E_2s)*Ha_to_eV:.2f} eV)")
    
    # 2. Define Table Energies (from Image provided)
    # Wavelengths in Angstroms
    lambdas = np.array([
        574.93, 520.7, 455.6, 405.0, 364.5, 331.4, 303.8, 
        260.4, 255.77, 227.9, 202.5, 182.3, 165.7, 140.2, 
        121.5, 107.2, 95.9, 86.8
    ])
    
    data = []
    
    # 3. Loop Wavelengths
    for lam in lambdas:
        E_photon_eV = 12398.4 / lam
        E_photon_au = E_photon_eV / Ha_to_eV
        
        # --- CALC 2p (Threshold ~ 21.6 eV) ---
        E_kin_2p = E_photon_au - abs(E_2p)
        sigma_2p = 0.0
        
        if E_kin_2p > 0:
            # Transitions: p -> s (l=0) and p -> d (l=2)
            # Weights: 1 for s, 2 for d
            
            # p->s
            r_s, u_s, _ = compute_continuum_state(E_kin_2p, 0, 'Ne', 0, 0, 0)
            d_s = abs(dipole_matrix_element(r_s, u_s, r_2p, u_2p))
            
            # p->d
            r_d, u_d, _ = compute_continuum_state(E_kin_2p, 2, 'Ne', 0, 0, 0)
            d_d = abs(dipole_matrix_element(r_d, u_d, r_2p, u_2p))
            
            # Formula: 4*pi^2 * alpha * w * (1/3) * [1*Ds^2 + 2*Dd^2]
            # Multiplied by 6 electrons
            prefactor = (4 * np.pi**2 * (1/137.036) * E_photon_au) / 3.0
            sigma_2p_au = prefactor * (1.0*d_s**2 + 2.0*d_d**2)
            sigma_2p = sigma_2p_au * 28.0028 * 6.0 # 6 electrons
            
        # --- CALC 2s (Threshold ~ 48.5 eV) ---
        E_kin_2s = E_photon_au - abs(E_2s)
        sigma_2s = 0.0
        
        if E_kin_2s > 0:
            # Transition: s -> p (l=1) ONLY
            # Weight: 1 (from s to p) but s-state has 2l+1=1 degeneracy
            
            # s->p
            r_p, u_p, _ = compute_continuum_state(E_kin_2s, 1, 'Ne', 0, 0, 0)
            d_p = abs(dipole_matrix_element(r_p, u_p, r_2s, u_2s))
            
            # Formula: 4*pi^2 * alpha * w * [Dp^2]
            # Multiplied by 2 electrons
            prefactor = (4 * np.pi**2 * (1/137.036) * E_photon_au)
            sigma_2s_au = prefactor * (d_p**2)
            sigma_2s = sigma_2s_au * 28.0028 * 2.0 # 2 electrons

        data.append({
            'Lambda(A)': lam,
            'Energy(eV)': E_photon_eV,
            'Sim 2p (Mb)': sigma_2p,
            'Sim 2s (Mb)': sigma_2s,
            'Sim Total (Mb)': sigma_2p + sigma_2s
        })

    # 4. Display
    df = pd.DataFrame(data)
    print("\n>>> SIMULATION RESULTS (Compare to Sewell Table VIII)")
    print(df.to_string(index=False, float_format="%.2f"))
    
    # 5. Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df['Lambda(A)'], df['Sim 2p (Mb)'], 'b--', label='2p (Sim)')
    plt.plot(df['Lambda(A)'], df['Sim 2s (Mb)'], 'r--', label='2s (Sim)')
    plt.plot(df['Lambda(A)'], df['Sim Total (Mb)'], 'k-', linewidth=2, label='Total (Sim)')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Cross Section (Mb)')
    plt.title('Replication of Neon Partial Cross Sections')
    plt.legend()
    plt.gca().invert_xaxis() # High energy on left
    plt.grid(alpha=0.3)
    plt.savefig('Neon_2s_2p_Replication.png')
    print(">>> Plot saved.")

if __name__ == "__main__":
    reproduce_sewell_table()