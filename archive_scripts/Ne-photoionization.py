import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import trapezoid

# Import your stack
from potential import VGASW_total_debye
from continuum import compute_continuum_state
from cross_section import dipole_matrix_element

# Constants
Ha_to_eV = 27.211386
Bohr_to_Angstrom = 0.529177
ALPHA_FS = 1.0 / 137.036

def solve_all_s_states(species='Ne', A=0.0, U=0.0, mu=0.0, n_states=5):
    """
    Solves for the first `n_states` s-orbitals (l=0).
    Returns list of (Energy, Wavefunction) tuples.
    """
    R_max = 60.0
    N = 6000
    r = np.linspace(1e-5, R_max, N)
    dr = r[1] - r[0]
    
    # Potential
    V = VGASW_total_debye(r, A, U, mu, species=species)
    
    # Hamiltonian
    k = 1.0 / (2.0 * dr**2)
    d = 2.0 * k + V[1:-1]
    e = -k * np.ones(len(d) - 1)
    
    # Solve for top N states
    w, v = eigh_tridiagonal(d, e, select='i', select_range=(0, n_states-1))
    
    states = []
    for i in range(len(w)):
        u = np.zeros_like(r)
        u[1:-1] = v[:, i]
        norm = np.sqrt(trapezoid(u**2, r))
        states.append( (w[i], u / norm) )
        
    return r, states

def identify_atomic_states(r, free_states, conf_states):
    """
    Maps confined states to free states using Max Overlap Principle.
    Returns a dictionary of matched confined states.
    """
    matched = {}
    
    print(f"{'State':<10} | {'Conf. Index':<12} | {'Energy (a.u.)':<15} | {'Overlap %':<10}")
    print("-" * 55)

    for name, (E_free, u_free) in free_states.items():
        best_overlap = -1.0
        best_idx = -1
        
        # Scan all calculated confined states to find the match
        for i, (E_conf, u_conf) in enumerate(conf_states):
            # Calculate overlap <u_free | u_conf>
            overlap = abs(trapezoid(u_free * u_conf, r))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        
        # Store the best match
        matched[name] = conf_states[best_idx]
        print(f"{name:<10} | {best_idx:<12} | {matched[name][0]:<15.4f} | {best_overlap*100:.1f}%")

    return matched

def calculate_gasw_cross_sections():
    print(">>> 1. Configuring Simulation...")
    # GASW Parameters for C60
    params_c60 = {'A': -3.59, 'U': 0.70, 'mu': 0.0} 
    params_free = {'A': 0.0, 'U': 0.0, 'mu': 0.0}
    
    # Energy Grid (0.1 to 40 eV to capture the broad 2s resonance)
    energies_eV = np.linspace(0.1, 35.0, 100)
    energies_au = energies_eV / Ha_to_eV
    
    results = {'E_eV': energies_eV, '1s': {}, '2s': {}}
    
    # ---------------------------------------------------------
    # 2. Solve & Match Bound States
    # ---------------------------------------------------------
    print("\n>>> 2. Solving Bound States...")
    
    # Get Free States (We know Index 0 is 1s, Index 1 is 2s for Free atom)
    r, all_free = solve_all_s_states('Ne', **params_free, n_states=3)
    target_free = {
        '1s': all_free[0],
        '2s': all_free[1]
    }
    
    # Get Confined States (Calculate extra states to account for cage states)
    _, all_conf = solve_all_s_states('Ne', **params_c60, n_states=6)
    
    # Match them
    print("\n>>> Matching Confined States to Atomic Orbitals:")
    target_conf = identify_atomic_states(r, target_free, all_conf)
    
    # ---------------------------------------------------------
    # 3. Calculate Cross Sections
    # ---------------------------------------------------------
    print("\n>>> 3. Calculating Cross Sections (Numerov Continuum)...")
    
    for shell in ['1s', '2s']:
        print(f"    Processing {shell}...")
        
        for case, p, state_dict in [('free', params_free, target_free), ('confined', params_c60, target_conf)]:
            E_b, u_b = state_dict[shell]
            sigmas = []
            
            for E_pe in energies_au:
                # Dipole transition l=0 -> l=1
                l_final = 1
                
                # Solve Continuum
                r_c, u_c, _ = compute_continuum_state(E_pe, l_final, 'Ne', **p)
                
                # Matrix Element
                D = dipole_matrix_element(r_c, u_c, r, u_b)
                
                # Cross Section (Mb)
                E_phot = E_pe - E_b
                if E_phot <= 0:
                    sigmas.append(0.0)
                    continue
                    
                prefactor = (4.0/3.0) * np.pi**2 * ALPHA_FS * E_phot
                sigma_Mb = (prefactor * D**2) * 28.0028
                sigmas.append(sigma_Mb)
            
            results[shell][case] = np.array(sigmas)

    return results

def plot_results(res):
    plt.figure(figsize=(8, 6))
    E = res['E_eV']
    
    # 1s
    plt.plot(E, res['1s']['free'], 'k--', linewidth=1.5, label=r'Ne $1s$ (Free)')
    plt.plot(E, res['1s']['confined'], 'b-', linewidth=2.0, label=r'Ne@C$_{60}$ $1s$ (GASW)')
    
    # 2s
    plt.plot(E, res['2s']['free'], 'k:', linewidth=1.5, label=r'Ne $2s$ (Free)')
    plt.plot(E, res['2s']['confined'], 'r--', linewidth=2.0, label=r'Ne@C$_{60}$ $2s$ (GASW)')
    
    plt.title(r'Ne@C$_{60}$ Photoionization (GASW Model)', fontsize=14)
    plt.xlabel('Photoelectron Energy (eV)', fontsize=12)
    plt.ylabel('Cross Section (Mb)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.xlim(0, 35)
    plt.ylim(0, 2.5) # Adjusted scale for 2s resonance
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ne_c60_gasw_corrected.png', dpi=120)
    print("\n>>> Plot saved to ne_c60_gasw_corrected.png")

if __name__ == "__main__":
    data = calculate_gasw_cross_sections()
    plot_results(data)