"""
PRODUCTION SCRIPT: Multi-Species, Multi-Depth, Multi-Density Photoionization Scan.
UPDATED: Added Progress Bars (tqdm) and cleaned up logging.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import freeze_support

# --- Library Imports ---
from potential import VGASW_total_debye
from bound import solve_ground_u
from cross_section import compute_cross_section_spectrum
from continuum import compute_continuum_state 

# --- Progress Bar Import with Fallback ---
try:
    from tqdm import tqdm
except ImportError:
    print("Note: 'tqdm' library not found. Falling back to standard loops.")
    def tqdm(iterable, **kwargs):
        if 'desc' in kwargs: print(f"\nStarted: {kwargs['desc']}")
        return iterable
    # Mock write method for compatibility
    tqdm.write = print

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

TARGET_SPECIES = ['H', 'Ar', 'Ne'] 
DEPTHS = [0.30, 0.56, 1.03]
MU_VALUES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# Energy Grid
E_GRID = np.concatenate([
    np.linspace(0.001, 0.3, 100),
    np.linspace(0.31, 2.0, 100)
])

BASE_DIR = "Simulation_Results"

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def get_gasw_params_analytic(D_target, sigma=1.70, R_const=-24.5):
    K = np.sqrt(2.0 * np.pi) * sigma
    denominator = 1.0 - (R_const / K)
    if abs(denominator) < 1e-10: return -3.59, 0.7 
    U = D_target / denominator
    A = U - D_target
    return A, U

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_csv(filename, energy, sigma_au, sigma_mb):
    header = "Energy_au,CrossSection_au,CrossSection_Mb"
    data = np.column_stack((energy, sigma_au, sigma_mb))
    np.savetxt(filename, data, delimiter=",", header=header, comments='', fmt='%.6e')

def save_run_log(filename, species, depth, run_data):
    with open(filename, 'w') as f:
        f.write(f"SIMULATION SUMMARY LOG\nSpecies: {species}\nCage Depth: {depth:.4f} a.u.\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'Mu':<8} | {'E_bound':<12} | {'NormErr':<10} | {'Thresh':<12} | {'Peak':<12}\n")
        f.write(f"{'-'*80}\n")
        for entry in run_data:
            mu = entry['mu']
            if entry['success']:
                f.write(f"{mu:<8.3f} | {entry['E_b']:<12.5f} | {entry['norm_err']:<10.1e} | {entry['thresh']:<12.3e} | {entry['peak']:<12.3e}\n")
            else:
                f.write(f"{mu:<8.3f} | {'FAILED':<40}\n")

# ==============================================================================
# 3. PLOTTING FUNCTIONS
# ==============================================================================

def plot_bound_diagnostic(r, u, V, E, label, path, r_max_plot=20):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    mask = r < r_max_plot
    
    ax1.plot(r[mask], V[mask], 'k--', alpha=0.6, label='V(r)')
    ax1.set_ylabel("Potential (a.u.)")
    ax1.set_xlabel("Radius (a.u.)")
    
    v_min = min(V[mask])
    ax1.set_ylim(v_min * 1.1, 0.5)
    
    ax2 = ax1.twinx()
    ax2.plot(r[mask], u[mask], 'b-', lw=2, label='u(r)')
    ax2.fill_between(r[mask], 0, u[mask], color='b', alpha=0.1)
    ax2.set_ylabel("Wavefunction")
    ax2.axhline(0, color='gray', lw=0.5)
    
    plt.title(f"Bound State: {label}\nE = {E:.5f} a.u.")
    fig.tight_layout()
    plt.savefig(path)
    plt.close('all') 

def plot_continuum_diagnostic(r, u_cont, V, E_pe, label, path, r_max_plot=40):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    mask = r < r_max_plot
    
    # Potential on continuum grid
    ax1.plot(r[mask], V[mask], 'k--', alpha=0.6, label='V(r)')
    ax1.set_ylabel("Potential (a.u.)")
    ax1.set_xlabel("Radius (a.u.)")
    ax1.set_ylim(min(V[mask])*1.1, E_pe * 3.0)
    
    ax2 = ax1.twinx()
    ax2.plot(r[mask], u_cont[mask], 'r-', lw=1.5, label=f'Cont (E={E_pe:.2f})')
    ax2.set_ylabel("Wavefunction u(r)")
    
    plt.title(f"Continuum Sample: {label}")
    fig.tight_layout()
    plt.savefig(path)
    plt.close('all')

def plot_sigma_comparison(energy, result_dict, species, depth, path):
    plt.figure(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(result_dict)))
    
    valid_data = False
    for (mu, data), color in zip(result_dict.items(), colors):
        if data is None: continue 
        plt.plot(energy, data['sigma_au'], lw=2, color=color, label=f'μ={mu}')
        valid_data = True
        
    plt.xlabel("Photoelectron Energy (a.u.)")
    plt.ylabel("Cross Section (a.u.)")
    plt.title(f"{species} @ Depth {depth} a.u. - Plasma Effect")
    if valid_data: plt.legend(title="Debye μ")
    plt.grid(alpha=0.3)
    plt.xlim(0, 1.5)
    plt.tight_layout()
    plt.savefig(path)
    plt.close('all')

# ==============================================================================
# 4. MAIN DRIVER
# ==============================================================================

def main():
    print("="*80)
    print("STARTING BATCH SIMULATION (With Progress Bars)")
    print("="*80)

    total_start = time.time()
    
    print("Checking Argon Bound State...")
    r, u, E, l, _ = solve_ground_u(VGASW_total_debye, species='Ar', A=0, U=0, mu=0)
    print(f"Argon Energy: {E:.4f} a.u.")
    
    if abs(E + 0.58) < 0.1:   
        print("✅ CORRECT: Found 3p Valence State.")
    elif E < -1.0:
        print("❌ ERROR: Found 2p Core State. Update bound.py!")
    # --- Loop Species ---
    # Using tqdm for loops
    species_pbar = tqdm(TARGET_SPECIES, desc="Species Loop", position=0)
    
    for species in species_pbar:
        
        # --- Loop Depths ---
        depth_pbar = tqdm(DEPTHS, desc=f"Depths ({species})", position=1, leave=False)
        
        for depth in depth_pbar:
            # Setup
            case_dir = os.path.join(BASE_DIR, species, f"Depth_{depth:.2f}")
            dir_csv = os.path.join(case_dir, "CSV_Data")
            dir_bound = os.path.join(case_dir, "Plots_BoundState")
            dir_cont = os.path.join(case_dir, "Plots_Continuum")
            for d in [dir_csv, dir_bound, dir_cont]: make_dirs(d)
            
            A_val, U_val = get_gasw_params_analytic(depth)
            batch_results = {}
            run_log_data = []
            
            # --- Loop Mu (Screening) ---
            # Inner loop progress bar
            mu_pbar = tqdm(MU_VALUES, desc="Scanning Plasma Density", position=2, leave=False)
            
            for mu in mu_pbar:
                label = f"{species}_D{depth}_mu{mu}"
                mu_pbar.set_postfix(mu=f"{mu:.2f}", status="Calc")
                
                try:
                    # A. Solve Bound State
                    r_b, u_b, E_b, l_init, norm_val = solve_ground_u(
                        VGASW_total_debye, species=species, 
                        R_max=100.0, N=5000, 
                        A=A_val, U=U_val, mu=mu
                    )
                    
                    if E_b > -1e-4:
                        tqdm.write(f"  [Info] {species} Unbound at mu={mu} (E={E_b:.4f})")
                        batch_results[mu] = None
                        run_log_data.append({'mu': mu, 'success': False})
                        continue
                    
                    # B. Plot Bound State
                    V_bound = VGASW_total_debye(r_b, A_val, U_val, mu, species=species)
                    plot_bound_diagnostic(r_b, u_b, V_bound, E_b, label, 
                                          os.path.join(dir_bound, f"Bound_{label}.png"))
                    
                    # C. Compute Cross Section
                    sigma, diag = compute_cross_section_spectrum(
                        E_GRID, r_b, u_b, E_b, l_init, species, 
                        A_val, U_val, mu, n_workers=None
                    )
                    
                    # D. Plot Continuum Sample
                    sample_idx = min(20, len(E_GRID)-1)
                    sample_E = E_GRID[sample_idx]
                    l_plot = l_init + 1
                    r_c, u_c, _ = compute_continuum_state(sample_E, l_plot, species, A_val, U_val, mu)
                    V_cont = VGASW_total_debye(r_c, A_val, U_val, mu, species=species)
                    
                    plot_continuum_diagnostic(r_c, u_c, V_cont, sample_E, label,
                                              os.path.join(dir_cont, f"Cont_{label}.png"))
                    
                    # E. Save Data
                    sigma_mb = sigma * 28.0028
                    save_csv(os.path.join(dir_csv, f"Data_{label}.csv"), E_GRID, sigma, sigma_mb)
                    
                    batch_results[mu] = {'sigma_au': sigma}
                    run_log_data.append({
                        'mu': mu, 'success': True, 'E_b': E_b, 
                        'norm_err': abs(1.0-norm_val), 'thresh': sigma[0], 'peak': np.max(sigma)
                    })
                    
                    # Update progress bar with physics result
                    mu_pbar.set_postfix(mu=f"{mu:.2f}", Eb=f"{E_b:.3f}")
                    
                except Exception as e:
                    tqdm.write(f"  [Error] {species} mu={mu}: {e}")
                    # Traceback usually prints to stderr, helpful to keep it minimal here
                    batch_results[mu] = None
                    run_log_data.append({'mu': mu, 'success': False})

            # Save Summary
            comp_path = os.path.join(case_dir, f"Comparison_Sigma_{species}_D{depth}.png")
            plot_sigma_comparison(E_GRID, batch_results, species, depth, comp_path)
            save_run_log(os.path.join(case_dir, "summary_log.txt"), species, depth, run_log_data)

    print("\n" + "="*80)
    print(f"COMPLETE. Saved to: {os.path.abspath(BASE_DIR)}")
    print("="*80)

if __name__ == "__main__":
    freeze_support()
    main()