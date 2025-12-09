import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import sys
import os
from multiprocessing import cpu_count
from time import time

# Try to import tqdm for progress bar, handle if missing
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Install it with 'pip install tqdm' for a progress bar.")
    # Dummy fallback
    def tqdm(iterable, **kwargs): return iterable

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = r"C:\Users\harry\literate-robot\references\dataset"
RESULTS_DIR = "results"

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from potential import VGASW_total_debye
    from bound import solve_ground_u
    from cross_section import compute_cross_section_spectrum
except ImportError:
    print("ERROR: Could not import physics modules from 'src/'.")
    sys.exit(1)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def load_marr_west_data(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        filepath = filename 
        if not os.path.exists(filepath):
            return None

    try:
        df = pd.read_csv(filepath)
        df.columns = [c.strip().lower() for c in df.columns]
        rename_map = {'wavelength(a)': 'lambda_A', 'cross_section(mb)': 'sigma_Mb'}
        for col in df.columns:
            if 'ev' in col or 'energy' in col:
                rename_map[col] = 'energy_eV'
                break
        df = df.rename(columns=rename_map)
        df['lambda_A'] = pd.to_numeric(df['lambda_A'], errors='coerce')
        df['sigma_Mb'] = pd.to_numeric(df['sigma_Mb'], errors='coerce')
        df = df.dropna(subset=['lambda_A', 'sigma_Mb'])
        df = df.sort_values('lambda_A', ascending=False)
        return df[['lambda_A', 'sigma_Mb']].values
    except Exception as e:
        return None

def run_fast_verification():
    # 1. Setup Parallelization
    num_cores = max(1, cpu_count() - 1) # Leave 1 core free for OS
    print(f"\n>>> Starting High-Performance Verification")
    print(f"    Parallel Engine: Enabled ({num_cores} workers)")
    print(f"    Data Source: {DATA_DIR}\n")
    
    Ha_to_eV = 27.211386
    
    studies = [
        ('He', 2.0, 'helium_data_marr_and_west.csv', "Helium (1s²)"),
        ('Ne', 6.0, 'neon_data_marr_and_west.csv',   "Neon (2p⁶)"),
        ('Ar', 6.0, 'argon_data_marr_and_west.csv',  "Argon (3p⁶)")
    ]
    
    # 2. Main Loop with Progress Bar
    # We use a loop wrapper to show progress
    pbar = tqdm(studies, desc="Overall Progress", unit="species")
    
    for species, occ, csv_file, title in pbar:
        # Update description to show what we are working on
        pbar.set_description(f"Processing {title}")
        
        start_time = time()
        
        # --- A. Load Data ---
        exp_data = load_marr_west_data(csv_file)
        if exp_data is None:
            tqdm.write(f"  [!] Skipped {title}: Data file not found.")
            continue
        exp_lam, exp_mb = exp_data[:, 0], exp_data[:, 1]
        
        # --- B. Physics Simulation ---
        # 1. Bound State
        r_b, u_b, E_b, l_init, _ = solve_ground_u(
            VGASW_total_debye, species=species, R_max=60.0, N=4000, 
            A=0.0, U=0.0, mu=0.0
        )
        Ip_eV = abs(E_b) * Ha_to_eV
        
        # 2. Continuum Spectrum (Parallelized)
        lam_min, lam_max = np.min(exp_lam), np.max(exp_lam)
        e_min_au = max(0.01, (12398.4/(lam_max*1.1) - Ip_eV)/Ha_to_eV)
        e_max_au = (12398.4/(lam_min*0.9) - Ip_eV)/Ha_to_eV
        
        # Use high resolution (1000 pts) because parallelization makes it cheap
        e_kin_au = np.linspace(e_min_au, e_max_au, 1000)
        
        # CALL WITH PARALLEL WORKERS
        sigma_au, _ = compute_cross_section_spectrum(
            e_kin_au, r_b, u_b, E_b, l_init, species, 0.0, 0.0, 0.0, 
            n_workers=num_cores 
        )
        
        # Convert Units
        photon_ev_sim = (e_kin_au * Ha_to_eV) + Ip_eV
        lambda_sim_A = 12398.4 / photon_ev_sim
        sigma_sim_mb = sigma_au * 28.0028 * occ
        
        # --- C. Analysis ---
        sim_val_at_exp = np.interp(exp_lam, lambda_sim_A[::-1], sigma_sim_mb[::-1])
        diff = sim_val_at_exp - exp_mb
        pct_error = 100.0 * diff / exp_mb
        
        # Save CSV
        df_out = pd.DataFrame({
            'Lambda (A)': exp_lam,
            'Exp (Mb)': exp_mb,
            'Sim (Mb)': np.round(sim_val_at_exp, 3),
            'Error (%)': np.round(pct_error, 2)
        })
        df_out.to_csv(os.path.join(RESULTS_DIR, f"Deviation_MarrWest_{species}.csv"), index=False)

        # --- D. Plotting ---
        # (Standard Plot Code)
        fig = plt.figure(figsize=(10, 8), dpi=100)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
        
        ax0 = plt.subplot(gs[0])
        ax0.plot(lambda_sim_A, sigma_sim_mb, 'b-', lw=2, label='Simulation (SAE)')
        ax0.scatter(exp_lam, exp_mb, c='r', s=25, label='Exp (Marr & West)', zorder=5)
        ax0.set_ylabel("Cross Section (Mb)"); ax0.legend(); ax0.grid(alpha=0.3)
        ax0.set_title(f"{title}: SAE vs Experiment")
        ax0.set_xticklabels([])
        ax0.set_xlim(np.max(lambda_sim_A), np.min(lambda_sim_A)) # High Lambda Left
        
        ax1 = plt.subplot(gs[1])
        ax1.axhline(0, c='k', ls='--')
        ax1.plot(exp_lam, pct_error, 'r-', alpha=0.6)
        ax1.scatter(exp_lam, pct_error, c='r', s=10)
        ax1.set_ylabel("Error (%)"); ax1.set_xlabel(r"Wavelength ($\AA$)"); ax1.grid(alpha=0.3)
        ax1.set_xlim(ax0.get_xlim())
        err_lim = min(100, np.percentile(np.abs(pct_error), 95) * 1.5)
        ax1.set_ylim(-err_lim, err_lim)
        
        plt.savefig(os.path.join(RESULTS_DIR, f"Verify_MarrWest_{species}.png"))
        plt.close()

        # Argon Zoom Plot
        if species == 'Ar':
            fig_z = plt.figure(figsize=(8, 6), dpi=100)
            plt.plot(lambda_sim_A, sigma_sim_mb, 'b-', lw=3, label='Simulation')
            plt.scatter(exp_lam, exp_mb, c='r', s=40, label='Marr & West', zorder=5)
            
            # Find minimum in relevant range
            mask = (lambda_sim_A > 100) & (lambda_sim_A < 600)
            if np.any(mask):
                min_idx = np.argmin(sigma_sim_mb[mask])
                real_idx = np.where(mask)[0][min_idx]
                plt.annotate(f'Min: {sigma_sim_mb[real_idx]:.2f} Mb', 
                             xy=(lambda_sim_A[real_idx], sigma_sim_mb[real_idx]),
                             xytext=(lambda_sim_A[real_idx]+100, sigma_sim_mb[real_idx]+1),
                             arrowprops=dict(facecolor='black', shrink=0.05))
            
            plt.title("Argon Cooper Minimum (Zoom)"); plt.xlabel(r"$\lambda$ ($\AA$)"); plt.ylabel("$\sigma$ (Mb)")
            plt.xlim(650, 150); plt.ylim(0, 5.0); plt.grid(alpha=0.3); plt.legend()
            plt.savefig(os.path.join(RESULTS_DIR, "Verify_MarrWest_Ar_Zoom.png"))
            plt.close()

        # Print summary for this step (using tqdm.write to avoid breaking progress bar)
        elapsed = time() - start_time
        avg_err = np.mean(np.abs(pct_error))
        tqdm.write(f"  > Finished {species} in {elapsed:.2f}s. Avg Error: {avg_err:.1f}%")

    print("\n>>> All verification tasks complete.")

if __name__ == "__main__":
    run_fast_verification()