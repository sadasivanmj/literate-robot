print(">>> Initializing High-Performance Verification Script...") 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import sys
import os
from time import time
from multiprocessing import cpu_count

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    print("    [!] 'tqdm' not installed. Progress bars will be disabled.")
    # Dummy fallback
    def tqdm(iterable, **kwargs): return iterable

# ==============================================================================
# 1. CONFIGURATION & SETUP
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'references', 'dataset')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

print(f"    Physics Src:  {SRC_DIR}")
print(f"    Results Dir:  {RESULTS_DIR}")

# Import Physics Modules
sys.path.append(SRC_DIR)
try:
    from potential import VGASW_total_debye
    from bound import solve_ground_u
    from cross_section import compute_cross_section_spectrum
    print(">>> Physics Engine Loaded.")
except ImportError as e:
    print(f"\nFATAL ERROR: Could not import physics modules: {e}")
    sys.exit(1)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Define Studies: (Element, Occupation, CSV_Filename/Mode, Title)
STUDIES = [
    ('H',  1.0, "ANALYTIC", "Hydrogen (1s)"),
    ('He', 2.0, "helium_data_marr_and_west.csv", "Helium (1s²)"),
    ('Ne', 6.0, "neon_data_marr_and_west.csv",   "Neon (2p⁶)"),
    ('Ar', 6.0, "argon_data_marr_and_west.csv",  "Argon (3p⁶)")
]

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def get_analytic_hydrogen(lambda_angstroms):
    """
    Returns analytic cross section (Mb) for Hydrogen 1s.
    Sigma = 6.304 Mb * (13.606 / E)^3
    """
    energies_eV = 12398.4 / lambda_angstroms
    Ry = 13.6057
    
    # Formula valid above threshold
    mask = energies_eV >= Ry
    sigma = np.zeros_like(energies_eV)
    
    # Simple Kramers/Bethe approximation for checking
    sigma[mask] = 6.304 * (Ry / energies_eV[mask])**3
    return sigma

def load_experimental_data(filename):
    """Loads raw experimental data or generates analytic data."""
    
    # CASE A: Analytic Hydrogen
    if filename == "ANALYTIC":
        # Generate synthetic 'experiment' points for plotting
        lam = np.linspace(911, 100, 50) # From threshold down to 100 A
        sig = get_analytic_hydrogen(lam)
        return lam, sig

    # CASE B: Load from CSV
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        # Try local folder
        if os.path.exists(filename): 
            filepath = filename
        else:
            print(f"    [!] Error: File not  found: {filepath}")
            return None, None

    try:
        df = pd.read_csv(filepath)
        df.columns = [c.strip().lower() for c in df.columns]
        
        col_lam = next((c for c in df.columns if 'wavelength' in c or 'lambda' in c), None)
        col_sig = next((c for c in df.columns if 'cross_section' in c or 'sigma' in c), None)
        
        if not col_lam or not col_sig:
            return None, None
            
        df['lambda'] = pd.to_numeric(df[col_lam], errors='coerce')
        df['sigma'] = pd.to_numeric(df[col_sig], errors='coerce')
        df = df.dropna(subset=['lambda', 'sigma'])
        
        # Sort by wavelength descending (Low Energy -> High Energy)
        df = df.sort_values('lambda', ascending=False)
        return df['lambda'].values, df['sigma'].values
        
    except Exception as e:
        print(f"    [!] CSV Error: {e}")
        return None, None

# ==============================================================================
# 3. MAIN ANALYSIS FUNCTION
# ==============================================================================
def process_element(species, occ, csv_file, title, pbar=None):
    if pbar: pbar.set_description(f"Simulating {species}")
    else: print(f"\n--- Processing {title} ---")
    
    start_time = time()
    Ha_to_eV = 27.211386

    # A. Load Experiment / Analytic Data
    exp_lam, exp_mb = load_experimental_data(csv_file)
    if exp_lam is None:
        return

    # B. Run Simulation
    # 1. Bound State
    r_b, u_b, E_b, l_init, _ = solve_ground_u(
        VGASW_total_debye, species=species, R_max=60.0, N=4000
    )
    Ip_eV = abs(E_b) * Ha_to_eV
    
    # 2. Continuum Spectrum (Parallelized)
    # Create grid covering experimental range + padding
    lam_min, lam_max = np.min(exp_lam), np.max(exp_lam)
    
    # Safe energy conversion (avoid div by zero if lam is super small)
    e_max_au = (12398.4 / max(lam_min * 0.8, 1.0) - Ip_eV) / Ha_to_eV
    e_min_au = 0.01
    
    # Use 1000 points for smooth curve
    e_kin_au = np.linspace(e_min_au, max(e_max_au, 5.0), 1000)
    
    # Parallel Config
    num_cores = max(1, cpu_count() - 1)
    
    # Compute Cross Section
    sigma_au, _ = compute_cross_section_spectrum(
        e_kin_au, r_b, u_b, E_b, l_init, species, 0.0, 0.0, 0.0, 
        n_workers=num_cores
    )
    
    # Convert Units
    photon_ev_sim = (e_kin_au * Ha_to_eV) + Ip_eV
    lambda_sim_A = 12398.4 / photon_ev_sim
    sigma_sim_mb = sigma_au * 28.0028 * occ

    # C. Save Simulation Data
    sim_csv_path = os.path.join(RESULTS_DIR, f"Simulation_{species}.csv")
    pd.DataFrame({
        'Energy (eV)': np.round(photon_ev_sim, 3),
        'Wavelength (A)': np.round(lambda_sim_A, 2),
        'Cross Section (Mb)': np.round(sigma_sim_mb, 4)
    }).to_csv(sim_csv_path, index=False)

    # D. Calculate Deviations
    # Interpolate (Flip arrays because interp requires increasing X)
    sim_at_exp = np.interp(exp_lam, lambda_sim_A[::-1], sigma_sim_mb[::-1])
    diff = sim_at_exp - exp_mb
    
    # Handle Analytic H where exp might be 0 below threshold
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_error = 100.0 * diff / exp_mb
        pct_error[np.isinf(pct_error)] = 0.0
        pct_error[np.isnan(pct_error)] = 0.0

    # Save Deviation Report
    dev_csv_path = os.path.join(RESULTS_DIR, f"Deviation_Report_{species}.csv")
    pd.DataFrame({
        'Wavelength (A)': exp_lam,
        'Exp (Mb)': np.round(exp_mb, 3),
        'Sim (Mb)': np.round(sim_at_exp, 3),
        'Error (%)': np.round(pct_error, 1)
    }).to_csv(dev_csv_path, index=False)

    # E. Plotting
    fig = plt.figure(figsize=(10, 8), dpi=100)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)

    # Top: Comparison
    ax0 = plt.subplot(gs[0])
    ax0.plot(lambda_sim_A, sigma_sim_mb, 'b-', linewidth=2, label='SAE Simulation')
    label_exp = 'Analytic Theory' if csv_file == 'ANALYTIC' else 'Experiment (Marr & West)'
    ax0.scatter(exp_lam, exp_mb, color='red', s=30, label=label_exp, zorder=5)
    
    ax0.set_ylabel("Cross Section (Mb)")
    ax0.set_title(f"{title}: Simulation vs {label_exp}")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax0.set_xticklabels([])
    
    # Invert X-Axis (High Wavelength on Left for Cooper style, or Right for Standard)
    # Let's stick to High Lambda (Low Energy) on RIGHT to match standard X-Axis increasing
    # BUT spectroscopic plots often put Energy on X. 
    # Let's put Wavelength decreasing to the right (Energy increasing).
    ax0.set_xlim(np.max(lambda_sim_A), np.min(lambda_sim_A)) 

    # Bottom: Deviation
    ax1 = plt.subplot(gs[1])
    ax1.axhline(0, color='k', linestyle='--')
    ax1.plot(exp_lam, pct_error, 'r-o', markersize=4, linewidth=1)
    ax1.set_ylabel("Error (%)")
    ax1.set_xlabel(r"Wavelength ($\AA$)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(ax0.get_xlim())
    
    err_limit = min(200, np.percentile(np.abs(pct_error), 95) * 1.5) if len(pct_error) > 0 else 50
    ax1.set_ylim(-err_limit, err_limit)

    plot_path = os.path.join(RESULTS_DIR, f"Analysis_{species}.png")
    plt.savefig(plot_path)
    plt.close()
    
    if pbar: 
        pbar.write(f"    > Finished {species} (Ip={Ip_eV:.2f} eV). Saved {os.path.basename(plot_path)}")

# ==============================================================================
# 4. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print(f"    > Parallel Engine: Enabled ({max(1, cpu_count()-1)} cores)")
    
    # Progress Bar Loop
    with tqdm(STUDIES, unit="atom") as pbar:
        for sp, occ, csv, ti in pbar:
            process_element(sp, occ, csv, ti, pbar)
            
    print("\n>>> All verification tasks complete. Check 'results/' folder.")