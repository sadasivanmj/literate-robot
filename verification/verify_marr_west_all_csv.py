"""
High-Performance Verification Script (Final Version).
Features:
- ADAPTIVE GRID: Concentrates points near Threshold and Cooper Minimum.
- ROBUST DETECTION: Correctly finds local Cooper minimum (avoids high-E tail).
- Dynamic Plotting Limits & Comparison to Experiments.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import sys
import os
from multiprocessing import cpu_count

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'references', 'dataset')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

sys.path.append(SRC_DIR)
try:
    from potential import VGASW_total_debye, SAE_PARAMS
    from bound import solve_ground_u
    from cross_section import compute_cross_section_spectrum
except ImportError:
    sys.exit("Critical Error: Could not import physics modules.")

os.makedirs(RESULTS_DIR, exist_ok=True)

STUDIES = [
    ('H',  1.0, "ANALYTIC", "Hydrogen (1s)"),
    ('He', 2.0, "helium_data_marr_and_west.csv", "Helium (1s²)"),
    ('Ne', 6.0, "neon_data_marr_and_west.csv",   "Neon (2p⁶)"),
    ('Ar', 6.0, "argon_data_marr_and_west.csv",  "Argon (3p⁶)")
]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_analytic_hydrogen(lambda_angstroms):
    energies_eV = 12398.4 / lambda_angstroms
    Ry = 13.6057
    mask = energies_eV >= Ry
    sigma = np.zeros_like(energies_eV)
    sigma[mask] = 6.304 * (Ry / energies_eV[mask])**3
    return sigma

def load_experimental_data(filename):
    if filename == "ANALYTIC":
        lam = np.linspace(911, 100, 200)
        sig = get_analytic_hydrogen(lam)
        return lam, sig

    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        if os.path.exists(filename): filepath = filename
        else: return None, None

    try:
        df = pd.read_csv(filepath)
        df.columns = [c.strip().lower() for c in df.columns]
        col_lam = next((c for c in df.columns if 'lambda' in c or 'wave' in c), None)
        col_sig = next((c for c in df.columns if 'sigma' in c or 'cross' in c), None)
        
        if not col_lam or not col_sig: return None, None
        
        df['lambda'] = pd.to_numeric(df[col_lam], errors='coerce')
        df['sigma'] = pd.to_numeric(df[col_sig], errors='coerce')
        df = df.dropna().sort_values('lambda', ascending=False)
        
        return df['lambda'].values, df['sigma'].values
    except:
        return None, None

# ==============================================================================
# ANALYSIS ENGINE
# ==============================================================================
def process_element(species, occ, csv_file, title, pbar=None):
    if pbar: pbar.set_description(f"Simulating {species}")
    
    # 1. Load Experiment
    exp_lam, exp_mb = load_experimental_data(csv_file)
    if exp_lam is None: return

    # 2. Physics Parameters
    if species in SAE_PARAMS:
        params = SAE_PARAMS[species].get('default', list(SAE_PARAMS[species].values())[0])
        l_init = params.ground_l
    else:
        l_init = 0

    # 3. Bound State
    r_b, u_b, E_b, _, _ = solve_ground_u(
        VGASW_total_debye, species=species, R_max=80.0, N=6000, A=0.0, U=0.0
    )
    Ip_eV = abs(E_b) * 27.211386

    # 4. ADAPTIVE CONTINUUM GRID
    # ---------------------------------------------------------
    lam_min, lam_max = np.min(exp_lam), np.max(exp_lam)
    
    sim_lam_min = max(1.0, lam_min * 0.9)
    sim_lam_max = lam_max * 1.1
    
    # Convert Experimental Range to Energy (a.u.)
    e_max_au = (12398.4 / sim_lam_min - Ip_eV) / 27.211386
    e_min_au = max(0.01, (12398.4 / sim_lam_max - Ip_eV) / 27.211386)
    
    if e_max_au <= e_min_au: e_max_au = e_min_au + 5.0

    # --- DYNAMIC GRID GENERATION ---
    grid_segments = []
    # Base Grid (Coarse background)
    grid_segments.append(np.linspace(e_min_au, e_max_au, 100))
    
    # Threshold Refinement (Universal)
    threshold_cap = min(e_max_au, 0.5) 
    if threshold_cap > e_min_au:
        grid_segments.append(np.linspace(e_min_au, threshold_cap, 150))
        
    # Cooper Minimum Refinement (Argon Specific)
    if species == 'Ar':
        cm_start = 0.7
        cm_end = 1.3
        if cm_start < e_max_au and cm_end > e_min_au:
            cm_start = max(cm_start, e_min_au)
            cm_end = min(cm_end, e_max_au)
            grid_segments.append(np.linspace(cm_start, cm_end, 150))
            
    e_kin_au = np.unique(np.concatenate(grid_segments))
    e_kin_au = np.sort(e_kin_au)
    # ---------------------------------------------------------
    
    # 5. Compute Sigma
    num_cores = max(1, cpu_count() - 1)
    sigma_au, _ = compute_cross_section_spectrum(
        e_kin_au, r_b, u_b, E_b, l_init, species, 
        A=0.0, U=0.0, mu=0.0, 
        n_workers=num_cores, verbose=False
    )
    
    photon_ev_sim = (e_kin_au * 27.211386) + Ip_eV
    lambda_sim_A = 12398.4 / photon_ev_sim
    sigma_sim_mb = sigma_au * 28.0028 * occ

    # 6. Plotting
    fig = plt.figure(figsize=(10, 8), dpi=120)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)

    # --- Main Plot ---
    ax0 = plt.subplot(gs[0])
    ax0.plot(lambda_sim_A, sigma_sim_mb, 'b-', lw=2, label=f'SAE Simulation (IP={Ip_eV:.1f} eV)')
    ax0.plot(exp_lam, exp_mb, 'r.', ms=5, alpha=0.6, label='Experiment (Marr & West)')
    
    # Force Limits to Experimental Data (Low Energy Right)
    ax0.set_xlim(lam_max, lam_min) 
    
    ax0.set_ylabel("Cross Section (Mb)", fontweight='bold')
    ax0.set_title(f"{title}: Comparison", fontweight='bold')
    ax0.legend(loc='upper right')
    ax0.grid(True, alpha=0.3)
    ax0.set_xticklabels([])

    # --- [CORRECTED] Argon Cooper Minimum Zoom ---
    if species == 'Ar':
        # 1. SEARCH WINDOW: Look for min only between 0.1 and 5.0 a.u. kinetic energy
        # This ignores the high-energy tail where cross-section naturally drops to ~0
        search_mask = (e_kin_au > 0.1) & (e_kin_au < 5.0)
        
        if np.any(search_mask):
            win_energies = photon_ev_sim[search_mask]
            win_sigma = sigma_sim_mb[search_mask]
            win_au = e_kin_au[search_mask]
            
            local_min_idx = np.argmin(win_sigma)
            min_sigma = win_sigma[local_min_idx]
            min_energy = win_energies[local_min_idx]
            min_au = win_au[local_min_idx]
        else:
            # Fallback (unlikely)
            min_idx = np.argmin(sigma_sim_mb)
            min_sigma = sigma_sim_mb[min_idx]
            min_energy = photon_ev_sim[min_idx]
            min_au = (min_energy - Ip_eV) / 27.211386
        
        lit_au = 0.92
        lit_ev = (lit_au * 27.211386) + 15.76 
        
        # Inset Placement: Top-Right (unobstructed)
        ax_ins = inset_axes(ax0, width="40%", height="40%", 
                            loc='upper right', 
                            bbox_to_anchor=(0.0, -0.1, 1.0, 1.0), 
                            bbox_transform=ax0.transAxes,
                            borderpad=2)
        
        ax_ins.plot(photon_ev_sim, sigma_sim_mb, 'b-', lw=2)
        ax_ins.plot(12398.4/exp_lam, exp_mb, 'r.', ms=5, alpha=0.5)
        
        ax_ins.axvline(min_energy, color='b', ls='--', alpha=0.5)
        ax_ins.text(min_energy, min_sigma*1.5, f"Sim: {min_au:.2f} a.u.", 
                    color='b', fontsize=8, rotation=90, verticalalignment='bottom')
        
        ax_ins.axvline(lit_ev, color='green', ls=':', alpha=0.8)
        ax_ins.text(lit_ev, min_sigma*1.5, f"Lit: {lit_au:.2f} a.u.", 
                    color='green', fontsize=8, rotation=90, verticalalignment='bottom')

        ax_ins.set_xlim(20, 80)
        ax_ins.set_ylim(0, 10)
        ax_ins.set_xlabel("Photon Energy (eV)", fontsize=8)
        ax_ins.set_ylabel("Sigma (Mb)", fontsize=8)
        ax_ins.set_title("Cooper Minimum Zoom", fontsize=9, fontweight='bold')
        ax_ins.grid(True, alpha=0.2)
        ax_ins.tick_params(labelsize=8)
        
        print(f"\n[Argon Analysis]")
        print(f"  Cooper Min (Sim): {min_au:.3f} a.u. ({min_energy:.1f} eV)")
        print(f"  Cooper Min (Lit): {lit_au:.3f} a.u. ({lit_ev:.1f} eV)")
        print(f"  Deviation:        {abs(min_au - lit_au):.3f} a.u.")

    # --- Deviation Plot ---
    ax1 = plt.subplot(gs[1])
    sim_interp = np.interp(exp_lam, lambda_sim_A[::-1], sigma_sim_mb[::-1])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        diff = 100.0 * (sim_interp - exp_mb) / exp_mb
    
    ax1.axhline(0, color='k', ls='--', alpha=0.5)
    ax1.plot(exp_lam, diff, 'k-', lw=1)
    ax1.fill_between(exp_lam, diff, 0, where=(diff>0), color='red', alpha=0.3)
    ax1.fill_between(exp_lam, diff, 0, where=(diff<0), color='blue', alpha=0.3)
    
    ax1.set_xlim(lam_max, lam_min)
    ax1.set_ylim(-100, 100)
    ax1.set_ylabel("Error (%)", fontweight='bold')
    ax1.set_xlabel(r"Wavelength ($\AA$)", fontweight='bold')
    ax1.grid(True, alpha=0.3)

    plt.savefig(os.path.join(RESULTS_DIR, f"Verify_{species}_Final.png"))
    plt.close()

if __name__ == "__main__":
    print(f">>> Processing {len(STUDIES)} studies with Adaptive Grids & Corrected Analysis...")
    iterator = tqdm(STUDIES, unit="atom")
    for sp, occ, csv, ti in iterator:
        process_element(sp, occ, csv, ti, iterator if isinstance(iterator, tqdm) else None)
    print("\n>>> Done.")