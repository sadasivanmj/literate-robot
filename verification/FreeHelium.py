import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.interpolate import PchipInterpolator

# Import your physics stack
from potential import VGASW_total_debye, solve_gasw_parameters
from bound import solve_ground_u
from cross_section import compute_cross_section_spectrum

# ==============================================================================
# 1. COMPLETE EXPERIMENTAL DATASET (Samson & Stolte 2002, Table 1)
# ==============================================================================
# Extracted directly from the paper for maximum density
# Columns: Photon Energy (eV), Cross Section (Mb)
ss_data = np.array([
    [24.60, 7.56], [24.80, 7.46], [25.05, 7.33], [25.30, 7.19], [25.56, 7.06],
    [25.83, 6.94], [26.10, 6.81], [26.38, 6.68], [26.66, 6.55], [26.95, 6.43],
    [27.25, 6.30], [27.86, 6.05], [28.50, 5.81], [29.17, 5.57], [29.87, 5.33],
    [30.61, 5.10], [31.39, 4.87], [32.20, 4.64], [33.06, 4.42], [33.97, 4.20],
    [34.92, 4.00], [35.94, 3.78], [37.01, 3.57], [38.15, 3.37], [39.36, 3.18],
    [40.65, 2.98], [42.03, 2.80], [44.28, 2.52], [47.68, 2.18], [50.60, 1.94],
    [52.76, 1.78], [56.35, 1.55], [60.48, 1.34], [65.25, 1.14], [70.85, 0.961],
    [77.49, 0.792], [85.50, 0.637], [95.37, 0.497], [103.3, 0.412], [112.7, 0.335],
    [124.0, 0.265], [137.8, 0.202], [155.0, 0.147], [177.1, 0.100]
])

exp_ev = ss_data[:, 0]
exp_mb = ss_data[:, 1]

def run_advanced_analysis():
    print(">>> 1. Configuring High-Precision Helium Simulation...")
    
    # Physics Constants
    Ha_to_eV = 27.211386
    OCCUPATION = 2.0  # Helium 1s^2
    
    # A. Solve Bound State
    print("    Solving 1s state (R_max=60.0, N=4000)...")
    r_b, u_b, E_b, l_init, _ = solve_ground_u(
        VGASW_total_debye, species='He', R_max=60.0, N=4000, 
        A=0.0, U=0.0, mu=0.0
    )
    Ip_eV = abs(E_b) * Ha_to_eV
    print(f"    Ionization Potential: {Ip_eV:.4f} eV")

    # B. Simulation Grid (Dense)
    # 200 points from threshold up to 180 eV to match full dataset
    e_kin_max = (185.0 - Ip_eV) / Ha_to_eV
    e_kin_au = np.linspace(0.01, e_kin_max, 200)
    
    # C. Compute Spectrum
    print("    Computing Cross Sections...")
    sigma_au, _ = compute_cross_section_spectrum(
        e_kin_au, r_b, u_b, E_b, l_init, 
        'He', 0.0, 0.0, 0.0, n_workers=1
    )
    
    # D. Units Conversion
    sim_ev = (e_kin_au * Ha_to_eV) + Ip_eV
    sim_mb = sigma_au * 28.0028 * OCCUPATION

    # ==========================================================================
    # 2. STATISTICAL COMPARISON (Interpolation)
    # ==========================================================================
    # We interpolate the Simulation onto the Experimental grid for exact error calc
    # PchipInterpolator preserves monotonicity (no wiggles)
    interpolator = PchipInterpolator(sim_ev, sim_mb)
    sim_at_exp_points = interpolator(exp_ev)
    
    # Calculate Residuals
    abs_diff = sim_at_exp_points - exp_mb
    pct_error = 100.0 * (abs_diff / exp_mb)
    
    # Save Data Table
    df = pd.DataFrame({
        'Energy (eV)': exp_ev,
        'Exp (Mb)': exp_mb,
        'Sim (Mb)': np.round(sim_at_exp_points, 3),
        'Diff (Mb)': np.round(abs_diff, 4),
        'Error (%)': np.round(pct_error, 2)
    })
    df.to_csv("Helium_Scientific_Comparison.csv", index=False)
    print(f"    Avg Absolute Error: {np.mean(np.abs(pct_error)):.2f}%")

    # ==========================================================================
    # 3. SCIENTIFIC PLOTTING (Dual Panel)
    # ==========================================================================
    print(">>> 3. Generating Analysis Plot...")
    
    fig = plt.figure(figsize=(10, 12), dpi=150)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)
    
    # --- PANEL 1: Linear Scale (Threshold Physics) ---
    ax1 = plt.subplot(gs[0])
    ax1.scatter(exp_ev, exp_mb, color='red', s=30, label='Experiment (Samson 2002)', zorder=5)
    ax1.plot(sim_ev, sim_mb, 'b-', linewidth=2, label='Simulation (SAE, Tong-Lin)')
    
    ax1.set_title(r'Helium Photoionization: Threshold Region (Linear)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cross Section (Mb)', fontsize=12)
    ax1.set_xlim(20, 60)
    ax1.set_ylim(0, 9)
    ax1.legend(fontsize=12)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # --- PANEL 2: Log-Log Scale (Tail Physics) ---
    ax2 = plt.subplot(gs[1])
    ax2.loglog(exp_ev, exp_mb, 'ro', markersize=5, label='Experiment')
    ax2.loglog(sim_ev, sim_mb, 'b-', linewidth=2, label='Simulation')
    
    ax2.set_title(r'High-Energy Tail (Log-Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cross Section (Mb)', fontsize=12)
    ax2.set_xlim(20, 200)
    ax2.set_ylim(0.08, 10)
    ax2.grid(True, which='major', linestyle='-', alpha=0.6)
    ax2.grid(True, which='minor', linestyle=':', alpha=0.3)
    
    # Add Power Law Reference line (E^-3.5)
    # Determine local slope at 100 eV for annotation
    idx_100 = np.argmin(np.abs(sim_ev - 100))
    slope = (np.log(sim_mb[idx_100+1]) - np.log(sim_mb[idx_100])) / \
            (np.log(sim_ev[idx_100+1]) - np.log(sim_ev[idx_100]))
    ax2.text(80, 0.5, f"Tail Slope $\\alpha \\approx {slope:.2f}$", fontsize=10, color='blue')

    # --- PANEL 3: Residuals (Error Analysis) ---
    ax3 = plt.subplot(gs[2])
    ax3.plot(exp_ev, pct_error, 'k.-', linewidth=1)
    ax3.fill_between(exp_ev, pct_error, 0, where=(pct_error>0), color='blue', alpha=0.2)
    ax3.fill_between(exp_ev, pct_error, 0, where=(pct_error<0), color='red', alpha=0.2)
    
    ax3.axhline(0, color='black', linewidth=1)
    ax3.set_title('Relative Deviation (Simulation - Exp) / Exp', fontsize=12)
    ax3.set_xlabel('Photon Energy (eV)', fontsize=12)
    ax3.set_ylabel('Error (%)', fontsize=10)
    ax3.set_xlim(20, 180)
    ax3.set_ylim(-80, 20) # SAE tail usually underestimates significantly
    ax3.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig('Helium_Scientific_Analysis.png')
    print(">>> Plot saved to 'Helium_Scientific_Analysis.png'")
    print(">>> Data saved to 'Helium_Scientific_Comparison.csv'")

if __name__ == "__main__":
    run_advanced_analysis()