import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import sys
import os

# Ensure we can import from src
# Adjust this path if your folder structure is different
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from potential import VGASW_total_debye
from bound import solve_ground_u
from cross_section import compute_cross_section_spectrum

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# ==============================================================================
# 1. COOPER 1962 DATA POINTS
# ==============================================================================
cooper_he_lam = np.array([504, 450, 400, 350, 300, 250, 200])
cooper_he_mb  = np.array([7.6, 6.2, 5.0, 4.0, 3.1, 2.2, 1.5])

cooper_ne_lam = np.array([575, 500, 400, 300, 200, 100])
cooper_ne_mb  = np.array([12.0, 10.0, 11.5, 11.0, 8.0, 2.0]) 

cooper_ar_lam = np.array([780, 700, 600, 500, 400, 300, 200])
cooper_ar_mb  = np.array([38.0, 35.0, 25.0, 12.0, 2.0, 0.5, 1.0]) 

def run_expert_verification():
    print(">>> Starting Expert Data Analysis of Cooper (1962)...")
    Ha_to_eV = 27.211386
    
    configs = [
        ('He', 2.0, cooper_he_lam, cooper_he_mb, "Helium"),
        ('Ne', 6.0, cooper_ne_lam, cooper_ne_mb, "Neon"),
        ('Ar', 6.0, cooper_ar_lam, cooper_ar_mb, "Argon")
    ]
    
    for species, occ, exp_lam, exp_mb, name in configs:
        print(f"\n--- Processing {name} ({species}) ---")
        
        # 1. Solve Bound State
        r_b, u_b, E_b, l_init, _ = solve_ground_u(
            VGASW_total_debye, species=species, R_max=60.0, N=4000, 
            A=0.0, U=0.0, mu=0.0
        )
        Ip_eV = abs(E_b) * Ha_to_eV
        print(f"    Ionization Potential: {Ip_eV:.2f} eV")
        
        # 2. Compute Spectrum (High Resolution for smooth interpolation)
        e_kin_au = np.linspace(0.01, 6.0, 1000)
        sigma_au, _ = compute_cross_section_spectrum(
            e_kin_au, r_b, u_b, E_b, l_init, species, 0.0, 0.0, 0.0, n_workers=1
        )
        
        # Convert to Physics Units
        photon_ev_sim = (e_kin_au * Ha_to_eV) + Ip_eV
        lambda_sim_A = 12398.4 / photon_ev_sim
        sigma_sim_mb = sigma_au * 28.0028 * occ
        
        # 3. Deviation Analysis
        # Note: np.interp expects x to be increasing. lambda decreases as energy increases.
        # We flip both arrays to interpolate correctly.
        sim_val_at_exp = np.interp(exp_lam, lambda_sim_A[::-1], sigma_sim_mb[::-1])
        
        diff = sim_val_at_exp - exp_mb
        pct_error = 100.0 * diff / exp_mb
        
        # Dataframe
        df = pd.DataFrame({
            'Lambda (A)': exp_lam,
            'Cooper (Mb)': exp_mb,
            'Sim (Mb)': np.round(sim_val_at_exp, 2),
            'Diff (Mb)': np.round(diff, 2),
            'Error (%)': np.round(pct_error, 1)
        })
        print(df.to_string(index=False))
        df.to_csv(f"results/Deviation_{species}.csv", index=False)

        # ======================================================================
        # 4. PLOT 1: STANDARD ANALYSIS (Dual Panel)
        # ======================================================================
        fig = plt.figure(figsize=(10, 8), dpi=120)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
        
        # Top Panel: Cross Section
        ax0 = plt.subplot(gs[0])
        ax0.plot(lambda_sim_A, sigma_sim_mb, 'b-', linewidth=2.5, label='Simulation (SAE)')
        ax0.scatter(exp_lam, exp_mb, color='red', marker='s', s=50, label='Cooper (1962)', zorder=5)
        
        ax0.set_ylabel("Cross Section (Mb)", fontsize=12)
        ax0.set_title(f"{name} Photoionization: Simulation vs. Cooper (1962)", fontsize=14, fontweight='bold')
        ax0.legend()
        ax0.grid(True, alpha=0.3)
        ax0.set_xticklabels([]) # Hide x labels for top plot
        
        # Set limits based on data range
        ax0.set_xlim(np.max(lambda_sim_A), np.min(lambda_sim_A)) # Spectroscopy style (High Lambda left)
        
        # Bottom Panel: Residuals
        ax1 = plt.subplot(gs[1])
        ax1.axhline(0, color='black', linewidth=1, linestyle='--')
        ax1.plot(exp_lam, pct_error, 'r-o', markersize=4, linewidth=1)
        
        ax1.set_ylabel("Deviation (%)", fontsize=10)
        ax1.set_xlabel(r"Wavelength ($\AA$)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(ax0.get_xlim()) # Match x-axis
        
        # Save
        plt.savefig(f"results/Cooper_{species}_Detailed.png")
        plt.close()
        
        # ======================================================================
        # 5. PLOT 2: ARGON ZOOM (Special Request)
        # ======================================================================
        if species == 'Ar':
            print("    Generating Argon Cooper Minimum Zoom...")
            plt.figure(figsize=(8, 6), dpi=150)
            
            # Plot Curve
            plt.plot(lambda_sim_A, sigma_sim_mb, 'b-', linewidth=3, label='Simulation (SAE)')
            plt.scatter(exp_lam, exp_mb, color='red', marker='s', s=80, label='Cooper (1962)', zorder=5)
            
            # Find Minimum
            min_idx = np.argmin(sigma_sim_mb)
            min_lam = lambda_sim_A[min_idx]
            min_val = sigma_sim_mb[min_idx]
            
            # Focus on the relevant region
            plt.xlim(650, 150) # Standard spectroscopy direction
            plt.ylim(0, 5.0)   # Cut off the huge threshold peak
            
            # Annotation
            plt.annotate(f'Minimum:\n{min_val:.2f} Mb at {min_lam:.0f} $\AA$', 
                         xy=(min_lam, min_val), xytext=(min_lam+100, min_val+1.5),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", alpha=0.8))

            plt.title("Argon: Cooper Minimum Detail", fontsize=14, fontweight='bold')
            plt.xlabel(r"Wavelength ($\AA$)", fontsize=12)
            plt.ylabel("Cross Section (Mb)", fontsize=12)
            plt.legend()
            plt.grid(True, which='both', alpha=0.3)
            plt.minorticks_on()
            
            plt.savefig("results/Cooper_Ar_Zoom.png")
            plt.close()

    print("\n>>> Analysis Complete. Check 'results/' folder.")

if __name__ == "__main__":
    run_expert_verification()