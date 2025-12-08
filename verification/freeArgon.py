import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import your physics modules
from potential import VGASW_total_debye
from bound import solve_ground_u
from cross_section import compute_cross_section_spectrum

# ==============================================================================
# 1. EXPERIMENTAL DATA (Samson & Stolte 2002)
# ==============================================================================
ss_photon_ev = np.array([
    15.76, 16.0, 17.0, 18.0, 19.0, 20.0, 22.0, 24.0, 26.0, 28.0,
    30.0, 35.0, 40.0, 45.0, 48.0, 50.0, 55.0, 60.0, 70.0, 80.0
])

ss_sigma_mb = np.array([
    35.0, 34.5, 32.0, 29.5, 27.0, 24.5, 19.8, 15.5, 11.8, 8.8,
    6.4,  2.4,  0.8,  0.3,  0.2,  0.25, 0.5,  0.8,  1.2,  1.4
])

def analyze_cooper_minimum():
    print(">>> Simulating Free Argon for Cooper Minimum Analysis...")
    
    # 1. Physics Setup
    Ha_to_eV = 27.211386
    OCCUPATION = 6.0 # Argon 3p^6 (Scaling factor for magnitude)
    
    r_b, u_b, E_b, l_init, _ = solve_ground_u(
        VGASW_total_debye, species='Ar', R_max=80.0, N=4000, 
        A=0.0, U=0.0, mu=0.0
    )
    Ip_eV = abs(E_b) * Ha_to_eV

    # 2. Compute Spectrum (High res grid around the minimum)
    # Cooper min is expected around 40-55 eV
    e_kin_au = np.concatenate([
        np.linspace(0.01, 1.0, 50),   # Threshold region
        np.linspace(1.05, 3.0, 50)    # Tail region
    ])
    
    sigma_au, _ = compute_cross_section_spectrum(
        e_kin_au, r_b, u_b, E_b, l_init, 
        'Ar', 0.0, 0.0, 0.0, n_workers=1
    )
    
    # Convert to Plotting Units
    sim_ev = (e_kin_au * Ha_to_eV) + Ip_eV
    # We multiply by 6 to account for the full 3p^6 shell occupation
    sim_mb = sigma_au * 28.0028 * OCCUPATION 

    # ==========================================================================
    # 3. ANALYSIS: FIND THE MINIMA
    # ==========================================================================
    
    # A. Experimental Minimum
    idx_exp_min = np.argmin(ss_sigma_mb)
    exp_min_E = ss_photon_ev[idx_exp_min]
    exp_min_sig = ss_sigma_mb[idx_exp_min]
    
    # B. Simulation Minimum
    # We look for the minimum in the region > 30 eV to avoid threshold behavior
    mask = sim_ev > 30.0 
    masked_mb = sim_mb[mask]
    masked_ev = sim_ev[mask]
    
    idx_sim_min = np.argmin(masked_mb)
    sim_min_E = masked_ev[idx_sim_min]
    sim_min_sig = masked_mb[idx_sim_min]

    # C. Print Comparison
    print("\n" + "="*50)
    print("   COOPER MINIMUM COMPARISON (Argon)")
    print("="*50)
    print(f"{'Feature':<15} | {'Experiment':<15} | {'Simulation':<15}")
    print("-" * 50)
    print(f"{'Position (eV)':<15} | {exp_min_E:<15.2f} | {sim_min_E:<15.2f}")
    print(f"{'Depth (Mb)':<15} | {exp_min_sig:<15.3f} | {sim_min_sig:<15.3f}")
    print("-" * 50)
    
    shift = sim_min_E - exp_min_E
    print(f"\n>>> Energy Shift: {shift:+.2f} eV")
    if abs(shift) < 5.0:
        print(">>> STATUS: EXCELLENT AGREEMENT (Physics is verified)")
    else:
        print(">>> STATUS: DEVIATION DETECTED (Check potential parameters)")

    # ==========================================================================
    # 4. PLOTTING
    # ==========================================================================
    plt.figure(figsize=(9, 7), dpi=120)
    
    # Plot Data
    plt.scatter(ss_photon_ev, ss_sigma_mb, color='red', s=50, label='Experiment (Samson 2002)')
    plt.plot(sim_ev, sim_mb, 'b-', linewidth=2, label='Simulation (SAE x 6)')
    
    # Highlight Minima
    plt.plot(exp_min_E, exp_min_sig, 'ko', markersize=10, markerfacecolor='none', 
             markeredgewidth=2, label='Exp. Minimum')
    plt.plot(sim_min_E, sim_min_sig, 'bx', markersize=10, markeredgewidth=3, 
             label='Sim. Minimum')
    
    # Arrows
    plt.annotate(f'Exp: {exp_min_E} eV', xy=(exp_min_E, exp_min_sig), 
                 xytext=(exp_min_E-10, exp_min_sig+5),
                 arrowprops=dict(facecolor='red', shrink=0.05))
                 
    plt.annotate(f'Sim: {sim_min_E:.1f} eV', xy=(sim_min_E, sim_min_sig), 
                 xytext=(sim_min_E+5, sim_min_sig+5),
                 arrowprops=dict(facecolor='blue', shrink=0.05))

    plt.title('Argon Cooper Minimum Analysis', fontsize=16)
    plt.xlabel('Photon Energy (eV)', fontsize=14)
    plt.ylabel('Cross Section (Mb)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(15, 80)
    plt.ylim(0, 45) # Adjusted for scaled cross section
    
    plt.tight_layout()
    plt.savefig('Argon_Cooper_Comparison.png')
    print(">>> Plot saved to 'Argon_Cooper_Comparison.png'")

if __name__ == "__main__":
    analyze_cooper_minimum()