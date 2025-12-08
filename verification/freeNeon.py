import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import your physics stack
from potential import VGASW_total_debye
from bound import solve_ground_u
from cross_section import compute_cross_section_spectrum

# ==============================================================================
# 1. THEORETICAL DATA (Verner & Yakovlev 1996)
# ==============================================================================
theory_ev = np.array([21.6, 22.0, 24.0, 26.0, 28.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0, 100.0])
theory_mb = np.array([6.30, 6.25, 6.00, 5.75, 5.50, 5.30, 4.80, 4.30, 3.50, 2.80, 1.90, 1.30])

def run_deviation_analysis():
    print(">>> Calculating Percentage Deviation for Neon (2p)...")
    
    # Constants
    Ha_to_eV = 27.211386
    OCCUPATION_NUMBER = 6.0 # Neon 2p^6
    
    # 1. Solve Bound State
    species = 'Ne'
    r_b, u_b, E_b, l_init, _ = solve_ground_u(
        VGASW_total_debye, species=species, R_max=80.0, N=4000, 
        A=0.0, U=0.0, mu=0.0
    )
    Ip_eV = abs(E_b) * Ha_to_eV
    print(f"    Ionization Potential: {Ip_eV:.4f} eV")

    # 2. Convert Theory Photon Energy -> Target Kinetic Energy (a.u.)
    # Kinetic = (Photon - Ip) / 27.211
    # Note: If Photon < Ip, we clip to a small number to avoid errors, though theory_ev > Ip usually.
    target_kin_eV = theory_ev - Ip_eV
    target_kin_au = target_kin_eV / Ha_to_eV
    
    # Filter out any negative energies (below threshold)
    valid_mask = target_kin_au > 0.001
    calc_kin_au = target_kin_au[valid_mask]
    valid_theory_ev = theory_ev[valid_mask]
    valid_theory_mb = theory_mb[valid_mask]

    # 3. Compute Simulation EXACTLY at these points
    sigma_au_raw, _ = compute_cross_section_spectrum(
        calc_kin_au, r_b, u_b, E_b, l_init, 
        species, 0.0, 0.0, 0.0, n_workers=1
    )
    
    # 4. Scale and Convert
    # Multiply by 6 because SAE calculates per-electron, Theory is total shell
    sigma_mb_sim = sigma_au_raw * 28.0028 * OCCUPATION_NUMBER

    # 5. Calculate Deviation
    # Deviation = (Sim - Theory) / Theory * 100
    deviation_pct = ((sigma_mb_sim - valid_theory_mb) / valid_theory_mb) * 100.0
    
    # 6. Create Dataframe for Display
    df = pd.DataFrame({
        'Energy (eV)': valid_theory_ev,
        'Theory (Mb)': valid_theory_mb,
        'Sim (Mb)': np.round(sigma_mb_sim, 3),
        'Diff (Mb)': np.round(sigma_mb_sim - valid_theory_mb, 3),
        'Deviation (%)': np.round(deviation_pct, 2)
    })
    
    print("\n" + "="*60)
    print("   NEON CROSS SECTION DEVIATION ANALYSIS (Scaled N=6)")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    avg_dev = np.mean(np.abs(deviation_pct))
    print(f"\nAverage Absolute Deviation: {avg_dev:.2f}%")

    # 7. Plot for visual confirmation
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(valid_theory_ev, valid_theory_mb, 'rs-', label='Theory (Verner 1996)')
    plt.plot(valid_theory_ev, sigma_mb_sim, 'bo--', label='Simulation (Scaled x6)')
    plt.fill_between(valid_theory_ev, valid_theory_mb, sigma_mb_sim, color='gray', alpha=0.1)
    
    plt.title(f"Neon Validation (Avg Deviation: {avg_dev:.1f}%)")
    plt.xlabel("Photon Energy (eV)")
    plt.ylabel("Cross Section (Mb)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("Neon_Deviation_Table.png")
    print(">>> Plot saved to 'Neon_Deviation_Table.png'")

if __name__ == "__main__":
    run_deviation_analysis()