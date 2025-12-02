"""
Reproduction of Figure 8(a) from Qi et al. (Phys. Rev. A 80, 063404, 2009).
Target: Photoionization of H(1s) in Debye plasmas.

Key parameters from paper:
- Screening lengths delta: 50, 20, 9, 8.86, 5, 4.52, 1 (in Bohr radii)
- Units: Energy in Ry, Cross section in 10^-16 cm^2
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

# Import solvers
from bound import solve_ground_u
from cross_section import compute_cross_section_spectrum
from potential import VGASW_total_debye

def main():
    print("="*80)
    print("REPRODUCING QI ET AL. (2009) - FIGURE 8a")
    print("="*80)

    # 1. Define Screening Parameters (mu = 1/delta)
    # ------------------------------------------------
    # Note: The paper uses delta = Z*D. For H, Z=1, so mu = 1/delta.
    deltas = [50.0, 20.0, 9.0, 8.86, 5.0, 4.52, 1.0]
    mu_values = [1.0/d for d in deltas]
    mu_labels = [f"δ={d} (μ={m:.4f})" for d, m in zip(deltas, mu_values)]
    
    # Add "No Screening" case
    deltas.insert(0, float('inf'))
    mu_values.insert(0, 0.0)
    mu_labels.insert(0, "No Screening")

    # 2. Define Energy Grid (Logarithmic + Resonance Points)
    # ------------------------------------------------
    # Range: 10^-6 Ry to 10^1 Ry
    # Convert to a.u.: E_au = E_Ry / 2
    
    # Base logarithmic grid
    E_Ry_log = np.logspace(-6, 1.2, 300) 
    
    # Specific resonance energies from paper text (in Ry)
    # Resonance 1: delta=8.86 -> E ~ 1.85e-5 Ry
    # Resonance 2: delta=4.52 -> E ~ 2.58e-4 Ry
    E_resonances = np.array([1.85e-5, 2.58e-4])
    
    # Add dense clusters around resonances to capture peaks
    E_dense_1 = np.linspace(1.80e-5, 1.90e-5, 50)
    E_dense_2 = np.linspace(2.50e-4, 2.70e-4, 50)
    
    # Combine and sort
    E_Ry_all = np.unique(np.concatenate([E_Ry_log, E_dense_1, E_dense_2]))
    E_pe_au = E_Ry_all / 2.0 # Convert to a.u. for calculation
    
    print(f"Calculations per curve: {len(E_pe_au)} energy points")

    # 3. Run Calculations
    # ------------------------------------------------
    results = {}
    colors = ['k', 'grey', 'lightblue', 'teal', 'lime', 'green', 'red', 'gray']
    styles = ['-', '--', ':', '-.', '-', ':', '-', ':']

    for idx, (mu, label) in enumerate(zip(mu_values, mu_labels)):
        print(f"Processing: {label}...")
        
        # Solve Bound State
        # Use large box for low mu to capture delocalization
        R_box = 250.0 if mu < 0.1 else 100.0
        
        r_bound, u_bound, E_bound, _, _ = solve_ground_u(
            VGASW_total_debye, R_max=R_box, N=6000, ell=0,
            A=0.0, U=0.0, mu=mu
        )
        
        if E_bound > -1e-5:
            print(f"  -> Unbound state! Skipping.")
            continue

        # Compute Cross Sections (Parallel)
        sigma_au, _, _ = compute_cross_section_spectrum(
            E_pe_au, r_bound, u_bound, E_bound, ell_cont=1,
            A=0.0, U=0.0, mu=mu, use_parallel=True, verbose=False
        )
        
        # Convert units for plotting
        # Sigma a.u. -> 10^-16 cm^2
        # 1 a.u. = 28.0 Mb = 0.28 * 10^-16 cm^2
        sigma_plot = sigma_au * 0.280028
        
        results[label] = sigma_plot

    # 4. Plotting (Imitating Fig 8a)
    # ------------------------------------------------
    plt.figure(figsize=(8, 10))
    
    for i, label in enumerate(results.keys()):
        # Skip plotting delta=1 if it's too small (like in the paper, it drops off fast)
        plt.loglog(E_Ry_all, results[label], 
                   color=colors[i], linestyle=styles[i], linewidth=2, label=label)

    # Match axes limits from paper
    plt.xlim(1e-6, 20)
    plt.ylim(1e-4, 200) # To fit the peaks
    
    plt.xlabel("Scaled Photoelectron Energy (Ry)", fontsize=12)
    plt.ylabel(r"$Z^2 \sigma$ (units of $10^{-16}$ cm$^2$)", fontsize=12)
    plt.title("Reproduction of Qi et al. (2009) Fig. 8a", fontsize=14)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    
    plt.savefig("reproduce_qi_fig8.png", dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: reproduce_qi_fig8.png")
    plt.show()

if __name__ == "__main__":
    freeze_support()
    main()