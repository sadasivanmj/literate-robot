"""
Photoionization cross section of non-confined hydrogen at different plasma densities.
Optimized for Windows parallel processing and high-throughput execution.

UPDATES:
- FIXED Unit Conversion: 1 a.u. = 28.0 Mb (was 0.28)
- Added explicit 'a.u.' columns for direct validation.
- Explicitly calculates at E=0.0001 a.u.
- Updated density scan range: [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
- Added missing x-axis label to Near-Threshold plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import freeze_support

# Import solvers
from bound import solve_ground_u
from cross_section import compute_cross_section_spectrum
from potential import VGASW_total_debye

def main():
    print("="*80)
    print("PHOTOIONIZATION CROSS SECTION vs PLASMA DENSITY")
    print("Non-confined Hydrogen: A = U = 0, varying μ")
    print("="*80)

    # ============================================================================
    # SETUP
    # ============================================================================
    # Debye screening parameter μ (Updated detailed range)
    mu_values = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    mu_labels = [f'μ={mu}' if mu > 0 else 'Free' for mu in mu_values]

    # Energy Grid: Add 0.0001 explicitly for threshold analysis
    E_base = np.linspace(0.01, 2.0, 80)
    E_pe_array = np.insert(E_base, 0, 0.0001) # [0.0001, 0.01, ..., 2.0]
    
    print(f"Densities to scan: {len(mu_values)}")
    print(f"Energy points per scan: {len(E_pe_array)}")
    
    # Container for results
    results = {}
    colors = plt.cm.viridis(np.linspace(0, 1, len(mu_values)))

    # ============================================================================
    # MAIN COMPUTATION LOOP
    # ============================================================================
    total_start = time.time()

    for idx, (mu, label) in enumerate(zip(mu_values, mu_labels)):
        print(f"\n[{idx+1}/{len(mu_values)}] Processing {label}...")
        
        t0 = time.time()
        
        # 1. Solve Bound State (Serial)
        # For high screening (weak binding), the wavefunction delocalizes, needing a larger box.
        R_box = 150.0 if mu > 0.5 else 100.0
        
        try:
            r_bound, u_bound, E_bound, Veff, norm = solve_ground_u(
                VGASW_total_debye,
                R_max=R_box, N=6000, ell=0,
                A=0.0, U=0.0, mu=mu
            )
            
            if E_bound > -0.001:
                print(f"   -> Unbound or very weak (E={E_bound:.5f}). Skipping.")
                results[mu] = {'bound_state_exists': False, 'E_bound': E_bound}
                continue

            # 2. Compute Cross Sections (Parallel over Energies)
            # Result 'sigma' is in Atomic Units (a.u.)
            sigma_au, D, diag = compute_cross_section_spectrum(
                E_pe_array,
                r_bound, u_bound, E_bound,
                ell_cont=1,
                A=0.0, U=0.0, mu=mu,
                use_parallel=True,
                verbose=False
            )
            
            # Convert to Megabarns (1 a.u. area ≈ 28.0 Mb)
            sigma_Mb = sigma_au * 28.0028
            
            peak_idx = np.argmax(sigma_au)
            peak_au = sigma_au[peak_idx]
            thresh_au = sigma_au[0]
            
            print(f"   -> E_1s: {E_bound:.5f} a.u.")
            print(f"   -> Threshold (E~0): {thresh_au:.4f} a.u. ({sigma_Mb[0]:.2f} Mb)")

            results[mu] = {
                'bound_state_exists': True,
                'E_bound': E_bound,
                'sigma_au': sigma_au,
                'sigma_Mb': sigma_Mb,
                'r_bound': r_bound,
                'u_bound': u_bound,
                'mu': mu,
                'label': label,
                'threshold_sigma_au': thresh_au
            }

        except Exception as e:
            print(f"   -> ERROR: {e}")
            results[mu] = {'bound_state_exists': False, 'error': str(e)}

    total_time = time.time() - total_start
    print("\n" + "="*80)
    print(f"All calculations finished in {total_time:.2f} seconds.")
    print("="*80)

    # ============================================================================
    # DATA TABLE GENERATION
    # ============================================================================
    print("\n" + "="*90)
    print(f"{'DATA SUMMARY':^90}")
    print("="*90)
    # Table Header
    print(f"{'μ (a.u.⁻¹)':<10} {'Binding E':<12} {'Peak σ (a.u.)':<15} {'Threshold σ (a.u.)':<20} {'Thresh σ (Mb)':<15}")
    print("-" * 90)
    
    for mu in mu_values:
        res = results.get(mu, {})
        if res.get('bound_state_exists'):
            E_b = res['E_bound']
            peak = res['sigma_au'].max()
            thresh_au = res['threshold_sigma_au']
            thresh_mb = thresh_au * 28.0028
            
            print(f"{mu:<10.3f} {E_b:<12.5f} {peak:<15.4f} {thresh_au:<20.5f} {thresh_mb:<15.2f}")
        else:
            print(f"{mu:<10.3f} {'Unbound':<12} {'---':<15} {'---':<20} {'---':<15}")
    
    print("="*90)
    print("VALIDATION NOTE: Free Hydrogen (μ=0) threshold should be ~0.22225 a.u.")
    print("="*90)

    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    plot_results(results, mu_values, E_pe_array, colors)


def plot_results(results, mu_values, E_pe_array, colors):
    """
    Separate plotting logic to keep main clean.
    """
    print("Generating plots...")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # --- Panel 1: Cross Sections (a.u.) ---
    ax1 = fig.add_subplot(gs[0, :])
    for mu, color in zip(mu_values, colors):
        res = results[mu]
        if res.get('bound_state_exists'):
            ax1.plot(E_pe_array, res['sigma_au'], color=color, lw=2, 
                     label=res['label'], alpha=0.8)
    
    ax1.set_xlabel('Photoelectron Energy (a.u.)')
    ax1.set_ylabel('Cross Section (a.u.)')
    ax1.set_title('Photoionization Cross Section (Atomic Units)', fontweight='bold')
    ax1.legend(ncol=2)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 2.0)
    # Add marker for theoretical limit
    ax1.plot(0.0001, 0.22225, 'rx', markersize=10, markeredgewidth=2, label='Theory (Free)')

    # --- Panel 2: Peak Sigma vs Mu ---
    ax2 = fig.add_subplot(gs[1, 0])
    mus = []
    peaks = []
    for mu in mu_values:
        res = results[mu]
        if res.get('bound_state_exists'):
            mus.append(mu)
            peaks.append(res['sigma_au'].max())
            
    ax2.plot(mus, peaks, 'o-', color='navy')
    ax2.set_xlabel('Debye Parameter μ (a.u.⁻¹)')
    ax2.set_ylabel('Peak Cross Section (a.u.)')
    ax2.set_title('Peak Magnitude vs Screening')
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Binding Energy vs Mu ---
    ax3 = fig.add_subplot(gs[1, 1])
    energies = [results[mu]['E_bound'] for mu in mus] # matching valid mus
    ax3.plot(mus, energies, 's-', color='darkred')
    ax3.axhline(-0.5, ls='--', color='gray', alpha=0.5)
    ax3.set_xlabel('Debye Parameter μ')
    ax3.set_ylabel('Binding Energy (a.u.)')
    ax3.set_title('Ground State Energy Shift')
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Wavefunctions ---
    ax4 = fig.add_subplot(gs[2, 0])
    for mu, color in zip(mu_values[:5], colors[:5]): # Limit to first 5
        res = results[mu]
        if res.get('bound_state_exists'):
            r = res['r_bound']
            u = res['u_bound']
            mask = r < 20
            ax4.plot(r[mask], u[mask], color=color, label=res['label'])
    ax4.set_title('Wavefunction Delocalization')
    ax4.set_xlabel('r (a.u.)')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # --- Panel 5: Threshold Zoom ---
    ax5 = fig.add_subplot(gs[2, 1])
    mask_thresh = E_pe_array < 0.5
    for mu, color in zip(mu_values, colors):
        res = results[mu]
        if res.get('bound_state_exists'):
            ax5.plot(E_pe_array[mask_thresh], res['sigma_au'][mask_thresh], 
                     color=color, marker='o', ms=3)
    
    # Mark theoretical point
    ax5.plot(0, 0.22225, 'rx', markersize=8, markeredgewidth=2)
    ax5.set_ylabel('Cross Section (a.u.)')
    ax5.set_xlabel('Photoelectron Energy (a.u.)')  # Added Label
    ax5.set_title('Near-Threshold Behavior (a.u.)')
    ax5.grid(alpha=0.3)

    plt.savefig('plasma_scan_results.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved: plasma_scan_results.png")
    plt.show()

if __name__ == "__main__":
    freeze_support()
    main()