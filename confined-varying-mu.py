"""
Photoionization cross section of non-confined hydrogen at different plasma densities.
FIXED for compatibility with the new SAE Library structure.
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
    print("PHOTOIONIZATION CROSS SECTION vs PLASMA DENSITY (SAE Fixed)")
    print("Non-confined Hydrogen: A = U = 0, varying μ")
    print("="*80)

    # ============================================================================
    # SETUP
    # ============================================================================
    # Debye screening parameter μ
    mu_values = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    mu_labels = [f'μ={mu}' if mu > 0 else 'Free' for mu in mu_values]

    # Energy Grid (a.u.)
    E_base = np.linspace(0.01, 2.0, 80)
    E_pe_array = np.insert(E_base, 0, 0.0001) 
    
    print(f"Densities to scan: {len(mu_values)}")
    
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
        R_box = 150.0 if mu > 0.5 else 100.0
        
        try:
            # UPDATED CALL: Pass species='H', unpack l_init
            r_bound, u_bound, E_bound, l_init, norm = solve_ground_u(
                VGASW_total_debye,
                species='H',
                R_max=R_box, N=6000,
                A=0.0, U=0.0, mu=mu
            )
            
            if E_bound > -0.001:
                print(f"    -> Unbound or very weak (E={E_bound:.5f}). Skipping.")
                results[mu] = {'bound_state_exists': False, 'E_bound': E_bound}
                continue

            # 2. Compute Cross Sections (Parallel)
            # UPDATED CALL: Matches new cross_section.py signature
            sigma_au, diag = compute_cross_section_spectrum(
                E_pe_array,
                r_bound, u_bound, E_bound,
                l_initial=l_init,  # Use l from bound solver
                species='H',       # Required for SAE
                A=0.0, U=0.0, mu=mu,
                n_workers=None     # Auto-detect
            )
            
            # Convert to Megabarns
            sigma_Mb = sigma_au * 28.0028
            
            peak_idx = np.argmax(sigma_au)
            peak_au = sigma_au[peak_idx]
            thresh_au = sigma_au[0]
            
            print(f"    -> E_1s: {E_bound:.5f} a.u.")
            print(f"    -> Threshold: {thresh_au:.4f} a.u. ({sigma_Mb[0]:.2f} Mb)")

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
            print(f"    -> ERROR: {e}")
            # Raise to see full trace if needed, or store error
            import traceback
            traceback.print_exc()
            results[mu] = {'bound_state_exists': False, 'error': str(e)}

    total_time = time.time() - total_start
    print("\n" + "="*80)
    print(f"All calculations finished in {total_time:.2f} seconds.")
    print("="*80)

    # ============================================================================
    # DATA TABLE & PLOTTING
    # ============================================================================
    print("\n" + "="*90)
    print(f"{'DATA SUMMARY':^90}")
    print("="*90)
    print(f"{'μ (a.u.⁻¹)':<10} {'Binding E':<12} {'Peak σ (a.u.)':<15} {'Threshold σ':<15}")
    print("-" * 90)
    
    for mu in mu_values:
        res = results.get(mu, {})
        if res.get('bound_state_exists'):
            print(f"{mu:<10.3f} {res['E_bound']:<12.5f} {res['sigma_au'].max():<15.4f} {res['threshold_sigma_au']:<15.5f}")
        else:
            print(f"{mu:<10.3f} {'Unbound':<12} {'---':<15} {'---':<15}")
    
    plot_results(results, mu_values, E_pe_array, colors)


def plot_results(results, mu_values, E_pe_array, colors):
    print("Generating plots...")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Cross Sections
    ax1 = fig.add_subplot(gs[0, :])
    for mu, color in zip(mu_values, colors):
        res = results[mu]
        if res.get('bound_state_exists'):
            ax1.plot(E_pe_array, res['sigma_au'], color=color, lw=2, label=res['label'])
    
    ax1.set_xlabel('Photoelectron Energy (a.u.)')
    ax1.set_ylabel('Cross Section (a.u.)')
    ax1.set_title('Photoionization Cross Section (Atomic Units)', fontweight='bold')
    ax1.legend(ncol=2)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 2.0)
    # Theory point
    ax1.plot(0.0001, 0.22225, 'rx', markersize=10, markeredgewidth=2, label='Theory')

    # Panel 2: Peak vs Mu
    ax2 = fig.add_subplot(gs[1, 0])
    mus = [mu for mu in mu_values if results[mu].get('bound_state_exists')]
    peaks = [results[mu]['sigma_au'].max() for mu in mus]
    ax2.plot(mus, peaks, 'o-', color='navy')
    ax2.set_xlabel('Debye Parameter μ')
    ax2.set_ylabel('Peak σ (a.u.)')
    ax2.grid(alpha=0.3)

    # Panel 3: Binding Energy
    ax3 = fig.add_subplot(gs[1, 1])
    enes = [results[mu]['E_bound'] for mu in mus]
    ax3.plot(mus, enes, 's-', color='darkred')
    ax3.set_xlabel('Debye Parameter μ')
    ax3.set_ylabel('Binding Energy (a.u.)')
    ax3.grid(alpha=0.3)

    # Panel 4: Wavefunctions
    ax4 = fig.add_subplot(gs[2, 0])
    for mu, color in zip(mu_values[:5], colors[:5]):
        res = results[mu]
        if res.get('bound_state_exists'):
            r = res['r_bound']
            u = res['u_bound']
            ax4.plot(r[r<20], u[r<20], color=color, label=res['label'])
    ax4.set_title('Wavefunction Delocalization')
    ax4.grid(alpha=0.3)

    # Panel 5: Threshold Zoom
    ax5 = fig.add_subplot(gs[2, 1])
    mask = E_pe_array < 0.5
    for mu, color in zip(mu_values, colors):
        res = results[mu]
        if res.get('bound_state_exists'):
            ax5.plot(E_pe_array[mask], res['sigma_au'][mask], color=color, marker='o', ms=3)
    ax5.plot(0, 0.22225, 'rx')
    ax5.set_title('Near-Threshold Behavior')
    ax5.grid(alpha=0.3)

    plt.savefig('plasma_scan_results.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved: plasma_scan_results.png")
    plt.show()

if __name__ == "__main__":
    freeze_support()
    main()