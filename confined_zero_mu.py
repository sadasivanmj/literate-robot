"""
Reproduction of Figure 4 from Saha et al. (2019).
Target: Photoionization of H(1s) in H@C60 with ASW and GASW potentials.
Depths considered: 0.30, 0.46, 0.56, 1.03 a.u.

UPDATES:
- Compatible with new SAE library (passes species='H').
- Uses analytic GASW parameter solution.
- Generates 1x3 Layout.
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

# Import solvers
from bound import solve_ground_u
from cross_section import compute_cross_section_spectrum
from potential import VGASW_total_debye

def get_gasw_params_analytic(D_target, sigma=1.70, R_const=-24.5):
    """
    Analytic solution for GASW parameters A and U given target depth D.
    Constraint: (A * sqrt(2pi) * sigma) / U = R_const
    Constraint: A - U = -D_target
    """
    K = np.sqrt(2.0 * np.pi) * sigma
    denominator = 1.0 - (R_const / K)
    
    if abs(denominator) < 1e-10:
        raise ValueError("Singularity in GASW parameter solution (R/K ~ 1)")
        
    U = D_target / denominator
    A = U - D_target
    
    return A, U

def main():
    print("="*80)
    print("REPRODUCING SAHA ET AL. (2019) - FIGURE 4 ")
    print("="*80)

    # 1. Define Cases
    depths = [0.30, 0.46, 0.56, 1.03]
    
    # Energy Grid (a.u.) - Dense near threshold for resonance resolution
    E_dense = np.linspace(0.001, 0.15, 50)
    E_mid = np.linspace(0.16, 1.5, 100)
    E_tail = np.linspace(1.51, 3.0, 50)
    E_au = np.concatenate([E_dense, E_mid, E_tail])

    results_asw = {}
    results_gasw = {}
    
    # 2. Calculate Free H (Reference)
    print("Computing Free H...")
    # Update: Pass species='H', unpack l_initial
    r_free, u_free, E_free, l_init_free, _ = solve_ground_u(
        VGASW_total_debye, species='H', R_max=80.0, N=4000,
        A=0.0, U=0.0, mu=0.0
    )
    
    # Update: Pass l_initial, species='H'
    sigma_free, _ = compute_cross_section_spectrum(
        E_au, r_free, u_free, E_free, l_initial=l_init_free, species='H',
        A=0.0, U=0.0, mu=0.0, n_workers=None
    )
    
    # 3. Calculate ASW Cases (A=0, U=Depth)
    print("\nComputing ASW Cases...")
    for D in depths:
        print(f"  Depth U = {D:.2f} a.u.")
        r_b, u_b, E_b, l_init, _ = solve_ground_u(
            VGASW_total_debye, species='H', R_max=80.0, N=4000,
            A=0.0, U=D, mu=0.0
        )
        
        if E_b > -1e-5:
            # If atom becomes unbound or E > 0, sigma is 0
            sigma = np.zeros_like(E_au)
        else:
            sigma, _ = compute_cross_section_spectrum(
                E_au, r_b, u_b, E_b, l_initial=l_init, species='H',
                A=0.0, U=D, mu=0.0, n_workers=None
            )
        results_asw[D] = sigma

    # 4. Calculate GASW Cases (Analytic A, U)
    print("\nComputing GASW Cases (Analytic Parameters)...")
    for D in depths:
        A_val, U_val = get_gasw_params_analytic(D)
        print(f"  Depth V(rc) = {D:.2f} a.u. -> A={A_val:.4f}, U={U_val:.4f}")
        
        r_b, u_b, E_b, l_init, _ = solve_ground_u(
            VGASW_total_debye, species='H', R_max=80.0, N=4000,
            A=A_val, U=U_val, mu=0.0
        )
        
        if E_b > -1e-5:
            sigma = np.zeros_like(E_au)
        else:
            sigma, _ = compute_cross_section_spectrum(
                E_au, r_b, u_b, E_b, l_initial=l_init, species='H',
                A=A_val, U=U_val, mu=0.0, n_workers=None
            )
        results_gasw[D] = sigma

    # 5. Plotting (1x3 Layout)
    # ------------------------------------------------
    print("\nGenerating Figure 4...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colors matching paper style roughly
    colors = ['r', 'lime', 'b', 'cyan'] 
    
    # Panel (a): ASW
    ax1.plot(E_au, sigma_free, 'k-', lw=2, label='Free')
    for i, D in enumerate(depths):
        ax1.plot(E_au, results_asw[D], color=colors[i], lw=2, label=f'U={D:.2f}')
        
    ax1.set_xlabel("Photoelectron Energy (a.u.)", fontsize=12)
    ax1.set_ylabel("Cross Section (a.u.)", fontsize=12)
    ax1.text(0.05, 0.92, "(a) H@C60-ASW", transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1.5)
    ax1.set_ylim(0, 0.25)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel (b): GASW
    ax2.plot(E_au, sigma_free, 'k-', lw=2, label='Free')
    for i, D in enumerate(depths):
        ax2.plot(E_au, results_gasw[D], color=colors[i], lw=2, label=f'V(rc)={D:.2f}')
        
    ax2.set_xlabel("Photoelectron Energy (a.u.)", fontsize=12)
    ax2.set_ylabel("Cross Section (a.u.)", fontsize=12)
    ax2.text(0.05, 0.92, "(b) H@C60-GASW", transform=ax2.transAxes, fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1.5)
    ax2.set_ylim(0, 0.25)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)
    
    # INSET for GASW D=0.56 (Cooper Minimum-like feature)
    ax_ins = ax2.inset_axes([0.45, 0.35, 0.5, 0.3])
    D_ins = 0.56
    mask_ins = (E_au >= 0.5) & (E_au <= 3.0)
    ax_ins.plot(E_au[mask_ins], results_gasw[D_ins][mask_ins], 'b-', lw=2)
    ax_ins.set_xlabel("Energy", fontsize=8)
    ax_ins.set_ylabel("σ", fontsize=8)
    ax_ins.set_xlim(0.5, 3.0)
    ax_ins.set_ylim(0, 0.005)
    ax_ins.grid(alpha=0.3)
    ax_ins.text(0.1, 0.8, f'V(rc)={D_ins}', transform=ax_ins.transAxes, fontsize=8, color='b')

    # Panel (c): Comparison of ASW vs GASW
    ax3.plot(E_au, sigma_free, 'k-', lw=2, label='Free')
    
    # Compare D=0.46
    ax3.plot(E_au, results_asw[0.46], 'r-', lw=2, label='ASW 0.46')
    ax3.plot(E_au, results_gasw[0.46], 'lime', lw=2, ls='--', label='GASW 0.46')
    
    # Compare D=0.56
    ax3.plot(E_au, results_asw[0.56], 'b-', lw=2, label='ASW 0.56')
    ax3.plot(E_au, results_gasw[0.56], 'c--', lw=2, label='GASW 0.56')
    
    ax3.set_xlabel("Photoelectron Energy (a.u.)", fontsize=12)
    ax3.set_ylabel("Cross Section (a.u.)", fontsize=12)
    ax3.text(0.05, 0.92, "(c) Comparison", transform=ax3.transAxes, fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 1.5)
    ax3.set_ylim(0, 0.25)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('saha_fig4_reproduction.png', dpi=300)
    print("\n✓ Plot saved: saha_fig4_reproduction.png")
    plt.show()

if __name__ == "__main__":
    freeze_support()
    main()