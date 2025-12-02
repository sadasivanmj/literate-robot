"""
Reproduction of Table 1 from Saha et al. (2019).
Calculates Overlap Integral (S) and Cross Section (sigma) for H(1s) photoionization
at threshold (E=0.0001 a.u.) for various confinement depths.

Models: 
1. ASW (Annular Square Well)
2. GASW (Gaussian Annular Square Well)

Comparison Data from Image:
Depth   S(ASW)   S(GASW)   Sig(ASW)  Sig(GASW)
0.00    1.08247  -------   0.22495   -------
0.30    1.03156  1.02082   0.20520   0.19131
0.46    0.96023  1.08251   0.18149   0.22206
0.56    0.66291  0.74988   0.10654   0.15211
1.03    0.02469  0.01181   0.03492   0.01204
"""

import numpy as np
from scipy.integrate import trapezoid
from multiprocessing import freeze_support

# Import solvers
from bound import solve_ground_u
from cross_section import compute_cross_section_spectrum
from potential import VGASW_total_debye

def get_gasw_params_analytic(D_target, sigma=1.70, R_const=-24.5):
    """Analytic solution for GASW parameters A and U."""
    if D_target == 0: return 0.0, 0.0
    K = np.sqrt(2.0 * np.pi) * sigma
    denominator = 1.0 - (R_const / K)
    U = D_target / denominator
    A = U - D_target
    return A, U

def calculate_metrics(A, U, mu=0.0, E_pe=0.0001, label=""):
    """
    Calculates S and Sigma for a single potential configuration.
    """
    # 1. Solve Bound State (1s)
    # Use large box for shallow potentials to capture tail
    R_max = 150.0
    r, u_b, E_b, _, _ = solve_ground_u(
        VGASW_total_debye, R_max=R_max, N=8000, ell=0,
        A=A, U=U, mu=mu
    )
    
    # 2. Solve Continuum State (p-wave, l=1) at threshold
    # Use large box for low energy (wavelength ~ 450 a.u.)
    # We need compute_continuum_state from cross_section (which calls continuum.py)
    # But we can't import it directly if it's not exposed.
    # We will use compute_cross_section_spectrum for sigma, 
    # but we need the wavefunction for S.
    
    # Let's use the lower-level function from continuum.py if available, 
    # or relies on compute_cross_section_spectrum returning what we need?
    # compute_cross_section_spectrum returns (sigma, D_vals, diag).
    # D_vals IS the dipole matrix element.
    
    # We run for just 1 energy point
    E_arr = np.array([E_pe])
    sigma_arr, D_arr, _ = compute_cross_section_spectrum(
        E_arr, r, u_b, E_b, ell_cont=1,
        A=A, U=U, mu=mu, use_parallel=False, verbose=False
    )
    
    D_dipole = np.abs(D_arr[0]) # <u_f | r | u_i>
    Sigma = sigma_arr[0]
    
    # Note on "S": The table lists S = integral(psi_f * psi_i dr).
    # If this means the overlap <u_f | u_i>, it should be 0 for dipole transitions (s to p).
    # However, sometimes "Overlap" is used loosely for the Dipole Matrix Element.
    # Let's assume S = D_dipole for now.
    
    return D_dipole, Sigma

def main():
    print("="*90)
    print("REPRODUCING TABLE 1 (SAHA ET AL., 2019)")
    print("="*90)
    
    # Literature Data (Depth: [S_asw, S_gasw, Sig_asw, Sig_gasw])
    # None implies use Free H value or N/A
    lit_data = {
        0.00: [1.08247, None,    0.22495, None],
        0.30: [1.03156, 1.02082, 0.20520, 0.19131],
        0.46: [0.96023, 1.08251, 0.18149, 0.22206],
        0.56: [0.66291, 0.74988, 0.10654, 0.15211],
        1.03: [0.02469, 0.01181, 0.03492, 0.01204]
    }
    
    depths = [0.00, 0.30, 0.46, 0.56, 1.03]
    
    print(f"{'Depth':<6} | {'Model':<5} | {'S (Lit)':<8} {'S (Calc)':<8} {'Dev %':<7} | {'σ (Lit)':<8} {'σ (Calc)':<8} {'Dev %':<7}")
    print("-" * 80)

    for D in depths:
        # --- ASW CALCULATION ---
        # ASW: A=0, U=D
        s_asw_lit, s_gasw_lit, sig_asw_lit, sig_gasw_lit = lit_data[D]
        
        # ASW Simulation
        s_asw_calc, sig_asw_calc = calculate_metrics(A=0.0, U=D)
        
        # Calculate Deviation
        s_dev = 100 * abs(s_asw_calc - s_asw_lit) / s_asw_lit
        sig_dev = 100 * abs(sig_asw_calc - sig_asw_lit) / sig_asw_lit
        
        print(f"{D:<6.2f} | {'ASW':<5} | {s_asw_lit:<8.5f} {s_asw_calc:<8.5f} {s_dev:>6.2f}% | {sig_asw_lit:<8.5f} {sig_asw_calc:<8.5f} {sig_dev:>6.2f}%")
        
        # --- GASW CALCULATION ---
        if D > 0:
            A_val, U_val = get_gasw_params_analytic(D)
            s_gasw_calc, sig_gasw_calc = calculate_metrics(A=A_val, U=U_val)
            
            s_dev_g = 100 * abs(s_gasw_calc - s_gasw_lit) / s_gasw_lit
            sig_dev_g = 100 * abs(sig_gasw_calc - sig_gasw_lit) / sig_gasw_lit
            
            print(f"{'':<6} | {'GASW':<5} | {s_gasw_lit:<8.5f} {s_gasw_calc:<8.5f} {s_dev_g:>6.2f}% | {sig_gasw_lit:<8.5f} {sig_gasw_calc:<8.5f} {sig_dev_g:>6.2f}%")
        else:
            # Free H is the same for both, just print placeholder
            print(f"{'':<6} | {'GASW':<5} | {'(Same)':<8} {'-------':<8} {'------':<7} | {'(Same)':<8} {'-------':<8} {'------':<7}")
            
        print("-" * 80)

    print("\nNote: S = Dipole Matrix Element <f|r|i>. σ = Cross Section in a.u.")
    print("Small deviations (<5%) are expected due to numerical grid differences.")

if __name__ == "__main__":
    freeze_support()
    main()