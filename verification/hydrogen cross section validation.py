"""
Hydrogen 1s Photoionization Verification.
Compares Numerical Calculation (Saha Pipeline) vs Exact Analytical Formula.
"""
import numpy as np
import matplotlib.pyplot as plt
from bound import solve_ground_u
from potential import VGASW_total_debye
from cross_section import compute_cross_section_spectrum

# Fine-structure constant
ALPHA = 1.0 / 137.036

def get_analytical_H_sigma(E_pe_array):
    """
    Exact analytical cross section for H(1s) photoionization.
    Ref: Bethe & Salpeter, Quantum Mechanics of One- and Two-Electron Atoms.
    
    Formula (Atomic Units):
        sigma = (512 * pi^2 * alpha / 3) * (I / omega)^4 * f(k)
        f(k) = exp(-4*eta*arccot(eta)) / (1 - exp(-2*pi*eta))
    """
    sigma_exact = []
    I_H = 0.5  # Ionization potential (0.5 a.u.)
    
    for E in E_pe_array:
        omega = I_H + E
        k = np.sqrt(2 * E)
        eta = 1.0 / k
        
        # Pre-factor
        pre = (512 * np.pi**2 * ALPHA / 3.0) * (I_H / omega)**4
        
        # Exponential factor
        # arccot(eta) = arctan(1/eta) = arctan(k)
        arg = -4.0 * eta * np.arctan(k)
        num = np.exp(arg)
        den = 1.0 - np.exp(-2 * np.pi * eta)
        
        sigma = pre * (num / den)
        sigma_exact.append(sigma)
        
    return np.array(sigma_exact)

def run_verification():
    print("="*70)
    print("VERIFICATION: Hydrogen 1s Photoionization Cross-Section")
    print("="*70)

    # 1. Solve Bound State (Free H)
    print("\n[1] Solving Bound State (H 1s)...")
    r_bound, u_bound, E_bound, _, _ = solve_ground_u(
        VGASW_total_debye, species='H', R_max=60.0, N=4000, A=0.0, U=0.0
    )
    
    print(f"    E_calc: {E_bound:.6f} a.u.")
    print(f"    E_exact: -0.500000 a.u.")
    
    if abs(E_bound + 0.5) > 1e-4:
        print("\n❌ CRITICAL ERROR: Bound state energy is wrong.")
        print("   Please check bound.py for the 'off-by-one' array size bug.")
        return

    # 2. Compute Cross Section
    print("\n[2] Computing Numerical Cross Section...")
    # Grid: Threshold to high energy
    E_pe = np.linspace(0.01, 2.0, 50) 
    
    sigma_num, details = compute_cross_section_spectrum(
        E_pe, r_bound, u_bound, E_bound,
        l_initial=0, species='H', 
        A=0.0, U=0.0, mu=0.0,
        n_workers=None,
        method='standard'
    )
    
    # 3. Compute Analytical Result
    print("[3] Computing Analytical Benchmark...")
    sigma_ana = get_analytical_H_sigma(E_pe)
    
    # 4. Compare
    error = np.abs(sigma_num - sigma_ana)
    rel_error = error / sigma_ana
    max_rel_err = np.max(rel_error) * 100
    
    print(f"\n[4] Results:")
    print(f"    Max Relative Error: {max_rel_err:.2f}%")
    
    # 5. Plot
    plt.figure(figsize=(8, 6))
    plt.plot(E_pe, sigma_ana, 'k-', linewidth=2, label='Analytical (Exact)')
    plt.plot(E_pe, sigma_num, 'ro', markersize=5, label='Numerical (Code)')
    
    plt.xlabel('Photoelectron Energy (a.u.)')
    plt.ylabel('Cross Section (a.u.)')
    plt.title('Hydrogen 1s Photoionization Verification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log-Log inset to check threshold behavior
    ax_ins = plt.axes([0.5, 0.5, 0.35, 0.35])
    ax_ins.loglog(E_pe, sigma_ana, 'k-')
    ax_ins.loglog(E_pe, sigma_num, 'r.')
    ax_ins.set_title("Log-Log Scale")
    ax_ins.grid(True, which='both', alpha=0.2)
    
    plt.savefig('hydrogen_verification.png')
    plt.show()
    
    if max_rel_err < 1.0:
        print("\n✅ SUCCESS: Numerical engine matches theory (<1% error).")
        print("   You can now proceed to the Argon simulation.")
    else:
        print("\n⚠️ WARNING: Deviation detected. Check grid density or normalization.")

if __name__ == "__main__":
    run_verification()