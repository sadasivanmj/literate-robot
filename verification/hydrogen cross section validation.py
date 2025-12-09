"""
Sanity Check: Free Hydrogen 1s
"""
import numpy as np
import matplotlib.pyplot as plt
from bound import solve_ground_u
from potential import VGASW_total_debye

def run_test():
    print("="*60)
    print("HYDROGEN 1s SANITY CHECK")
    print("="*60)
    
    # Solve for Free Hydrogen (A=0, U=0, Z=1)
    r, u, E, V, norm = solve_ground_u(
        VGASW_total_debye, 
        species='H', 
        R_max=60.0, 
        N=4000, 
        A=0.0, U=0.0
    )
    
    theory_E = -0.5
    error = abs(E - theory_E)
    
    print(f"Energy Calculated: {E:.6f} a.u.")
    print(f"Theory (Exact):    {theory_E:.6f} a.u.")
    print(f"Error:             {error:.2e} a.u.")
    print(f"Normalization:     {norm:.6f}")
    
    if error < 1e-4:
        print("\n✅ SUCCESS: Bound state solver is working correctly.")
        
        # Quick Plot
        plt.figure(figsize=(6,4))
        plt.plot(r, u, label='Calculated 1s')
        # Analytic solution: u(r) = 2*r*exp(-r)
        u_analytic = 2.0 * r * np.exp(-r)
        plt.plot(r, u_analytic, 'r--', label='Analytic')
        plt.xlim(0, 15)
        plt.title(f"Hydrogen 1s Wavefunction (E={E:.4f})")
        plt.legend()
        plt.show()
    else:
        print("\n❌ FAILURE: Energy is wrong. Check bound.py logic.")

if __name__ == "__main__":
    run_test()