"""
Reproduce Figure 10(b) from Saha et al. (2019).
System: Ar@C60 using GASW Potential.
Quantity: Absolute square of dipole matrix element |d_if|^2.
"""
import numpy as np
import matplotlib.pyplot as plt
from potential import VGASW_total_debye, solve_gasw_parameters
from bound import solve_ground_u
from cross_section import compute_cross_section_spectrum

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SPECIES = 'Ar'
L_INITIAL = 1  # 3p state (l=1)
Depths_to_Plot = [0.0, 0.30, 0.46, 0.56] # Depths in a.u. (Fig 10b)
Colors = ['black', 'red', 'lime', 'blue']
Labels = ['Free', 
          r'$V_{GASW}(r_c)=0.30$ a.u.', 
          r'$V_{GASW}(r_c)=0.46$ a.u.', 
          r'$V_{GASW}(r_c)=0.56$ a.u.']

# Energy Grid: Dense near Cooper Min (~1.0 a.u.)
E_grid = np.concatenate([
    np.linspace(0.1, 0.7, 20),
    np.linspace(0.71, 1.5, 80),  # Dense region
    np.linspace(1.51, 3.0, 40)
])

def run_simulation():
    print("="*70)
    print(f"Reproducing Saha Fig 10(b): Ar@C60 Dipole Matrix Elements")
    print("="*70)

    results = {}
    
    for depth, label in zip(Depths_to_Plot, Labels):
        print(f"\n[Simulation] Depth = {depth:.2f} a.u. ({label})")
        
        # 1. Potential Parameters
        if abs(depth) < 1e-6:
            A_val, U_val = 0.0, 0.0
        else:
            A_val, U_val = solve_gasw_parameters(depth)
            print(f"  Solved Pot: A={A_val:.4f}, U={U_val:.4f}")

        # 2. Bound State (Ar 3p)
        # CRITICAL: N=20000 to resolve deep Argon core
        r_bound, u_bound, E_bound, _, _ = solve_ground_u(
            VGASW_total_debye, 
            species=SPECIES, 
            R_max=60.0, 
            N=20000,   # <--- FIX: Increased resolution
            A=A_val, U=U_val
        )
        print(f"  Bound Energy (3p): {E_bound:.4f} a.u.")
        
        # Sanity Check for Bound State
        if abs(E_bound) < 0.1:
            print("  ❌ ERROR: Bound state not found (E ~ 0). Skipping...")
            results[depth] = np.zeros_like(E_grid)
            continue

        # 3. Continuum & Dipole Calculation (Parallel)
        # Uses method='standard' but we construct |d_if|^2 manually below
        sigma_dummy, details_list = compute_cross_section_spectrum(
            E_grid, r_bound, u_bound, E_bound,
            l_initial=L_INITIAL, species=SPECIES,
            A=A_val, U=U_val, mu=0.0,
            n_workers=None,
            method='standard' 
        )

        # 4. Process Results: |d_if|^2 = 1/4 * |Ds|^2 + |Dd|^2
        dipole_sq_total = []
        for det in details_list:
            if 'error' in det:
                dipole_sq_total.append(0.0)
            else:
                D_s = det.get('D_l0', 0.0) # 3p -> es
                D_d = det.get('D_l2', 0.0) # 3p -> ed
                # Saha Eq (11) Weighting
                val = 0.25 * (abs(D_s)**2) + (abs(D_d)**2)
                dipole_sq_total.append(val)
            
        results[depth] = np.array(dipole_sq_total)

    # ------------------------------------------------------------------
    # COOPER MINIMUM ANALYSIS
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("COOPER MINIMUM ANALYSIS")
    print("="*70)
    print(f"{'Case':<25} {'Min Loc (a.u.)':<15} {'Ref (Saha)':<15}")
    print("-" * 70)
    
    for depth in Depths_to_Plot:
        y_vals = results[depth]
        # Search window for Cooper Min: 0.5 - 1.5 a.u.
        mask = (E_grid > 0.5) & (E_grid < 1.5)
        
        if np.any(mask) and np.any(y_vals[mask] > 0):
            sub_grid = E_grid[mask]
            sub_y = y_vals[mask]
            min_idx = np.argmin(sub_y)
            e_min = sub_grid[min_idx]
            
            ref = ""
            if depth == 0.0: ref = "0.92"
            elif abs(depth - 0.56) < 0.01: ref = "0.98"
            
            print(f"Depth {depth:<20.2f} {e_min:<15.3f} {ref:<15}")
        else:
             print(f"Depth {depth:<20.2f} {'---':<15} {'---':<15}")

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 6), dpi=120)
    
    for depth, color, label in zip(Depths_to_Plot, Colors, Labels):
        plt.plot(E_grid, results[depth], color=color, linewidth=2, label=label)

    plt.xlim(0.3, 3.0)
    plt.ylim(0.01, 0.035) 
    plt.tick_params(direction='in', which='both', top=True, right=True)
    plt.minorticks_on()
    plt.xlabel('Photoelectron energy (a.u.)', fontsize=12, fontweight='bold')
    plt.ylabel(r'$|d_{if}|^2$', fontsize=12, fontweight='bold')
    plt.legend(frameon=False, loc='upper right', fontsize=10)
    plt.title(f"Ar@C60 Dipole Matrix Elements (Reproduced)", fontsize=11)
    
    plt.tight_layout()
    plt.savefig('Saha_Fig10b_Reproduced.png')
    print("\n✅ Plot saved to Saha_Fig10b_Reproduced.png")
    plt.show()

if __name__ == "__main__":
    run_simulation()