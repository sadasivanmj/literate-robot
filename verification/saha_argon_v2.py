"""
Reproduce Figure 10(b) from Saha et al. (2019).
System: Ar@C60 using GASW Potential.
Quantity: Absolute square of dipole matrix element |d_if|^2.

Physics:
    |d_if|^2 = (1/4)|<3p|r|es>|^2 + |<3p|r|ed>|^2
    Cooper Minimum analysis included.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from potential import VGASW_total_debye, solve_gasw_parameters, SAE_PARAMS
from bound import solve_ground_u
from cross_section import compute_cross_section_spectrum

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SPECIES = 'Ar'
L_INITIAL = 1  # 3p state
Depths_to_Plot = [0.0, 0.30, 0.46, 0.56] # Depths in a.u. (Fig 10b)
Colors = ['black', 'red', 'lime', 'blue']
Labels = ['Free', 
          r'$V_{GASW}(r_c)=0.30$ a.u.', 
          r'$V_{GASW}(r_c)=0.46$ a.u.', 
          r'$V_{GASW}(r_c)=0.56$ a.u.']

# Energy Grid (Photoelectron Energy in a.u.)
# Dense grid near Cooper minimum (~1.0 a.u.) for accurate detection
E_grid = np.concatenate([
    np.linspace(0.1, 0.7, 30),
    np.linspace(0.71, 1.5, 100),  # Dense region for Cooper Min
    np.linspace(1.51, 3.0, 50)
])

def run_simulation():
    print("="*70)
    print(f"Reproducing Saha Fig 10(b): Ar@C60 Dipole Matrix Elements")
    print("="*70)

    results = {}
    
    # ------------------------------------------------------------------
    # LOOP OVER DEPTHS
    # ------------------------------------------------------------------
    for depth, label in zip(Depths_to_Plot, Labels):
        print(f"\n[Simulation] Depth = {depth:.2f} a.u. ({label})")
        
        # 1. Potential Parameters
        if abs(depth) < 1e-6:
            # Free Atom
            A_val, U_val = 0.0, 0.0
        else:
            # Confined: Solve for A, U using Saha constraints
            A_val, U_val = solve_gasw_parameters(depth)
            print(f"  Solved Pot: A={A_val:.4f}, U={U_val:.4f}")

        # 2. Bound State (Ar 3p)
        # Note: Saha parameters for Ar l=1 are strictly defined in potential.py
        r_bound, u_bound, E_bound, _, _ = solve_ground_u(
            VGASW_total_debye, 
            species=SPECIES, 
            R_max=60.0, N=6000, 
            A=A_val, U=U_val
        )
        print(f"  Bound Energy (3p): {E_bound:.4f} a.u.")

        # 3. Continuum & Dipole Calculation (Parallel)
        # We use compute_cross_section_spectrum to leverage the parallel engine,
        # but we extract the raw dipole elements 'details' instead of just sigma.
        sigma_dummy, details_list = compute_cross_section_spectrum(
            E_grid, r_bound, u_bound, E_bound,
            l_initial=L_INITIAL, species=SPECIES,
            A=A_val, U=U_val, mu=0.0,
            n_workers=None, # Auto-detect
            method='standard' # We will manually weight the channels below
        )

        # 4. Process Results: Construct |d_if|^2
        # Formula: |d_if|^2 = 1/4 * |Ds|^2 + |Dd|^2
        dipole_sq_total = []
        
        for i, det in enumerate(details_list):
            if 'error' in det:
                dipole_sq_total.append(0.0)
                continue
                
            # Extract raw matrix elements
            D_s = det.get('D_l0', 0.0) # l=0 (s-wave)
            D_d = det.get('D_l2', 0.0) # l=2 (d-wave)
            
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
    print(f"{'Case':<25} {'Min Location (a.u.)':<20} {'Literature (approx)':<20}")
    print("-" * 70)
    
    for depth in Depths_to_Plot:
        y_vals = results[depth]
        # Find local minimum in the relevant range (0.5 - 1.5 a.u.)
        # We simply find the index of the minimum value in that window
        mask = (E_grid > 0.5) & (E_grid < 1.5)
        
        if np.any(mask):
            sub_grid = E_grid[mask]
            sub_y = y_vals[mask]
            min_idx = np.argmin(sub_y)
            e_min = sub_grid[min_idx]
            
            lit_val = "0.92 (Free)" if depth == 0.0 else ""
            if abs(depth - 0.56) < 0.01: lit_val = "~0.98 (GASW)"
            
            print(f"Depth {depth:<20.2f} {e_min:<20.3f} {lit_val:<20}")
        else:
             print(f"Depth {depth:<20.2f} {'Not found':<20} {'---':<20}")

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 6), dpi=120)
    
    for depth, color, label in zip(Depths_to_Plot, Colors, Labels):
        plt.plot(E_grid, results[depth], color=color, linewidth=2, label=label)

    # Styling to match Saha Fig 10(b)
    plt.xlim(0.3, 3.0)
    plt.ylim(0.01, 0.035) # Matches Fig 10(b) Y-axis range
    
    # Ticks inside
    plt.tick_params(direction='in', which='both', top=True, right=True)
    plt.minorticks_on()
    
    plt.xlabel('Photoelectron energy (a.u.)', fontsize=12, fontweight='bold')
    plt.ylabel(r'$|d_{if}|^2$', fontsize=12, fontweight='bold')
    
    # Legend
    plt.legend(frameon=False, loc='upper right', fontsize=10)
    
    # Inset Text matching style
    plt.text(0.5, 0.032, r'(b)   Ar@C$_{60}$-GASW', 
             fontsize=12, fontweight='bold', 
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

    plt.tight_layout()
    plt.savefig('Saha_Fig10b_Reproduced.png')
    plt.show()

if __name__ == "__main__":
    run_simulation()