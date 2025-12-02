# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 08:36:20 2025

@author: harry
"""

"""
FIXED script to reproduce Saha et al. (2019) with proper parameters.
Key fixes:
1. Increased grid resolution (R_max=150, N=10000)
2. SERIAL computation to avoid numerical artifacts
3. Proper asymptotic region for normalization
4. Energy grid with better threshold resolution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from potential import VGASW_total_debye
from bound import solve_ground_u
from continuum import compute_continuum_state
from cross_section import compute_cross_section_spectrum

print("="*80)
print("SAHA ET AL. (2019) - FIXED REPRODUCTION")
print("="*80)

# ============================================================================
# GASW Parameters (EXACTLY as in Saha et al.)
# ============================================================================
A_lit = -3.59
sigma_lit = 1.70
r_c = 6.7
U_lit = 0.7
D_target = 0.56  # Target depth from Table 1
R_const = -24.5

def gasw_equations(x, D):
    A, U = x
    if U <= 0 or A >= 0:
        return [1e10, 1e10]
    eq1 = (A - U) + D
    eq2 = (A * np.sqrt(2*np.pi) * sigma_lit) / U - R_const
    return [eq1, eq2]

factor = 1 - R_const / (np.sqrt(2*np.pi) * sigma_lit)
U_guess = D_target / factor
A_guess = U_guess - D_target

(A_cage, U_cage), *_ = fsolve(
    lambda x: gasw_equations(x, D_target),
    [A_guess, U_guess],
    full_output=True
)

print(f"\nGASW Parameters:")
print(f"  A = {A_cage:.6f} a.u.")
print(f"  U = {U_cage:.6f} a.u.")
print(f"  V_GASW(r_c) = {A_cage - U_cage:.6f} a.u.")

# ============================================================================
# FIX #1: HIGH-RESOLUTION BOUND STATES
# ============================================================================
print("\n" + "="*80)
print("FIX #1: Computing HIGH-RESOLUTION bound states")
print("="*80)

# Free hydrogen - higher resolution
print("\nFree H:")
r_bound_free, u_bound_free, E_bound_free, _, norm_free = solve_ground_u(
    VGASW_total_debye,
    R_max=100.0,  # Increased from 60
    N=8000,       # Increased from 6000
    ell=0,
    A=0.0, U=0.0, mu=0.0
)
print(f"  E_1s = {E_bound_free:.8f} a.u. (expected: -0.5000000)")
print(f"  Error = {abs(E_bound_free + 0.5):.2e}")
print(f"  Norm = {norm_free:.8f}")

# Confined hydrogen - higher resolution
print("\nConfined H (GASW):")
r_bound_conf, u_bound_conf, E_bound_conf, _, norm_conf = solve_ground_u(
    VGASW_total_debye,
    R_max=150.0,  # Increased for better confinement representation
    N=10000,      # Higher resolution
    ell=0,
    A=A_cage, U=U_cage, mu=0.0
)
print(f"  E_1s = {E_bound_conf:.8f} a.u.")
print(f"  Norm = {norm_conf:.8f}")
print(f"  ΔE = {(E_bound_conf - E_bound_free)*27.2114:.3f} eV (confinement shift)")

# ============================================================================
# FIX #2: MODIFIED continuum.py with better grid
# ============================================================================
print("\n" + "="*80)
print("FIX #2: Using optimized continuum grid")
print("="*80)

# We need to modify the default R_max and N in compute_continuum_state
# Create a wrapper that forces higher resolution
def compute_continuum_high_res(E_pe, ell_cont, A, U, mu):
    """High-resolution continuum solver for accurate cross sections"""
    from continuum import solve_continuum
    from normalization import (normalize_continuum_coulomb_free,
                              normalize_continuum_coulomb_phase)
    
    # CRITICAL: Use larger grid for continuum
    R_max_cont = 200.0 if E_pe < 0.1 else 150.0  # Larger for low energy
    N_cont = 12000  # High resolution
    
    r_cont, u_raw = solve_continuum(E_pe, ell_cont, A, U, mu,
                                    R_max=R_max_cont, N=N_cont)
    
    # Choose normalization
    if abs(mu) < 1e-10:
        if abs(A) < 1e-12 and abs(U) < 1e-12:
            # Pure Coulomb
            u_norm, diag = normalize_continuum_coulomb_free(
                r_cont, u_raw, E_pe, ell_cont, Z=1.0
            )
            diag['normalization_type'] = 'coulomb_free'
        else:
            # Coulomb + GASW
            u_norm, diag = normalize_continuum_coulomb_phase(
                r_cont, u_raw, E_pe, ell_cont, Z=1.0
            )
            diag['normalization_type'] = 'coulomb_phase'
    
    return r_cont, u_norm, diag

# ============================================================================
# FIX #3: ENERGY GRID WITH BETTER THRESHOLD RESOLUTION
# ============================================================================
print("\n" + "="*80)
print("FIX #3: Optimized energy grid")
print("="*80)

# Use log spacing near threshold, linear at higher energies
E_threshold = np.logspace(-4, -1, 30)  # 0.0001 to 0.1 (log spacing)
E_mid = np.linspace(0.12, 1.0, 45)     # 0.1 to 1.0 (linear)
E_high = np.linspace(1.05, 2.0, 25)    # 1.0 to 2.0 (linear)
E_PE = np.concatenate([E_threshold, E_mid, E_high])

print(f"Energy grid:")
print(f"  Total points: {len(E_PE)}")
print(f"  Range: [{E_PE.min():.4f}, {E_PE.max():.2f}] a.u.")
print(f"  Threshold region: {len(E_threshold)} points (log-spaced)")

# ============================================================================
# FIX #4: SERIAL COMPUTATION ONLY
# ============================================================================
print("\n" + "="*80)
print("FIX #4: Computing cross sections (SERIAL, high-res)")
print("="*80)

ell_cont = 1  # 1s → εp

# Method 1: Use modified high-res continuum
from cross_section import dipole_matrix_element, photoionization_cross_section
from tqdm import tqdm

print("\n1. FREE HYDROGEN (high-resolution)")
sigma_free = []
D_free_list = []

for E_pe in tqdm(E_PE, desc="Free H"):
    r_c, u_c, diag = compute_continuum_high_res(E_pe, ell_cont, 0.0, 0.0, 0.0)
    D, _ = dipole_matrix_element(r_c, u_c, r_bound_free, u_bound_free)
    sigma = photoionization_cross_section(E_pe, E_bound_free, D)
    sigma_free.append(sigma)
    D_free_list.append(D)

sigma_free = np.array(sigma_free)

print(f"\n  At threshold (E_PE = {E_PE[0]:.4f}):")
print(f"    σ = {sigma_free[0]:.6f} a.u.")
print(f"    Literature: σ = 0.22225 a.u.")
print(f"    Ratio: {sigma_free[0]/0.22225:.4f}")

print("\n2. CONFINED HYDROGEN (high-resolution)")
sigma_conf = []
D_conf_list = []

for E_pe in tqdm(E_PE, desc="Confined H"):
    r_c, u_c, diag = compute_continuum_high_res(E_pe, ell_cont, 
                                                 A_cage, U_cage, 0.0)
    D, _ = dipole_matrix_element(r_c, u_c, r_bound_conf, u_bound_conf)
    sigma = photoionization_cross_section(E_pe, E_bound_conf, D)
    sigma_conf.append(sigma)
    D_conf_list.append(D)

sigma_conf = np.array(sigma_conf)

print(f"\n  At threshold (E_PE = {E_PE[0]:.4f}):")
print(f"    σ = {sigma_conf[0]:.6f} a.u.")
print(f"    Literature: σ ≈ 0.15 a.u.")
print(f"    Ratio: {sigma_conf[0]/0.15:.4f}")

# ============================================================================
# COMPARISON WITH SAHA ET AL.
# ============================================================================
print("\n" + "="*80)
print("COMPARISON WITH SAHA ET AL.")
print("="*80)

print(f"\nThreshold values (E_PE = {E_PE[0]:.4f}):")
print(f"  Free H:      {sigma_free[0]:.5f} vs 0.22225 (lit) → "
      f"error = {100*abs(sigma_free[0]-0.22225)/0.22225:.1f}%")
print(f"  Confined H:  {sigma_conf[0]:.5f} vs ~0.15 (lit) → "
      f"error = {100*abs(sigma_conf[0]-0.15)/0.15:.1f}%")

# Check shape properties
print(f"\nShape analysis:")
# Free H should be smooth and decreasing
dσ_free = np.diff(sigma_free)
is_monotonic = np.sum(dσ_free > 0) < len(dσ_free) * 0.1  # Allow <10% violations
print(f"  Free H monotonically decreasing: {is_monotonic}")

# Confined H should show oscillations
sign_changes = np.sum(np.diff(np.sign(dσ_free)) != 0)
print(f"  Free H oscillations: {sign_changes} (should be ~0-2)")

dσ_conf = np.diff(sigma_conf)
sign_changes_conf = np.sum(np.diff(np.sign(dσ_conf)) != 0)
print(f"  Confined H oscillations: {sign_changes_conf} (should be >5)")

# ============================================================================
# PLOTS (Reproduce Saha et al. Figure 4 style)
# ============================================================================
print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Full range (like Saha Fig 4c)
ax1 = axes[0, 0]
ax1.plot(E_PE, sigma_free, 'k-', lw=2.5, label='Free H', alpha=0.8)
ax1.plot(E_PE, sigma_conf, 'b-', lw=2.5, 
         label=f'Confined H (V={D_target:.2f})', alpha=0.8)
ax1.axhline(0.22225, color='gray', ls='--', alpha=0.5, label='Free lit.')
ax1.axhline(0.15, color='lightblue', ls='--', alpha=0.5, label='Conf lit.')
ax1.set_xlabel('Photoelectron Energy $E_{PE}$ (a.u.)', fontsize=12)
ax1.set_ylabel('Cross Section σ (a.u.)', fontsize=12)
ax1.set_title('Photoionization: 1s → εp', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_xlim([0, 2])
ax1.set_ylim([0, 0.3])

# Panel 2: Near threshold zoom
ax2 = axes[0, 1]
mask = E_PE <= 0.5
ax2.plot(E_PE[mask], sigma_free[mask], 'k-', lw=2.5, label='Free H')
ax2.plot(E_PE[mask], sigma_conf[mask], 'b-', lw=2.5, label='Confined H')
ax2.plot(E_PE[0], sigma_free[0], 'ko', ms=8, 
         label=f'Our: {sigma_free[0]:.4f}')
ax2.plot(E_PE[0], sigma_conf[0], 'bo', ms=8,
         label=f'Our: {sigma_conf[0]:.4f}')
ax2.axhline(0.22225, color='gray', ls='--', alpha=0.7)
ax2.axhline(0.15, color='lightblue', ls='--', alpha=0.7)
ax2.set_xlabel('$E_{PE}$ (a.u.)', fontsize=12)
ax2.set_ylabel('σ (a.u.)', fontsize=12)
ax2.set_title('Near Threshold (Zoom)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_xlim([0, 0.5])
ax2.set_ylim([0.05, 0.25])

# Panel 3: Ratio (confinement effect)
ax3 = axes[1, 0]
ratio = sigma_conf / sigma_free
ax3.plot(E_PE, ratio, 'r-', lw=2.5, alpha=0.8)
ax3.axhline(1.0, color='k', ls='--', alpha=0.5, label='No confinement')
ax3.axhline(0.15/0.22225, color='gray', ls=':', alpha=0.7,
            label=f'Lit ratio: {0.15/0.22225:.3f}')
ax3.set_xlabel('$E_{PE}$ (a.u.)', fontsize=12)
ax3.set_ylabel('$σ_{confined} / σ_{free}$', fontsize=12)
ax3.set_title('Confinement Effect', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)
ax3.set_xlim([0, 2])

# Panel 4: Log-log (check threshold scaling)
ax4 = axes[1, 1]
ax4.loglog(E_PE, sigma_free, 'k-', lw=2, label='Free H')
ax4.loglog(E_PE, sigma_conf, 'b-', lw=2, label='Confined H')
ax4.set_xlabel('$E_{PE}$ (a.u.)', fontsize=12)
ax4.set_ylabel('σ (a.u.)', fontsize=12)
ax4.set_title('Log-Log: Threshold Scaling', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('saha_fixed_reproduction.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: saha_fixed_reproduction.png")

# ============================================================================
# FINAL ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("FINAL ASSESSMENT")
print("="*80)

threshold_error_free = abs(sigma_free[0] - 0.22225) / 0.22225 * 100
threshold_error_conf = abs(sigma_conf[0] - 0.15) / 0.15 * 100

print(f"\n✓ Threshold accuracy:")
print(f"  Free H:     {threshold_error_free:.1f}% error")
print(f"  Confined H: {threshold_error_conf:.1f}% error")

if threshold_error_free < 5 and threshold_error_conf < 15:
    print("\n✓ SUCCESS: Threshold values match Saha et al. within acceptable error")
else:
    print("\n⚠ WARNING: Threshold values differ significantly")

if is_monotonic and sign_changes_conf > 3:
    print("✓ SUCCESS: Shape matches Saha et al. expectations")
    print("  - Free H: smooth, decreasing")
    print("  - Confined H: oscillatory (confinement resonances)")
else:
    print("⚠ Shape issues detected - see diagnostic output above")

print("\nIf shape still doesn't match:")
print("  1. Check that your normalization.py matches the fixed version")
print("  2. Verify Numerov has k² = 2(E - V_eff) not 2(V_eff - E)")
print("  3. Try even larger R_max (200-250 a.u.) for E_PE < 0.01")
print("  4. Ensure asymptotic region starts at r > 50 a.u.")

print("\n" + "="*80)
plt.show()`             