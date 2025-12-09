"""
Continuum state solver using Numba-optimized Numerov method.
Features:
- Wavelength-adaptive grid for low-energy accuracy.
- Connection to GASW potential.
"""
import numpy as np
from numba import njit
from potential import VGASW_total_debye, SAE_PARAMS
from normalization import (
    normalize_continuum_coulomb_free,
    normalize_continuum_coulomb_phase,
    energy_normalize_continuum,
)

# ==============================================================================
# JIT KERNEL
# ==============================================================================
@njit(fastmath=True, cache=True)
def numerov_loop(u, k_squared, h2_12, N):
    for n in range(1, N - 1):
        k2_nm1 = k_squared[n - 1]
        k2_n = k_squared[n]
        k2_np1 = k_squared[n + 1]
        
        numerator = (2.0 * (1.0 - 5.0 * h2_12 * k2_n) * u[n] 
                    - (1.0 + h2_12 * k2_nm1) * u[n - 1])
        
        denominator = 1.0 + h2_12 * k2_np1
        if abs(denominator) < 1e-30: denominator = 1e-30
        
        val = numerator / denominator
        u[n + 1] = val
        
        if abs(val) > 1e100: return False # Overflow guard
    return True

# ==============================================================================
# SOLVER LOGIC
# ==============================================================================

def solve_continuum(E, ell, species, A, U, mu, R_max=120.0, N=6000, **kwargs):
    if E <= 0.0: raise ValueError(f"E must be positive, got {E}")
    
    # 1. Grid
    r = np.linspace(1e-4, R_max, N)
    h = r[1] - r[0]
    h2_12 = h * h / 12.0
    
    # 2. Potential
    # Note: We pass 'ell' to VGASW to handle the GSZ switch logic
    V_total = VGASW_total_debye(r, A=A, U=U, mu=mu, species=species, l_wave=ell, **kwargs)
    
    r_safe = np.clip(r, 1e-20, None)
    V_eff = V_total + ell * (ell + 1) / (2.0 * r_safe**2)
    k_squared = 2.0 * (E - V_eff)
    
    # 3. Integrate
    u = np.zeros_like(r)
    # Power-series start near origin
    scale = 1e-10 
    u[0] = scale * r[0]**(ell + 1)
    u[1] = scale * r[1]**(ell + 1)
    
    if not numerov_loop(u, k_squared, h2_12, N):
        raise RuntimeError(f"Numerov overflow at E={E}")

    return r, u

def compute_continuum_state(E_pe, ell_cont, species, A, U, mu, **kwargs):
    """
    High-level wrapper with Adaptive Grid and Normalization.
    """
    # 1. Adaptive Grid Sizing (Crucial for Wigner threshold)
    # Scaling based on de Broglie wavelength
    if E_pe < 1e-5:   R_max, N = 25000.0, 100000
    elif E_pe < 1e-4: R_max, N = 10000.0, 50000
    elif E_pe < 1e-2: R_max, N = 2000.0, 15000
    elif E_pe < 0.1:  R_max, N = 500.0, 8000
    else:             R_max, N = 200.0, 6000

    r_cont, u_raw = solve_continuum(E_pe, ell_cont, species, A, U, mu, 
                                    R_max=R_max, N=N, **kwargs)
    
    # 2. Identify Asymptotic Charge (Z_eff at infinity)
    Z_asy = 1.0
    if species in SAE_PARAMS:
        p = SAE_PARAMS[species].get('default', list(SAE_PARAMS[species].values())[0])
        Z_asy = p.Z_c

    # 3. Normalize
    diag = {}
    if abs(mu) < 1e-10:
        # Coulomb Boundary Conditions (Pure or Phase-Shifted)
        if abs(A) < 1e-12 and abs(U) < 1e-12 and species == 'H':
            # Pure H case
            u_norm, info = normalize_continuum_coulomb_free(r_cont, u_raw, E_pe, ell_cont, Z=Z_asy)
        else:
            # Short-range potential present (Cage or Non-H Atom)
            u_norm, info = normalize_continuum_coulomb_phase(r_cont, u_raw, E_pe, ell_cont, Z=Z_asy)
            
        diag.update(info)
    else:
        # Debye Screened (Exponential decay -> Plane wave at infinity)
        u_norm, info = energy_normalize_continuum(r_cont, u_raw, E_pe, ell=ell_cont, A=A, U=U, mu=mu)
        diag.update(info)
        
    return r_cont, u_norm, diag