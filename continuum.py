"""
Continuum state solver for photoionization using Numba-optimized Numerov method.
Updated for Single Active Electron (SAE) approximation.
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
# JIT-Compiled Core
# ==============================================================================
@njit(fastmath=True, cache=True)
def numerov_loop(u, k_squared, h2_12, N):
    """
    JIT-compiled Numerov recurrence relation.
    Solves u'' + k^2(r)u = 0.
    """
    for n in range(1, N - 1):
        k2_nm1 = k_squared[n - 1]
        k2_n = k_squared[n]
        k2_np1 = k_squared[n + 1]
        
        numerator = (2.0 * (1.0 - 5.0 * h2_12 * k2_n) * u[n] 
                    - (1.0 + h2_12 * k2_nm1) * u[n - 1])
        
        denominator = 1.0 + h2_12 * k2_np1
        
        # Avoid strict division by zero (numerical stability)
        if abs(denominator) < 1e-30: denominator = 1e-30
        
        val = numerator / denominator
        u[n + 1] = val
        
        # Relaxed overflow check for tunneling regions or divergences
        if abs(val) > 1e100: return False
            
    return True

# ==============================================================================
# Solver Implementation
# ==============================================================================
def solve_continuum(E, ell, species, A, U, mu, R_max=120.0, N=6000):
    """
    Solves the radial Schrödinger equation for positive energy E.
    """
    if E <= 0.0: raise ValueError(f"Continuum energy must be positive, got {E}")
    
    # Setup Grid
    r = np.linspace(1e-4, R_max, N)
    h = r[1] - r[0]
    h2_12 = h * h / 12.0
    
    # Calculate Potential
    # Pass 'species' to get the correct Z_eff(r) for the atom
    r_safe = np.clip(r, 1e-20, None)
    V_total = VGASW_total_debye(r, A, U, mu, species=species)
    
    # Effective potential with centrifugal barrier
    V_eff = V_total + ell * (ell + 1) / (2.0 * r_safe**2)
    
    # k^2(r) = 2(E - V_eff)
    k_squared = 2.0 * (E - V_eff)
    
    # Initial Conditions (small r behavior ~ r^(l+1))
    u = np.zeros_like(r)
    scale = 1e-10 # Arbitrary small start
    u[0] = scale * r[0]**(ell + 1)
    u[1] = scale * r[1]**(ell + 1)
    
    # Run Solver
    if not numerov_loop(u, k_squared, h2_12, N):
        raise RuntimeError(f"Numerov overflow at E={E} for species {species}")

    return r, u

def compute_continuum_state(E_pe, ell_cont, species, A, U, mu):
    """
    High-level wrapper with Wavelength-Adaptive Grid.
    
    Parameters:
        E_pe     : Photoelectron energy (a.u.)
        ell_cont : Continuum angular momentum (l_final)
        species  : Atomic species ('H', 'He', 'Ne', 'Ar')
        A, U     : Confinement parameters
        mu       : Debye screening parameter
        
    Returns:
        r_cont   : Radial grid
        u_norm   : Normalized wavefunction
        diag     : Normalization diagnostics
    """
    # 1. Wavelength-Adaptive Grid Logic
    # Low energy = long wavelength = needs larger R_max to capture oscillations
    if E_pe < 1e-5:
        R_max_adaptive = 25000.0
        N_adaptive = 100000
    elif E_pe < 1e-4:
        R_max_adaptive = 10000.0
        N_adaptive = 50000
    elif E_pe < 1e-2:
        R_max_adaptive = 2000.0
        N_adaptive = 15000
    elif E_pe < 0.1:
        R_max_adaptive = 500.0
        N_adaptive = 8000
    else:
        # Standard regime
        R_max_adaptive = 200.0
        N_adaptive = 6000

    # 2. Solve Radial Equation
    r_cont, u_raw = solve_continuum(E_pe, ell_cont, species, A, U, mu, 
                                    R_max=R_max_adaptive, N=N_adaptive)
    
    # 3. Determine Asymptotic Charge (Z_c)
    # For neutral atoms in SAE approximation, Z_c is typically 1.0.
    # The potential goes to -1/r at large distances.
    if species in SAE_PARAMS:
        Z_asy = SAE_PARAMS[species][0]
    else:
        Z_asy = 1.0

    # 4. Normalize
    diag = {}
    
    # Check for Debye screening (short-range potential)
    if abs(mu) > 1e-10:
        # Case A: Debye Screened -> Use Envelope Normalization
        # (Coulomb functions don't apply because potential decays exponentially)
        u_norm, info = energy_normalize_continuum(r_cont, u_raw, E_pe, ell=ell_cont, A=A, U=U, mu=mu)
        diag['type'] = 'envelope'
        
    else:
        # Case B: Coulombic Asymptote (-1/r)
        # Check if it's pure Free Hydrogen (no cage, no screening)
        is_free_hydrogen = (species == 'H' and abs(A) < 1e-12 and abs(U) < 1e-12)
        
        if is_free_hydrogen:
            # Use analytical Coulomb matching (often more stable for pure H)
            u_norm, info = normalize_continuum_coulomb_free(r_cont, u_raw, E_pe, ell_cont, Z=Z_asy)
            diag['type'] = 'coulomb_free'
        else:
            # Use Phase Shift Fitting (Standard for Ar, Ne, or confined H)
            # Fits u(r) ~ sin(kr - l*pi/2 + sigma + delta)
            u_norm, info = normalize_continuum_coulomb_phase(r_cont, u_raw, E_pe, ell_cont, Z=Z_asy, n_fit=None)
            diag['type'] = 'coulomb_phase'
            
    diag.update(info)
    return r_cont, u_norm, diag