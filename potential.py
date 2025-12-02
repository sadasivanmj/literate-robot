"""
Potential energy functions for confined atoms (SAE model).
OPTIMIZED: 
1. Analytic GASW parameters.
2. LRU Cache to store (A, U) for repeated depths.
3. Fully vectorized potential calculation.
"""
import numpy as np
from functools import lru_cache

# ==============================================================================
# SAE PARAMETERS DATABASE
# ==============================================================================
SAE_PARAMS = {
    'H':  [1.0, 0.000, 0.000, 0.000,   0.000, 0.000, 0.000, 0, 0],
    'He': [1.0, 1.231, 0.662, -1.325,  1.236, -0.231, 0.480, 0, 0],
    'Ne': [1.0, 8.069, 2.148, -3.570,  1.986, 0.931, 0.602, 1, 0],
    'Ar': [1.0, 16.039, 2.007, -25.543, 4.525, 0.961, 0.443, 1, 1],
}

def get_Z_eff(r, species):
    if species not in SAE_PARAMS: return 1.0
    p = SAE_PARAMS[species]
    Z_c, a1, a2, a3, a4, a5, a6 = p[:7]
    term1 = a1 * np.exp(-a2 * r)
    term2 = a3 * r * np.exp(-a4 * r)
    term3 = a5 * np.exp(-a6 * r)
    return Z_c + term1 + term2 + term3

def VGASW_total_debye(r, A, U, mu, species='H', sigma=1.70, r_c=6.7, Delta=2.8):
    """
    Total Potential Calculation (SAE + Debye + GASW).
    Vectorized for maximum performance.
    """
    r = np.asarray(r, dtype=float)
    r_safe = np.clip(r, 1e-14, None)
    
    # 1. Atomic Potential (Screened)
    Z_eff = get_Z_eff(r_safe, species)
    V_atom = -(Z_eff / r_safe) * np.exp(-mu * r_safe)
    
    # 2. Gaussian Potential
    # Pre-calculate constant to save operations
    two_sigma_sq = 2.0 * sigma**2
    V_gauss = A * np.exp(-((r - r_c)**2) / two_sigma_sq)
    
    # 3. Square Well Potential
    V_asw = np.zeros_like(r)
    r_in = r_c - Delta / 2.0
    r_out = r_c + Delta / 2.0
    
    # Fast boolean indexing
    mask = (r >= r_in) & (r <= r_out)
    V_asw[mask] = -U
    
    return V_atom + V_gauss + V_asw

@lru_cache(maxsize=128)
def solve_gasw_parameters(D, sigma=1.70, R_const=-24.5):
    """
    Analytic solution for GASW parameters (A, U).
    Cached: Repeated calls with same 'D' return instantly.
    """
    K = np.sqrt(2.0 * np.pi) * sigma
    
    denominator = 1.0 - (R_const / K)
    
    if abs(denominator) < 1e-12:
        return -3.59, 0.7 
        
    U = D / denominator
    A = U - D
    
    return A, U