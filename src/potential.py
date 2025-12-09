"""
Potential energy functions for confined atoms (SAE model).
Implements the specific GASW model from Saha et al. (2019).
"""
import numpy as np
from functools import lru_cache
from typing import NamedTuple, Dict, Union, Tuple

# ==============================================================================
# DATA STRUCTURES & PARAMETERS
# ==============================================================================

class TongLinParams(NamedTuple):
    Z_c: float
    a1: float; a2: float
    a3: float; a4: float
    a5: float; a6: float
    ground_l: int
    ground_n_idx: int

# Green-Sellin-Zachor (GSZ) Parameters [Garvey et al. 1975]
# Format: [Z, N_core, H, d]
GSZ_PARAMS = {
    'Ar': [18.0, 17.0, 3.492, 0.256], 
    'Ne': [10.0, 9.0,  2.164, 0.166], 
}

# Tong & Lin SAE Parameters [Tong & Lin 2005]
SAE_PARAMS: Dict[str, Dict[Union[int, str], TongLinParams]] = {
    'H': {
        'default': TongLinParams(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    },
    'He': {
        0:         TongLinParams(1.0, 1.231, 0.662, -1.325, 1.236, -0.231, 0.480, 0, 0),
        'default': TongLinParams(1.0, 1.231, 0.662, -1.325, 1.236, -0.231, 0.480, 0, 0)
    },
    'Ne': {
        1:         TongLinParams(1.0, 8.069, 2.148, -3.570, 1.986, 0.931, 0.602, 1, 1),
        'default': TongLinParams(1.0, 8.069, 2.148, -3.570, 1.986, 0.931, 0.602, 1, 1)
    },
    'Ar': {
        1:         TongLinParams(1.0, 16.039, 2.007, -25.543, 4.525, 0.961, 0.443, 1, 1),
        'default': TongLinParams(1.0, 16.039, 2.007, -25.543, 4.525, 0.961, 0.443, 1, 1)
    }
}

# ==============================================================================
# ATOMIC POTENTIALS
# ==============================================================================

def get_Z_eff_TongLin(r, p: TongLinParams):
    return (p.Z_c + 
            p.a1 * np.exp(-p.a2 * r) + 
            p.a3 * r * np.exp(-p.a4 * r) + 
            p.a5 * np.exp(-p.a6 * r))

def get_Z_eff_GSZ(r, species):
    if species not in GSZ_PARAMS: return 1.0
    Z_nuc, N_core, H, d = GSZ_PARAMS[species]
    val = r/d
    mask = val < 50.0
    omega = np.zeros_like(r)
    omega[mask] = 1.0 / (H * (np.exp(val[mask]) - 1.0) + 1.0)
    return 1.0 + N_core * omega

# ==============================================================================
# CONFINEMENT & TOTAL POTENTIAL
# ==============================================================================

def VGASW_total_debye(r, A=0.0, U=0.0, mu=0.0, species='H', r_c=6.7, Delta=1.70, l_wave=0, **kwargs):
    """
    Total Potential = Atom + Confinement (Gaussian + Square Well).
    
    Physics:
    1. Atom: SAE model (Tong-Lin or GSZ)
    2. Confinement: V_GASW = (A * Gaussian) + V_ASW(-U)
    
    Note on scaling:
      - 'A' here acts as the pre-factor. If solving from Saha parameters, 
        ensure A includes the 1/sqrt(2pi) factor.
      - 'Delta' here is the Gaussian width sigma. The code applies the sqrt(7) 
        scaling internally if you pass sigma, or you can pass the full width.
        Saha Eq(3): exp(- ((r-rc) / (sqrt(7)*sigma))^2 )
    """
    r = np.asarray(r, dtype=float)
    r_safe = np.clip(r, 1e-14, None)
    
    # --- 1. Atomic Potential ---
    # Hybrid logic: GSZ for noble gas continuum (l != 1), Tong-Lin otherwise
    use_gsz = False
    if species in ['Ar', 'Ne'] and l_wave != 1:
        use_gsz = True
        
    if use_gsz and species in GSZ_PARAMS:
        Z_eff = get_Z_eff_GSZ(r_safe, species)
    else:
        if species in SAE_PARAMS:
            elem = SAE_PARAMS[species]
            p = elem.get(l_wave, elem.get('default'))
            Z_eff = get_Z_eff_TongLin(r_safe, p)
        else:
            Z_eff = 1.0 # Pure Coulomb fallback

    V_atom = -(Z_eff / r_safe) * np.exp(-mu * r_safe)
    
    # --- 2. Confinement Potential ---
    V_conf = np.zeros_like(r)
    
    # A. Gaussian Component (Saha Eq 3)
    # The paper uses width = sqrt(7)*sigma. 
    # We assume 'Delta' passed in is just 'sigma' (1.70), so we apply sqrt(7).
    effective_width = np.sqrt(7) * Delta
    
    if abs(A) > 1e-12:
        V_conf += A * np.exp(-((r - r_c)**2) / (effective_width**2))

    # B. ASW Component (Saha Eq 1 & 3)
    # V_ASW is -U inside the shell, 0 outside.
    Delta_ASW = kwargs.get('Delta_ASW', 2.8)
    if abs(U) > 1e-12:
        r_inner = r_c - Delta_ASW / 2.0
        r_outer = r_c + Delta_ASW / 2.0
        mask = (r >= r_inner) & (r <= r_outer)
        V_conf[mask] -= U 

    return V_atom + V_conf

# ==============================================================================
# PARAMETER SOLVER
# ==============================================================================

@lru_cache(maxsize=128)
def solve_gasw_parameters(Target_Depth_Au: float) -> Tuple[float, float]:
    """
    Linear solver for (A, U) to match Saha's shape constraints.
    
    Returns:
        (A_peak, U_val)
        A_peak: The value to pass as 'A' to VGASW (includes 1/sqrt(2pi))
    """
    # Reference values from Saha et al. (2019)
    # A_paper = -3.59, U_ref = 0.7
    # The paper defines V_gauss = (A_paper / sqrt(2pi)) * exp(...)
    A_saha_param = -3.59
    A_peak_ref = A_saha_param / np.sqrt(2 * np.pi) 
    U_ref = 0.7
    
    # Constant Ratio constraint
    Ratio_Const = A_peak_ref / U_ref  
    
    # Solve linear system:
    # 1. A - U = -Target_Depth
    # 2. A = Ratio * U
    U_new = -Target_Depth_Au / (Ratio_Const - 1.0)
    A_new = U_new * Ratio_Const
    
    return A_new, U_new

if __name__ == "__main__":
    # Self-check
    print("[potential.py] Testing Parameter Solver...")
    A_chk, U_chk = solve_gasw_parameters(0.56)
    print(f"  Target 0.56 -> A={A_chk:.4f}, U={U_chk:.4f}")
    print(f"  Check Depth: -(A-U) = {-(A_chk - U_chk):.4f}")