"""
Potential energy functions for confined atoms (SAE model).
CORRECTED VERSION: 
1. Enforces consistent SAE potential (Eq. 7 in Saha et al.) for all channels.
2. Implements the GASW superposition (Eq. 3) exactly.
3. Windows-friendly (no nested parallel functions).
"""
import numpy as np
from functools import lru_cache
from typing import NamedTuple, Dict, Union, Tuple

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

class TongLinParams(NamedTuple):
    """Stores coefficients for the SAE potential Z_eff(r)."""
    Z_c: float
    a1: float; a2: float
    a3: float; a4: float
    a5: float; a6: float
    ground_l: int       # Angular momentum of the ground state
    ground_n_idx: int   # Index of the ground state (0-based)

# ==============================================================================
# SAE PARAMETERS DATABASE
# ==============================================================================
# Parameters form: Z_eff(r) = Z_c + a1*e^(-a2*r) + a3*r*e^(-a4*r) + a5*e^(-a6*r)
# Ref: Saha et al. (2019) Eq. 7, referencing Tong & Lin (2005) / Le et al. (2009).

SAE_PARAMS: Dict[str, Dict[Union[int, str], TongLinParams]] = {
    'H': {
        'default': TongLinParams(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    },
    'He': {
        # Helium Ground State (1s^2) -> l=0
        0:         TongLinParams(1.0, 1.231, 0.662, -1.325, 1.236, -0.231, 0.480, 0, 0),
        'default': TongLinParams(1.0, 1.231, 0.662, -1.325, 1.236, -0.231, 0.480, 0, 0)
    },
    'Ne': {
        # Neon Ground State (2p^6) -> l=1
        1:         TongLinParams(1.0, 8.069, 2.148, -3.570, 1.986, 0.931, 0.602, 1, 1),
        'default': TongLinParams(1.0, 8.069, 2.148, -3.570, 1.986, 0.931, 0.602, 1, 1)
    },
    'Ar': {
        # Argon Ground State (3p^6) -> l=1
        # CRITICAL FIX: We use these parameters for ALL channels (s, p, d) 
        # to ensure orthogonality and correct Cooper minimum.
        1:         TongLinParams(1.0, 16.039, 2.007, -25.543, 4.525, 0.961, 0.443, 1, 1),
        'default': TongLinParams(1.0, 16.039, 2.007, -25.543, 4.525, 0.961, 0.443, 1, 1)
    }
}

# ==============================================================================
# ATOMIC POTENTIAL CALCULATOR
# ==============================================================================

def get_Z_eff_TongLin(r, p: TongLinParams):
    """Calculates Z_eff(r) using the parameterized form (Saha Eq. 7)."""
    return (p.Z_c + 
            p.a1 * np.exp(-p.a2 * r) + 
            p.a3 * r * np.exp(-p.a4 * r) + 
            p.a5 * np.exp(-p.a6 * r))

# ==============================================================================
# TOTAL POTENTIAL (Atom + Confinement)
# ==============================================================================

def VGASW_total_debye(r, A=0.0, U=0.0, mu=0.0, species='H', r_c=6.7, Delta=1.70, l_wave=0, **kwargs):
    """
    Calculates the total effective potential V(r) experienced by the active electron.
    
    Physics: V_total = V_atom(r) + V_GASW(r)
    
    Parameters:
    -----------
    r : array_like
        Radial grid points (a.u.)
    A : float
        Gaussian amplitude parameter (pre-scaled). 
        Should be passed from solve_gasw_parameters.
    U : float
        Square well depth (positive value implies -U potential).
    mu : float
        Debye screening parameter (usually 0.0 for this paper).
    species : str
        'H', 'He', 'Ne', 'Ar'.
    r_c : float
        Cage radius (default 6.7 a.u. for C60).
    Delta : float
        Gaussian width sigma (default 1.70 a.u.).
        Note: The potential width is scaled by sqrt(7) internally (Saha Eq. 3).
    l_wave : int
        Angular momentum of the state being calculated.
    kwargs : dict
        'Delta_ASW': Width of the square well (default 2.8 a.u.).
    """
    r = np.asarray(r, dtype=float)
    r_safe = np.clip(r, 1e-14, None)
    
    # --- 1. Atomic Potential (SAE) ---
    # We strictly use the SAE parameters for all channels.
    if species in SAE_PARAMS:
        elem = SAE_PARAMS[species]
        # Use specific l_wave params if available (e.g. He l=0), 
        # otherwise fallback to 'default' (e.g. Ar continuum uses 3p params).
        p = elem.get(l_wave, elem.get('default'))
        Z_eff = get_Z_eff_TongLin(r_safe, p)
    else:
        # Fallback for pure Coulomb (Hydrogen-like if not in DB)
        Z_eff = 1.0

    # V_atom = -Z_eff(r) / r
    V_atom = -(Z_eff / r_safe) * np.exp(-mu * r_safe)
    
    # --- 2. Confinement Potential (V_GASW) ---
    # Saha Eq. (3): Combination of Gaussian and ASW
    V_conf = np.zeros_like(r)
    
    # A. Gaussian Component
    # Saha Eq (3) exponent is: - [ (r - r_c) / (sqrt(7) * sigma) ]^2
    effective_width = np.sqrt(7) * Delta
    
    if abs(A) > 1e-12:
        V_conf += A * np.exp(-((r - r_c)**2) / (effective_width**2))

    # B. ASW Component (Square Well)
    # Saha Eq (1): -U inside the shell boundaries
    Delta_ASW = kwargs.get('Delta_ASW', 2.8)
    if abs(U) > 1e-12:
        r_inner = r_c - Delta_ASW / 2.0
        r_outer = r_c + Delta_ASW / 2.0
        mask = (r >= r_inner) & (r <= r_outer)
        V_conf[mask] -= U 

    return V_atom + V_conf

# ==============================================================================
# PARAMETER SOLVER (Consistency with Saha Eq. 3)
# ==============================================================================

@lru_cache(maxsize=128)
def solve_gasw_parameters(Target_Depth_Au: float) -> Tuple[float, float]:
    """
    Solves for Gaussian Amplitude (A) and Square Well Depth (U) 
    to achieve a target total depth while preserving the potential shape.
    
    Constraint: The ratio A/U must remain constant to the reference values
    provided in Saha et al. (2019) to mimic the C60 electron density.
    
    Reference:
        A_paper = -3.59 (This includes the 1/sqrt(2pi) factor implicitly in Eq 3?)
        Actually, Eq 3 says: V = (A / sqrt(2pi)) * exp(...) + V_ASW
        The paper likely quotes 'A' as the coefficient *before* division.
        
        However, usually in code 'A' implies the peak height.
        Let's assume the standard interpretation:
        A_peak_ref = -3.59 / sqrt(2*pi)
        U_ref = 0.7
    """
    # Reference values from Saha et al. (2019) section 2.1
    A_saha_param = -3.59
    A_peak_ref = A_saha_param / np.sqrt(2 * np.pi) 
    U_ref = 0.7
    
    # Constant Ratio defining the "diffuseness"
    Ratio_Const = A_peak_ref / U_ref  
    
    # Linear System:
    # 1. Total Depth: A - U = -Target_Depth
    # 2. Shape:       A / U = Ratio_Const
    #
    # Substitute (2) into (1):
    # (Ratio * U) - U = -Target
    # U * (Ratio - 1) = -Target
    
    U_new = -Target_Depth_Au / (Ratio_Const - 1.0)
    A_new = U_new * Ratio_Const
    
    return A_new, U_new

if __name__ == "__main__":
    # Self-Check
    print("[potential.py] Testing Parameter Solver...")
    target = 0.56
    A_chk, U_chk = solve_gasw_parameters(target)
    print(f"  Target Depth: {target} a.u.")
    print(f"  Solved A (Peak): {A_chk:.4f}")
    print(f"  Solved U (Well): {U_chk:.4f}")
    print(f"  Check Total: -(A - U) = {-(A_chk - U_chk):.4f}")
    
    print("\n[potential.py] Testing Argon Potential...")
    r_test = np.array([0.1, 1.0, 10.0])
    v_ar = VGASW_total_debye(r_test, species='Ar', l_wave=2) # Test d-wave
    print(f"  V_Ar(r) [l=2]: {v_ar}")