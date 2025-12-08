"""
Bound state solver using finite difference method for SAE potentials.
Automatically handles angular momentum (l) and state index (n) based on species.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import trapezoid
from potential import VGASW_total_debye, SAE_PARAMS

def solve_ground_u(V_func, species='H', R_max=60.0, N=6000, **Vkwargs):
    """
    Solve radial Schrödinger equation for the valence bound state.
    """
    # 1. Determine L and State Index
    if species in SAE_PARAMS:
        # Params: [..., ground_L, state_index]
        ell = SAE_PARAMS[species][7]
        state_idx = SAE_PARAMS[species][8]
    else:
        ell = 0
        state_idx = 0
        
    # 2. Grid & Potential
    r = np.linspace(1e-5, R_max, N)
    dr = r[1] - r[0]
    
    V_total = V_func(r, species=species, **Vkwargs)
    
    # Centrifugal Barrier
    r_safe = np.clip(r, 1e-12, None)
    V_eff = V_total + ell * (ell + 1) / (2.0 * r_safe**2)
    
    # 3. Hamiltonian (Finite Difference)
    r_int = r[1:-1]
    V_int = V_eff[1:-1]
    
    k = 1.0 / (2.0 * dr**2)
    d = 2.0 * k + V_int
    e = -k * np.ones(len(r_int) - 1)
    
    # 4. Diagonalize
    # We need enough eigenvalues to reach 'state_idx'
    # e.g., if index=1, we need range (0, 1) to get the second state
    E_all, U_all = eigh_tridiagonal(d, e, select='i', select_range=(0, state_idx + 1))
    
    # Safety check
    if len(E_all) <= state_idx:
        print(f"Warning: Desired state index {state_idx} not found for {species}. Using ground.")
        idx = 0
    else:
        idx = state_idx
        
    E0 = E_all[idx]
    u_int = U_all[:, idx]
    
    # 5. Normalize & Phase
    max_idx = np.argmax(np.abs(u_int))
    if u_int[max_idx] < 0:
        u_int = -u_int
    
    nrm = np.sqrt(trapezoid(u_int**2, r_int))
    if nrm < 1e-15: nrm = 1.0
    u_int /= nrm
    
    u_full = np.zeros_like(r)
    u_full[1:-1] = u_int
    
    norm_check = trapezoid(u_full**2, r)
    
    return r, u_full, E0, ell, norm_check