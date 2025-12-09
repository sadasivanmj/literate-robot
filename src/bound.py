"""
Bound state solver using finite difference method.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import trapezoid
from potential import VGASW_total_debye, SAE_PARAMS, solve_gasw_parameters

def solve_ground_u(V_func, species='H', R_max=60.0, N=6000, **Vkwargs):
    """
    Solves for the ground state wavefunction u(r).
    """
    # 1. Lookup SAE Parameters (l and n)
    if species in SAE_PARAMS:
        elem_data = SAE_PARAMS[species]
        # Use default if specific l not needed, but for bound state we know l
        params = elem_data.get('default', list(elem_data.values())[0])
        ell = params.ground_l
        state_idx = params.ground_n_idx
    else:
        ell = 0
        state_idx = 0
        
    # 2. Grid Setup
    r = np.linspace(1e-5, R_max, N)
    dr = r[1] - r[0]
    
    # 3. Potential (Pass 'ell' for GSZ switch)
    V_total = V_func(r, species=species, l_wave=ell, **Vkwargs)
    
    # 4. Hamiltonian Construction
    r_safe = np.clip(r, 1e-12, None)
    # Centrifugal term added here
    V_eff = V_total + ell * (ell + 1) / (2.0 * r_safe**2)
    
    # Finite Difference Matrix
    # We use inner points 1 to N-2 (excluding 0 and N-1 boundaries)
    N_inner = len(r) - 2
    k = 1.0 / (2.0 * dr**2)
    
    # Diagonal d has size N_inner
    d = 2.0 * k + V_eff[1:-1]
    
    # Off-diagonal e must have size N_inner - 1
    # PREVIOUS BUG WAS HERE: e = -k * np.ones(len(r) - 2)
    e = -k * np.ones(N_inner - 1)
    
    # 5. Diagonalization
    try:
        # Solve only for the lowest (state_idx + 1) eigenvalues
        E_all, U_all = eigh_tridiagonal(d, e, select='i', select_range=(0, state_idx + 1))
    except Exception as e:
        print(f"Solver Error: {e}")
        # Return dummy data to prevent crash, but indicate failure
        return r, np.zeros_like(r), 0.0, V_eff, 0.0
    
    # Select target state
    if len(E_all) <= state_idx:
        idx = len(E_all) - 1
    else:
        idx = state_idx
        
    E0 = E_all[idx]
    u_int = U_all[:, idx]
    
    # 6. Post-Processing
    # Sign convention: Ensure main lobe is positive
    max_idx = np.argmax(np.abs(u_int))
    if u_int[max_idx] < 0:
        u_int = -u_int
        
    # Normalization
    nrm = np.sqrt(trapezoid(u_int**2, r[1:-1]))
    if nrm < 1e-15: nrm = 1.0
    u_int /= nrm
    
    # Pad boundaries
    u_full = np.zeros_like(r)
    u_full[1:-1] = u_int
    
    norm_check = trapezoid(u_full**2, r)
    return r, u_full, E0, V_eff, norm_check

if __name__ == "__main__":
    print("=== Bound State Verification ===")
    
    # Test 1: Free Hydrogen
    r, u, E, V, nrm = solve_ground_u(VGASW_total_debye, species='H', A=0, U=0)
    print(f"Free H Energy: {E:.6f} a.u. (Theory: -0.5)")
    
    # Test 2: Confined H (Depth 0.56)
    target_D = 0.56
    A_val, U_val = solve_gasw_parameters(target_D)
    r_c, u_c, E_c, V_c, nrm_c = solve_ground_u(
        VGASW_total_debye, species='H', A=A_val, U=U_val
    )
    print(f"Confined H (0.56) Energy: {E_c:.6f} a.u.")