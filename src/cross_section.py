"""
Photoionization cross section calculations with Single Active Electron (SAE) support.
Handles multi-channel transitions (e.g., p->s and p->d) and parallel processing.
"""
import numpy as np
from scipy.integrate import trapezoid
from multiprocessing import Pool, cpu_count
from functools import partial
from continuum import compute_continuum_state

# Fine-structure constant in atomic units
ALPHA_FS = 1.0 / 137.036

def dipole_matrix_element(r_cont, u_cont, r_bound, u_bound):
    """
    Calculate dipole matrix element.
    FIXED: Interpolates smooth continuum onto dense bound grid.
    """
    # Interpolate CONTINUUM onto BOUND grid
    u_cont_interp = np.interp(r_bound, r_cont, u_cont, left=0.0, right=0.0)
    
    # Integrate on the dense bound grid
    integrand = u_cont_interp * r_bound * u_bound
    
    # Use the dense grid for integration
    D = trapezoid(integrand, r_bound)
    
    return D


def _compute_single_energy(E_pe, r_bound, u_bound, E_bound, l_initial, species, A, U, mu, method='standard'):
    """
    Worker function for parallel processing.
    Computes total cross-section at a single energy by summing partial wave channels.
    
    Physics Formula (Bethe-Salpeter):
    sigma = (4*pi^2 * alpha * w / 3) * (1/(2l_i+1)) * [ Weights * |D|^2 ]
    """
    try:
        # 1. Determine allowed final angular momenta (Selection Rule: delta_l = +/- 1)
        l_finals = []
        
        # l -> l-1 (only if l > 0)
        if l_initial > 0:
            l_finals.append(l_initial - 1)
            
        # l -> l+1 (always allowed)
        l_finals.append(l_initial + 1)
        
        dipole_sum = 0.0
        details = {}
        
        # 2. Loop over final channels
        for l_f in l_finals:
            # Solve continuum state for this specific channel
            r_cont, u_cont, diag = compute_continuum_state(E_pe, l_f, species, A, U, mu)
            
            # Calculate Dipole Matrix Element D
            D = dipole_matrix_element(r_cont, u_cont, r_bound, u_bound)
            
            # --- Apply Angular Weighting Factors ---
            if method == 'saha' and l_initial == 1:
                # Saha / Rescattering Geometry (Differential 0 deg)
                # Specific for p-states (Ar, Ne): s-wave is suppressed.
                if l_f == 0: # s-wave (l-1)
                    weight = 0.25
                else:        # d-wave (l+1)
                    weight = 1.0
            else:
                # Standard Total Cross Section (Angle Integrated)
                # Weights come from angular momentum algebra (2l_max + 1, etc.)
                # For l -> l-1: weight = l
                # For l -> l+1: weight = l+1
                if l_f == l_initial - 1:
                    weight = l_initial
                else:
                    weight = l_initial + 1
                
            dipole_sum += weight * (abs(D)**2)
            
            # Store detail for debugging/analysis
            details[f'D_l{l_f}'] = D
            details[f'sigma_partial_l{l_f}'] = abs(D)**2 # Unscaled partial sigma proxy

        # 3. Calculate Final Cross Section
        E_photon = E_pe - E_bound # Photon energy
        
        # Pre-factor: (4 * pi^2 * alpha * E_ph) / (3 * (2l_i + 1))
        prefactor = (4.0 * np.pi**2 * ALPHA_FS * E_photon) / (3.0 * (2.0 * l_initial + 1.0))
        
        sigma = prefactor * dipole_sum
        
        return {
            'success': True,
            'E_pe': E_pe,
            'sigma': sigma,
            'details': details
        }
        
    except Exception as e:
        # Return error object securely without crashing the pool
        return {
            'success': False, 
            'E_pe': E_pe, 
            'error': str(e), 
            'sigma': 0.0
        }


def compute_cross_section_spectrum(E_pe_array, r_bound, u_bound, E_bound, 
                                   l_initial, species, A, U, mu, n_workers=None, method='standard'):
    """
    Compute photoionization cross section over an energy range using parallel workers.
    
    Parameters:
        E_pe_array : Array of photoelectron energies (a.u.)
        r_bound    : Bound state radial grid
        u_bound    : Bound state wavefunction
        E_bound    : Bound state energy (negative)
        l_initial  : Angular momentum of bound state (0 for s, 1 for p)
        species    : 'H', 'He', 'Ne', 'Ar'
        A, U       : Confinement parameters
        mu         : Debye screening parameter
        n_workers  : Number of CPU cores to use
        method     : 'standard' (default) or 'saha' (for Cooper minimum depth studies)
    
    Returns:
        sigma_array : Array of cross sections (a.u.)
        diag_list   : List of diagnostic dictionaries
    """
    # 1. Setup Parallel Processing
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    # Create partial function with fixed system parameters AND method
    worker_func = partial(_compute_single_energy,
                        r_bound=r_bound,
                        u_bound=u_bound,
                        E_bound=E_bound,
                        l_initial=l_initial,
                        species=species,
                        A=A, U=U, mu=mu,
                        method=method)
    
    # 2. Run Calculation Pool
    # We rely on the caller to protect the entry point (if __name__ == "__main__")
    results = []
    
    # For very small jobs, avoid overhead of Pool
    if len(E_pe_array) < 5:
        results = [worker_func(E) for E in E_pe_array]
    else:
        with Pool(processes=n_workers) as pool:
            # Map the energies to the worker function
            results = pool.map(worker_func, E_pe_array)
            
    # 3. Unpack Results
    sigma_list = []
    diag_list = []
    
    for res in results:
        if res['success']:
            sigma_list.append(res['sigma'])
            diag_list.append(res['details'])
        else:
            # Handle failures gracefully (e.g., numerov overflow at extreme energies)
            sigma_list.append(0.0)
            diag_list.append({'error': res.get('error', 'Unknown error')})
            
    return np.array(sigma_list), diag_list