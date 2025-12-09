"""
Photoionization cross section calculations.
OPTIMIZED: Numba (JIT), Multiprocessing, and Flexible Arguments (**kwargs).
"""
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from continuum import compute_continuum_state
from numba import njit

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, *args, **kwargs): return iterable

ALPHA_FS = 1.0 / 137.036

@njit(fastmath=True, cache=True)
def dipole_matrix_element(r_cont, u_cont, r_bound, u_bound):
    u_cont_interp = np.interp(r_bound, r_cont, u_cont)
    integrand = u_cont_interp * r_bound * u_bound
    
    # Manual Trapezoid for Numba
    n = len(r_bound)
    if n < 2: return 0.0
    integral = 0.0
    for i in range(n - 1):
        dr = r_bound[i+1] - r_bound[i]
        avg = 0.5 * (integrand[i] + integrand[i+1])
        integral += avg * dr
    return integral

# Updated Worker to accept **kwargs
def _compute_single_energy(E_pe, r_bound, u_bound, E_bound, l_initial, species, A, U, mu, method='standard', **kwargs):
    try:
        l_finals = []
        if l_initial > 0: l_finals.append(l_initial - 1)
        l_finals.append(l_initial + 1)
        
        dipole_sum = 0.0
        details = {}
        
        for l_f in l_finals:
            # Pass **kwargs (r_c, Delta) to continuum solver
            r_cont, u_cont, diag = compute_continuum_state(E_pe, l_f, species, A, U, mu, **kwargs)
            D = dipole_matrix_element(r_cont, u_cont, r_bound, u_bound)
            
            if method == 'saha' and l_initial == 1:
                weight = 0.25 if l_f == 0 else 1.0
            else:
                weight = l_initial if l_f == l_initial - 1 else l_initial + 1
            
            dipole_sum += weight * (abs(D)**2)
            details[f'D_l{l_f}'] = D
        
        E_photon = E_pe + abs(E_bound)
        prefactor = (4.0 * np.pi**2 * ALPHA_FS * E_photon) / (3.0 * (2.0 * l_initial + 1.0))
        sigma = prefactor * dipole_sum
        
        return {'success': True, 'sigma': sigma, 'details': details}

    except Exception as e:
        return {'success': False, 'error': str(e), 'sigma': 0.0}

# Updated Orchestrator to accept **kwargs
def compute_cross_section_spectrum(E_pe_array, r_bound, u_bound, E_bound, 
                                   l_initial, species, A=0.0, U=0.0, mu=0.0, 
                                   n_workers=None, method='standard', **kwargs):
    
    if n_workers is None: n_workers = max(1, cpu_count() - 1)
    
    # Pass **kwargs to the worker
    worker = partial(_compute_single_energy,
                     r_bound=r_bound, u_bound=u_bound, E_bound=E_bound,
                     l_initial=l_initial, species=species,
                     A=A, U=U, mu=mu, method=method, **kwargs)
    
    results = []
    if n_workers > 1 and len(E_pe_array) > 5:
        with Pool(processes=n_workers) as pool:
            iterator = pool.imap(worker, E_pe_array)
            results = list(tqdm(iterator, total=len(E_pe_array), 
                                desc=f"    Calc Sigma ({species})", unit="pts"))
    else:
        results = [worker(E) for E in tqdm(E_pe_array, desc=f"    Calc Sigma ({species})")]
            
    sigma_list = [res['sigma'] if res['success'] else 0.0 for res in results]
    diag_list  = [res.get('details', {}) if res['success'] else {'error': res.get('error')} for res in results]
            
    return np.array(sigma_list), diag_list