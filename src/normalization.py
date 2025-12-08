"""
Continuum normalization routines using mpmath for exact Coulomb functions.
Critical for low-energy threshold behavior where asymptotic approximations fail.
"""
import numpy as np
from scipy.special import gamma
import mpmath

def get_coulomb_fg(ell, eta, rho_array):
    """
    Helper to compute Regular (F) and Irregular (G) Coulomb functions 
    using mpmath, handling numpy array inputs.
    
    Returns:
        F, G (numpy arrays of floats)
    """
    # mpmath functions do not accept numpy arrays directly.
    # Since we only normalize on a small 'fit' window (e.g., 50 points),
    # a list comprehension is fast enough.
    
    F_list = []
    G_list = []
    
    # Set precision to double (standard float)
    mpmath.mp.dps = 15
    
    for rho in rho_array:
        # mpmath.coulombf(l, eta, z)
        # mpmath.coulombg(l, eta, z)
        f_val = mpmath.coulombf(ell, eta, rho)
        g_val = mpmath.coulombg(ell, eta, rho)
        F_list.append(float(f_val))
        G_list.append(float(g_val))
        
    return np.array(F_list), np.array(G_list)


# ======================================================================
# Pure Coulomb continuum (free hydrogen-like)
# ======================================================================

def normalize_continuum_coulomb_free(r, u_raw, E, ell, Z=1.0):
    """
    Energy-normalize a pure Coulomb continuum state.
    """
    r = np.asarray(r, dtype=float)
    u_raw = np.asarray(u_raw, dtype=float)

    if E <= 0.0:
        raise ValueError("normalize_continuum_coulomb_free: E must be > 0")

    k = np.sqrt(2.0 * E)
    eta = -Z / k

    # Asymptotic window: r > max(30, 5/k)
    r_asymp_min = max(30.0, 5.0 / k)
    r_asymp_max = r.max() - 10.0  

    mask = (r >= r_asymp_min) & (r <= r_asymp_max)

    # Fallback if grid is too small
    if np.count_nonzero(mask) < 20:
        n_points = len(r)
        start = max(0, n_points - 50)
        mask = np.zeros(n_points, dtype=bool)
        mask[start:-5] = True

    r_as = r[mask]
    u_as = u_raw[mask]
    
    # Downsample for speed if too many points
    if len(r_as) > 100:
        stride = len(r_as) // 50
        r_as = r_as[::stride]
        u_as = u_as[::stride]

    # Compute Exact Coulomb Functions
    rho = k * r_as
    F_l, _ = get_coulomb_fg(ell, eta, rho)

    # Fit u_raw ≈ A * F_l
    denom = np.sum(F_l**2)
    if denom == 0.0:
        A = 1.0
    else:
        A = np.sum(u_as * F_l) / denom

    # Fix global sign
    if A < 0:
        A = -A
        u_raw = -u_raw

    # Energy normalization factor
    norm_factor = np.sqrt(2.0 / (np.pi * k))
    u_E = (u_raw / A) * norm_factor

    # Coulomb phase (sigma_l)
    sigma_l = np.angle(gamma(ell + 1 + 1j * eta))

    info = {
        "k": float(k),
        "eta": float(eta),
        "sigma_l": float(sigma_l),
        "A": float(A),
        "normalization_factor": float(norm_factor),
        "r_as_min": float(r_as[0]),
        "r_as_max": float(r_as[-1]),
    }
    return u_E, info


# ======================================================================
# Coulomb + short-range (GASW cage, etc.)
# ======================================================================

def normalize_continuum_coulomb_phase(r, u_raw, E, ell, Z=1.0, n_fit=None,
                                      max_fit_points=100):
    """
    Coulomb + short-range case (GASW cage, SAE atoms, etc).
    Fits u_raw to a combination a*F_l + b*G_l to find phase shift.
    """
    r = np.asarray(r, dtype=float)
    u_raw = np.asarray(u_raw, dtype=float)

    if E <= 0.0:
        raise ValueError("normalize_continuum_coulomb_phase: E must be > 0")

    k = np.sqrt(2.0 * E)
    eta = -Z / k

    # 1. Choose fitting region (Tail of the grid)
    if n_fit is None:
        # Auto-select: Use last 10% of grid, but ensure r > 30
        r_min_cut = max(30.0, r[-1] * 0.8)
        mask = r >= r_min_cut
        r_fit = r[mask]
        u_fit = u_raw[mask]
    else:
        r_fit = r[-n_fit:]
        u_fit = u_raw[-n_fit:]

    # Downsample to keep mpmath fast
    if len(r_fit) > max_fit_points:
        stride = int(np.ceil(len(r_fit) / max_fit_points))
        r_fit = r_fit[::stride]
        u_fit = u_fit[::stride]

    if len(r_fit) < 5:
        # Emergency fallback if grid is tiny
        r_fit = r[-10:]
        u_fit = u_raw[-10:]

    # 2. Compute Exact Coulomb F and G
    rho = k * r_fit
    F, G = get_coulomb_fg(ell, eta, rho)

    # 3. Least-squares fit: u_fit ≈ a*F + b*G
    M = np.column_stack((F, G))
    coef, residuals, rank, s = np.linalg.lstsq(M, u_fit, rcond=None)
    a, b = coef

    A = np.sqrt(a**2 + b**2)
    delta = np.arctan2(b, a)

    # 4. Normalize
    if A == 0.0: A = 1.0
    
    norm_factor = np.sqrt(2.0 / (np.pi * k))
    u_E = (u_raw / A) * norm_factor

    info = {
        "k": float(k),
        "eta": float(eta),
        "a": float(a),
        "b": float(b),
        "A": float(A),
        "delta_l": float(delta),
        "r_fit_min": float(r_fit[0]),
        "n_fit": int(r_fit.size),
    }
    return u_E, info


# ======================================================================
# Debye-screened case (μ > 0)
# ======================================================================

def energy_normalize_continuum(r, u_raw, E, ell, A, U, mu):
    """
    Energy normalization for Debye-screened case (μ>0).
    Envelope-based normalization.
    """
    r = np.asarray(r, dtype=float)
    u_raw = np.asarray(u_raw, dtype=float)
    k = np.sqrt(2.0 * E)

    # Use the last part of the grid to estimate amplitude
    n_tail = max(50, len(r) // 10)
    u_tail = u_raw[-n_tail:]

    envelope = np.max(np.abs(u_tail))
    if envelope == 0.0: envelope = 1.0

    factor = np.sqrt(2.0 / (np.pi * k))
    u_E = u_raw * (factor / envelope)

    info = {
        "envelope": float(envelope),
        "factor": float(factor),
        "normalization_type": "envelope"
    }
    return u_E, info