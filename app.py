import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import trapezoid

# ==============================================================================
# 1. PURE PYTHON PHYSICS ENGINE (OPTIMIZED)
# ==============================================================================

# SAE Parameters
PARAMS = {
    'H':  np.array([1.0, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]),
    'He': np.array([1.0, 1.231, 0.662, -1.325, 1.236, -0.231, 0.480]),
    'Ne': np.array([1.0, 8.069, 2.148, -3.570, 1.986, 0.931, 0.602]),
    'Ar': np.array([1.0, 16.039, 2.007, -25.543, 4.525, 0.961, 0.443]),
}

META = {'H': (0, 0), 'He': (0, 0), 'Ne': (1, 0), 'Ar': (1, 1)}

def get_Z_eff(r, species):
    p = PARAMS[species]
    term1 = p[1] * np.exp(-p[2] * r)
    term2 = p[3] * r * np.exp(-p[4] * r)
    term3 = p[5] * np.exp(-p[6] * r)
    return p[0] + term1 + term2 + term3

def VGASW_total_debye(r, A, U, mu, species):
    r = np.asarray(r)
    # Avoid singularity
    r_safe = np.where(r < 1e-12, 1e-12, r)
    
    # 1. Atomic Potential
    Z_eff = get_Z_eff(r_safe, species)
    V_atom = -(Z_eff / r_safe) * np.exp(-mu * r_safe)
    
    # 2. Gaussian Wall
    sigma, r_c, Delta = 1.70, 6.7, 2.8
    V_gauss = A * np.exp(-((r - r_c)**2) / (2.0 * sigma**2))
    
    # 3. Square Well
    r_in = r_c - Delta / 2.0
    r_out = r_c + Delta / 2.0
    V_asw = np.zeros_like(r)
    mask = (r >= r_in) & (r <= r_out)
    V_asw[mask] = -U
    
    return V_atom + V_gauss + V_asw

# --- Optimized Solvers ---

def solve_gasw_parameters(D, sigma=1.70, R_const=-24.5):
    K = np.sqrt(2.0 * np.pi) * sigma
    denominator = 1.0 - (R_const / K)
    if abs(denominator) < 1e-12: return -3.59, 0.7 
    U = D / denominator
    A = U - D
    return A, U

def solve_ground_u(species, A=0.0, U=0.0, mu=0.0):
    """
    Bound State Solver.
    STRATEGY: Use a small R_max but very high N to get high density at the start.
    """
    # High Density Settings
    R_max = 30.0   # Atom is small, we don't need a huge box
    N = 4000       # Very dense grid (lots of points in the beginning)
    
    ell, state_idx = META[species]
    r = np.linspace(1e-5, R_max, N)
    dr = r[1] - r[0]
    
    V_total = VGASW_total_debye(r, A, U, mu, species)
    r_safe = np.maximum(r, 1e-12)
    V_eff = V_total + ell * (ell + 1) / (2.0 * r_safe**2)
    
    r_int = r[1:-1]
    V_int = V_eff[1:-1]
    k = 1.0 / (2.0 * dr**2)
    
    d = 2.0 * k + V_int
    e = -k * np.ones(len(r_int) - 1)
    
    try:
        E_all, U_all = eigh_tridiagonal(d, e, select='i', select_range=(0, state_idx + 1))
    except Exception:
        return r, None, 0.0, ell
        
    if len(E_all) <= state_idx: return r, None, 0.0, ell
        
    E0 = E_all[state_idx]
    u_int = U_all[:, state_idx]
    
    # Normalize
    if u_int[np.argmax(np.abs(u_int))] < 0: u_int = -u_int
    nrm = np.sqrt(trapezoid(u_int**2, r_int))
    u_int /= nrm
    
    u_full = np.zeros_like(r)
    u_full[1:-1] = u_int
    
    return r, u_full, E0, ell

def solve_continuum_pure(E, ell, species, A, U, mu):
    """
    Continuum Solver.
    STRATEGY: Use a larger R_max (needed for waves) but lower N to save speed.
    """
    R_max = 100.0  # Needs room to oscillate
    N = 2000       # Less points to make the loop faster
    
    r = np.linspace(1e-4, R_max, N)
    h = r[1] - r[0]
    h2_12 = h * h / 12.0
    
    V_total = VGASW_total_debye(r, A, U, mu, species)
    r_safe = np.maximum(r, 1e-20)
    centrifugal = ell * (ell + 1) / (2.0 * r_safe**2)
    
    k_squared = 2.0 * (E - (V_total + centrifugal))
    
    # Numerov Loop
    u = np.zeros(N)
    u[0] = 0.0
    u[1] = 1e-10
    
    c1 = 1.0 - 5.0 * h2_12 * k_squared
    c2 = 1.0 + h2_12 * k_squared
    
    for n in range(1, N - 1):
        val = (2.0 * c1[n] * u[n] - c2[n-1] * u[n-1]) / c2[n+1]
        u[n+1] = val
        if abs(val) > 1e10: break

    # Normalize
    k_asy = np.sqrt(2 * E)
    norm_factor = np.sqrt(2 / (np.pi * k_asy))
    tail_vals = u[int(N*0.9):]
    max_amp = np.max(np.abs(tail_vals))
    if max_amp > 1e-20: u *= (norm_factor / max_amp)
            
    return r, u

@st.cache_data
def calculate_cross_section_scan(energies, r_b, u_b, E_b, l_i, species, A, U, mu):
    ALPHA = 1.0 / 137.036
    l_finals = [l for l in [l_i-1, l_i+1] if l >= 0]
    sigmas = []
    
    for E_pe in energies:
        dipole_sum = 0.0
        for l_f in l_finals:
            r_c, u_c = solve_continuum_pure(E_pe, l_f, species, A, U, mu)
            # Interpolate Bound State onto Continuum Grid
            u_b_interp = np.interp(r_c, r_b, u_b)
            # Integrate
            D_int = trapezoid(u_c * r_c * u_b_interp, r_c)
            weight = l_i if l_f < l_i else l_i + 1
            dipole_sum += weight * abs(D_int)**2
            
        E_ph = E_pe - E_b
        sigma = (4 * np.pi**2 * ALPHA * E_ph / 3) * (1/(2*l_i + 1)) * dipole_sum
        sigmas.append(sigma)
    return np.array(sigmas)

# ==============================================================================
# 2. STREAMLIT UI
# ==============================================================================

st.set_page_config(page_title="Safe Photoionization Lab", layout="wide")
st.title("⚛️ Photoionization Lab")

# Sidebar
st.sidebar.header("Settings")
species = st.sidebar.selectbox("Species", ["H", "He", "Ne", "Ar"], index=3)
use_cage = st.sidebar.checkbox("Enable C60 Cage", value=True)

# SPECIFIC DEPTHS REQUESTED
cage_depth = st.sidebar.selectbox("Cage Depth (D)", [0.30, 0.46, 0.56, 1.03], index=0)

mu = st.sidebar.slider("Screening (μ)", 0.0, 0.5, 0.0, 0.01)

# Logic
if use_cage:
    A_val, U_val = solve_gasw_parameters(cage_depth)
else:
    A_val, U_val = 0.0, 0.0

# Solve Bound State
r_b, u_b, E_b, l_init = solve_ground_u(species, A=A_val, U=U_val, mu=mu)

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Bound State")
    if u_b is None or E_b > 0:
        st.error("Atom Unbound! Increase Depth.")
    else:
        # 1. DISPLAY EIGENVALUE PROMINENTLY
        st.metric(label="Energy Eigenvalue", value=f"{E_b:.5f} a.u.")
        
        # 2. Plot Wavefunction
        fig_b, ax_b = plt.subplots(figsize=(4, 3))
        ax_b.plot(r_b, u_b, 'r-', lw=2)
        ax_b.fill_between(r_b, u_b, alpha=0.1, color='red')
        
        # Checkbox for Log Scale visualization
        if st.checkbox("Log Scale (Visual only)"):
            ax_b.set_xscale("log")
            ax_b.set_xlim(0.01, 30)
        else:
            ax_b.set_xlim(0, 20)
            
        ax_b.set_title(f"Wavefunction u(r)")
        ax_b.set_xlabel("r (a.u.)")
        ax_b.grid(True, alpha=0.3)
        st.pyplot(fig_b)

with col1:
    st.subheader("Cross Section")
    if u_b is not None:
        e_max = st.slider("Max Energy", 0.5, 5.0, 2.5)
        # Optimized resolution for Pure Python
        n_points = st.slider("Resolution", 20, 100, 50) 
        
        if st.button("Calculate Cross Section"):
            energies = np.linspace(0.0001, e_max, n_points)
            
            with st.spinner("Computing (Optimized Dual-Grid)..."):
                sigmas = calculate_cross_section_scan(
                    energies, r_b, u_b, E_b, l_init, species, A_val, U_val, mu
                )
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(energies, sigmas, 'b-o', markersize=4, label=f'Depth={cage_depth}')
            
            # Find Peak
            peak_idx = np.argmax(sigmas)
            ax.annotate(f'Peak: {energies[peak_idx]:.2f} a.u.', 
                        xy=(energies[peak_idx], sigmas[peak_idx]),
                        xytext=(energies[peak_idx]+0.5, sigmas[peak_idx]),
                        arrowprops=dict(facecolor='black', shrink=0.05))

            ax.set_xlabel("Photoelectron Energy (a.u.)")
            ax.set_ylabel("Cross Section (Mb)")
            ax.set_title(f"Photoionization of {species}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)