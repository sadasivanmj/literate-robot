"""
Master runner - reads config.txt and executes selected calculations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Import all modules
from potential import VGASW_total_debye, solve_gasw_parameters
from bound import solve_ground_u
from continuum import compute_continuum_state
from cross_section import compute_cross_section_spectrum


def read_config(filename='config.txt'):
    """Parse simple config.txt file."""
    config = {}
    current_section = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                config[current_section] = {}
            elif '=' in line and current_section:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Parse values
                if value.lower() in ['yes', 'true']:
                    value = True
                elif value.lower() in ['no', 'false']:
                    value = False
                elif ',' in value:
                    value = [float(x.strip()) for x in value.split(',')]
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
                
                config[current_section][key] = value
    
    return config


def part1_gasw_parameters(config):
    """Solve for GASW parameters."""
    print("\n" + "="*70)
    print("PART 1: Solving GASW Parameters")
    print("="*70)
    
    targets = config['GASW_PARAMETERS']['target_depths']
    sigma = config['GASW_PARAMETERS']['sigma']
    R_const = config['GASW_PARAMETERS']['R_const']
    
    results = []
    for D in targets:
        A, U = solve_gasw_parameters(D, sigma, R_const)
        results.append({'V_GASW': D, 'A': A, 'U': U})
        print(f"  D = {D:.3f} → A = {A:.3f}, U = {U:.3f}")
    
    df = pd.DataFrame(results)
    
    if config['OUTPUT']['save_csv']:
        Path("results/csv").mkdir(parents=True, exist_ok=True)
        df.to_csv("results/csv/gasw_parameters.csv", index=False)
        print("✓ Saved: results/csv/gasw_parameters.csv")
    
    return df


def part2_varying_vgasw(config, gasw_df):
    """Plot wavefunctions for different V_GASW values."""
    print("\n" + "="*70)
    print("PART 2: Varying V_GASW at μ=0")
    print("="*70)
    
    mu = config['VARYING_VGASW']['mu_fixed']
    r_min, r_max = config['VARYING_VGASW']['plot_range']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['black', 'red', 'blue', 'magenta']
    
    energies = []
    wf_data = []
    
    for (_, row), color in zip(gasw_df.iterrows(), colors):
        r, u, E0, Veff, norm = solve_ground_u(
            VGASW_total_debye,
            R_max=config['NUMERICAL']['R_max'],
            N=int(config['NUMERICAL']['N_points']),
            ell=0,
            A=row['A'],
            U=row['U'],
            mu=mu
        )
        
        print(f"  V_GASW = {row['V_GASW']:.2f}: E₀ = {E0:.6f} a.u., norm = {norm:.6f}")
        energies.append({'V_GASW': row['V_GASW'], 'E0': E0, 'norm': norm})
        
        mask = r <= r_max
        wf_data.append({'V_GASW': row['V_GASW'], 'r': r[mask], 'u': u[mask]})
        
        # Plot wavefunction
        ax1.plot(r[mask], u[mask], color=color, lw=2, 
                label=f"V={row['V_GASW']:.2f}")
        
        # Plot potential
        V = VGASW_total_debye(r, row['A'], row['U'], mu)
        ax2.plot(r[mask], V[mask], color=color, lw=2,
                label=f"V={row['V_GASW']:.2f}")
    
    # Configure plots
    ax1.set_xlabel("r (a.u.)", fontsize=12)
    ax1.set_ylabel("u(r) (a.u.⁻¹ᐟ²)", fontsize=12)
    ax1.set_title(f"Wavefunctions (μ={mu})", fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel("r (a.u.)", fontsize=12)
    ax2.set_ylabel("V(r) (a.u.)", fontsize=12)
    ax2.set_title(f"Potentials (μ={mu})", fontsize=13)
    ax2.set_ylim(-1,1)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.axhline(0, color='gray', ls=':', alpha=0.5)
    
    plt.tight_layout()
    
    if config['OUTPUT']['save_plots']:
        Path("results/plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"results/plots/varying_vgasw_mu{mu:.2f}.png",
                   dpi=config['OUTPUT']['plot_dpi'], bbox_inches='tight')
        print("✓ Saved: results/plots/varying_vgasw_mu{:.2f}.png".format(mu))
    
    if config['OUTPUT']['show_plots']:
        plt.show()
    else:
        plt.close()
    
    # Save CSV
    if config['OUTPUT']['save_csv']:
        pd.DataFrame(energies).to_csv("results/csv/varying_vgasw_energies.csv", index=False)
        print("✓ Saved: results/csv/varying_vgasw_energies.csv")
    
    return energies


def part3_varying_mu(config):
    """Plot wavefunctions for different μ values."""
    print("\n" + "="*70)
    print("PART 3: Varying μ at V_GASW=0")
    print("="*70)
    
    mu_values = config['VARYING_MU']['mu_values']
    A = config['VARYING_MU']['A_fixed']
    U = config['VARYING_MU']['U_fixed']
    r_min, r_max = config['VARYING_VGASW']['plot_range']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['black', 'red', 'blue', 'green']
    
    energies = []
    
    for mu, color in zip(mu_values, colors):
        r, u, E0, Veff, norm = solve_ground_u(
            VGASW_total_debye,
            R_max=config['NUMERICAL']['R_max'],
            N=int(config['NUMERICAL']['N_points']),
            ell=0,
            A=A,
            U=U,
            mu=mu
        )
        
        print(f"  μ = {mu:.2f}: E₀ = {E0:.6f} a.u., norm = {norm:.6f}")
        energies.append({'mu': mu, 'E0': E0, 'norm': norm})
        
        mask = r <= r_max
        
        # Plot wavefunction
        ax1.plot(r[mask], u[mask], color=color, lw=2, label=f"μ={mu:.2f}")
        
        # Plot potential
        V = VGASW_total_debye(r, A, U, mu)
        ax2.plot(r[mask], V[mask], color=color, lw=2, label=f"μ={mu:.2f}")
    
    # Configure plots
    ax1.set_xlabel("r (a.u.)", fontsize=12)
    ax1.set_ylabel("u(r) (a.u.⁻¹ᐟ²)", fontsize=12)
    ax1.set_title("Wavefunctions (V_GASW=0)", fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel("r (a.u.)", fontsize=12)
    ax2.set_ylabel("V(r) (a.u.)", fontsize=12)
    ax2.set_title("Potentials (V_GASW=0)", fontsize=13)
    ax2.set_ylim(-1,1)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.axhline(0, color='gray', ls=':', alpha=0.5)
    
    plt.tight_layout()
    
    if config['OUTPUT']['save_plots']:
        plt.savefig("results/plots/varying_mu_vgasw0.png",
                   dpi=config['OUTPUT']['plot_dpi'], bbox_inches='tight')
        print("✓ Saved: results/plots/varying_mu_vgasw0.png")
    
    if config['OUTPUT']['show_plots']:
        plt.show()
    else:
        plt.close()
    
    # Save CSV
    if config['OUTPUT']['save_csv']:
        pd.DataFrame(energies).to_csv("results/csv/varying_mu_energies.csv", index=False)
        print("✓ Saved: results/csv/varying_mu_energies.csv")
    
    return energies


def part4_bound_state(config):
    """Compute reference bound state."""
    print("\n" + "="*70)
    print("PART 4: Reference Bound State")
    print("="*70)
    
    A = config['BOUND_STATE']['A_cage']
    U = config['BOUND_STATE']['U_cage']
    mu = config['BOUND_STATE']['mu_bound']
    ell = config['BOUND_STATE']['ell_bound']
    
    r, u, E0, Veff, norm = solve_ground_u(
        VGASW_total_debye,
        R_max=config['NUMERICAL']['R_max'],
        N=int(config['NUMERICAL']['N_points']),
        ell=ell,
        A=A,
        U=U,
        mu=mu
    )
    
    print(f"  E₀ = {E0:.6f} a.u.")
    print(f"  Normalization: {norm:.6f}")
    print(f"  Peak at r = {r[np.argmax(u)]:.3f} a.u.")
    
    # Diagnostic plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    mask = r <= 16
    ax.plot(r[mask], u[mask], 'b-', lw=2, label="u₁s(r)")
    ax2 = ax.twinx()
    V = VGASW_total_debye(r, A, U, mu)
    ax2.plot(r[mask], V[mask], 'r-', lw=2, label="V(r)")
    
    ax.set_xlabel("r (a.u.)", fontsize=12)
    ax.set_ylabel("u(r)", fontsize=12, color='b')
    ax2.set_ylabel("V(r) (a.u.)", fontsize=12, color='r')
    ax2.set_ylim(-1,1)
    ax.set_title(f"Bound State (A={A}, U={U}, μ={mu})", fontsize=13)
    ax.grid(alpha=0.3)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    if config['OUTPUT']['save_plots']:
        plt.savefig("results/plots/bound_state_diagnostic.png",
                   dpi=config['OUTPUT']['plot_dpi'], bbox_inches='tight')
        print("✓ Saved: results/plots/bound_state_diagnostic.png")
    
    if config['OUTPUT']['show_plots']:
        plt.show()
    else:
        plt.close()
    
    # Save CSV
    if config['OUTPUT']['save_csv']:
        df = pd.DataFrame({'r': r, 'u': u, 'V': V})
        df.to_csv("results/csv/bound_state_1s.csv", index=False)
        print("✓ Saved: results/csv/bound_state_1s.csv")
    
    return r, u, E0


def part5_cross_section(config, r_bound, u_bound, E_bound):
    """Compute photoionization cross section."""
    print("\n" + "="*70)
    print("PART 5: Photoionization Cross Section")
    print("="*70)
    
    ell_cont = config['CROSS_SECTION']['ell_continuum']
    E_min, E_max, N_points = config['CROSS_SECTION']['E_pe_range']
    E_pe_array = np.linspace(E_min, E_max, int(N_points))
    
    A = config['BOUND_STATE']['A_cage']
    U = config['BOUND_STATE']['U_cage']
    mu = config['BOUND_STATE']['mu_bound']
    
    print(f"  Computing {len(E_pe_array)} energy points...")
    
    sigma_arr, D_arr, diag_list = compute_cross_section_spectrum(
        E_pe_array, r_bound, u_bound, E_bound, ell_cont, A, U, mu
    )
    
    print(f"  Max cross section: {sigma_arr.max():.6e} a.u.")
    print(f"  at E_pe = {E_pe_array[np.argmax(sigma_arr)]:.3f} a.u.")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(E_pe_array, sigma_arr, 'b-', lw=2)
    ax.set_xlabel("Photoelectron Energy E_pe (a.u.)", fontsize=12)
    ax.set_ylabel("Cross Section σ (a.u.)", fontsize=12)
    ax.set_title("Photoionization Cross Section (1s → εp)", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    if config['OUTPUT']['save_plots']:
        plt.savefig("results/plots/photoionization_cross_section.png",
                   dpi=config['OUTPUT']['plot_dpi'], bbox_inches='tight')
        print("✓ Saved: results/plots/photoionization_cross_section.png")
    
    if config['OUTPUT']['show_plots']:
        plt.show()
    else:
        plt.close()
    
    # Save CSV
    if config['OUTPUT']['save_csv']:
        df = pd.DataFrame({
            'E_pe': E_pe_array,
            'sigma': sigma_arr,
            'dipole_D': D_arr
        })
        df.to_csv("results/csv/photoionization_cross_section.csv", index=False)
        print("✓ Saved: results/csv/photoionization_cross_section.csv")
    
    return E_pe_array, sigma_arr


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("HYDROGEN PHOTOIONIZATION CALCULATOR")
    print("="*70)
    
    # Read configuration
    config = read_config('config.txt')
    print("\n✓ Configuration loaded from config.txt")
    
    # Run selected parts
    run_ctrl = config['RUN_CONTROL']
    
    gasw_df = None
    r_bound, u_bound, E_bound = None, None, None
    
    if run_ctrl.get('solve_gasw_parameters', False):
        gasw_df = part1_gasw_parameters(config)
    
    if run_ctrl.get('plot_varying_vgasw', False):
        if gasw_df is None:
            gasw_df = part1_gasw_parameters(config)
        part2_varying_vgasw(config, gasw_df)
    
    if run_ctrl.get('plot_varying_mu', False):
        part3_varying_mu(config)
    
    if run_ctrl.get('compute_bound_state', False):
        r_bound, u_bound, E_bound = part4_bound_state(config)
    
    if run_ctrl.get('compute_cross_section', False):
        if r_bound is None:
            r_bound, u_bound, E_bound = part4_bound_state(config)
        part5_cross_section(config, r_bound, u_bound, E_bound)
    
    print("\n" + "="*70)
    print("ALL TASKS COMPLETED")
    print("="*70)
    print("\nResults saved in:")
    print("  • results/csv/")
    print("  • results/plots/")


if __name__ == "__main__":
    main()