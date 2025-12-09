import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# CONFIGURATION
RESULTS_DIR = "results"  # Current folder (or change to your path)

def plot_final_clean(species, cutoff_wavelength, y_max=None):
    print(f"\n--- Plotting {species} (Cleaned & Aligned) ---")
    
    # 1. Define File Paths
    sim_file = os.path.join(RESULTS_DIR, f"Simulation_{species}.csv")
    exp_file = os.path.join(RESULTS_DIR, f"Deviation_Report_{species}.csv") # Or Deviation_MarrWest...
    
    if not os.path.exists(sim_file) or not os.path.exists(exp_file):
        print(f"  [!] Missing files for {species}")
        return

    # 2. Load Data
    try:
        df_sim = pd.read_csv(sim_file)
        df_exp = pd.read_csv(exp_file)
        
        # 3. FIX: Standardize Column Names (Strip whitespace)
        df_sim.columns = [c.strip() for c in df_sim.columns]
        df_exp.columns = [c.strip() for c in df_exp.columns]

        # 4. FIX: Detect Correct Columns explicitly
        # Simulation usually has 'Wavelength (A)'
        sim_x_col = next((c for c in df_sim.columns if 'Wavelength' in c or 'Lambda' in c), None)
        sim_y_col = next((c for c in df_sim.columns if 'Cross' in c or 'Sigma' in c), None)
        
        # Experiment usually has 'Wavelength (A)' or 'Lambda (A)'
        exp_x_col = next((c for c in df_exp.columns if 'Wavelength' in c or 'Lambda' in c), None)
        # Look for 'Exp', 'Marr', 'Cooper' for Y-axis
        exp_y_col = next((c for c in df_exp.columns if any(k in c for k in ['Exp', 'Marr', 'Cooper'])), None)

        if not all([sim_x_col, sim_y_col, exp_x_col, exp_y_col]):
            print("  [!] Could not auto-detect columns. Check CSV headers.")
            return

        # 5. FIX: Sort Data by Wavelength (Prevents "messy" scribbles)
        df_sim = df_sim.sort_values(by=sim_x_col, ascending=False) # High lambda first
        df_exp = df_exp.sort_values(by=exp_x_col, ascending=False)

        # 6. FIX: Domain Alignment (Filter Zoom)
        sim_zoom = df_sim[df_sim[sim_x_col] < cutoff_wavelength]
        exp_zoom = df_exp[df_exp[exp_x_col] < cutoff_wavelength]

        # 7. Plot
        plt.figure(figsize=(10, 7), dpi=120)
        
        # Plot Simulation (Blue Line)
        plt.plot(sim_zoom[sim_x_col], sim_zoom[sim_y_col], 
                 'b-', linewidth=2.5, label='Simulation (SAE)')
        
        # Plot Experiment (Red Dots - No Line)
        plt.scatter(exp_zoom[exp_x_col], exp_zoom[exp_y_col], 
                    color='red', s=50, label='Experiment', zorder=5)
        
        plt.title(f"{species} Photoionization Cross Section", fontsize=14, fontweight='bold')
        plt.xlabel(r"Wavelength ($\AA$)", fontsize=12)
        plt.ylabel("Cross Section (Mb)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 8. Set Limits (Inverted X-axis for Spectroscopy)
        # Use the WIDEST range so we see everything
        x_max = max(sim_zoom[sim_x_col].max(), exp_zoom[exp_x_col].max())
        x_min = min(sim_zoom[sim_x_col].min(), exp_zoom[exp_x_col].min())
        
        plt.xlim(cutoff_wavelength, x_min) # Inverted: Max -> Min
        
        if y_max:
            plt.ylim(0, y_max)
            
        output_file = f"Final_Clean_{species}.png"
        plt.savefig(output_file)
        plt.close()
        print(f"  > Saved: {output_file}")

    except Exception as e:
        print(f"  [!] Error: {e}")

if __name__ == "__main__":
    # ARGON: Cut at 650 A to hide the threshold spike and focus on the minimum
    plot_final_clean('Ar', cutoff_wavelength=650, y_max=5.0)
    
    # NEON: Cut at 500 A to align the main dome
    plot_final_clean('Ne', cutoff_wavelength=500, y_max=15.0)