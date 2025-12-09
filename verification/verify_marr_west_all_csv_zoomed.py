import pandas as pd
import matplotlib.pyplot as plt
import os

# CONFIGURATION
# Set this to where your CSV files are located
RESULTS_DIR = "." 

def plot_debug_separate(species, y_limit=10.0):
    print(f"\n--- Debug Plotting for {species} (Separate Plots) ---")
    
    # Define File Names
    sim_file = os.path.join(RESULTS_DIR, f"Simulation_{species}.csv")
    exp_file = os.path.join(RESULTS_DIR, f"Deviation_Report_{species}.csv")
    
    if not os.path.exists(sim_file) or not os.path.exists(exp_file):
        print(f"Error: Files for {species} not found in {RESULTS_DIR}")
        return
        
    try:
        # Load Data
        df_sim = pd.read_csv(sim_file)
        df_exp = pd.read_csv(exp_file)
        
        # Create Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # --- Plot 1: Simulation Only ---
        ax1.plot(df_sim['Wavelength (A)'], df_sim['Cross Section (Mb)'], 'b-', linewidth=2)
        ax1.set_title(f"{species}: Simulation Only (SAE)")
        ax1.set_xlabel("Wavelength (A)")
        ax1.set_ylabel("Cross Section (Mb)")
        ax1.set_ylim(0, y_limit) 
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis() # Standard spectroscopy convention
        
        # --- Plot 2: Experiment Only ---
        ax2.scatter(df_exp['Wavelength (A)'], df_exp['Exp (Mb)'], color='red', s=50)
        ax2.set_title(f"{species}: Experiment Only (Marr & West)")
        ax2.set_xlabel("Wavelength (A)")
        ax2.set_ylabel("Cross Section (Mb)")
        ax2.set_ylim(0, y_limit)
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()

        # Save
        output_file = f"Debug_Separate_{species}.png"
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"   > Saved: {output_file}")
        
    except Exception as e:
        print(f"   [!] Error plotting {species}: {e}")

if __name__ == "__main__":
    # Plot Argon (Zoom to 0-10 Mb to see Cooper Minimum)
    plot_debug_separate('Ar', y_limit=10.0)
    
    # Plot Neon (Zoom to 0-15 Mb to see the peak)
    plot_debug_separate('Ne', y_limit=15.0)