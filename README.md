# Hydrogen Photoionization in GASW Confinement

Computational study of photoionization cross sections for hydrogen atoms confined in Gaussian + Asymmetric Square Well (GASW) potentials with Debye screening.

## Features

- **GASW Potential**: Combines Coulomb, Gaussian, and square-well confinement
- **Debye Screening**: Models plasma screening effects
- **Bound States**: Finite-difference solver for 1s ground state
- **Continuum States**: Shooting method for photoelectron states
- **Cross Sections**: Photoionization cross sections via dipole matrix elements

## Installation

```bash
git clone https://github.com/yourusername/hydrogen-photoionization.git
cd hydrogen-photoionization
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- numpy
- scipy
- matplotlib
- pandas
- tqdm

## Quick Start

### 1. Edit Configuration
Open `config.txt` and set what you want to compute:

```ini
[RUN_CONTROL]
solve_gasw_parameters = yes
plot_varying_vgasw = yes
plot_varying_mu = yes
compute_bound_state = yes
compute_cross_section = yes
```

### 2. Run Everything
```bash
python run.py
```

Results appear in:
- `results/csv/` - Data tables
- `results/plots/` - Publication-quality figures

## What It Computes

### Part 1: GASW Parameters
Solves for potential parameters (A, U) matching target confinement depths.

**Output**: `gasw_parameters.csv`

### Part 2: Varying V_GASW
Wavefunctions and energies for different cage depths at fixed μ=0.

**Outputs**: 
- `varying_vgasw_energies.csv`
- `varying_vgasw_mu0.00.png`

### Part 3: Varying μ
Effect of Debye screening without cage confinement.

**Outputs**:
- `varying_mu_energies.csv`
- `varying_mu_vgasw0.png`

### Part 4: Reference Bound State
Computes 1s ground state for photoionization calculations.

**Outputs**:
- `bound_state_1s.csv`
- `bound_state_diagnostic.png`

### Part 5: Photoionization Cross Section
1s → εp transition cross sections vs photoelectron energy.

**Outputs**:
- `photoionization_cross_section.csv`
- `photoionization_cross_section.png`

## Module Structure

```
potential.py       - VGASW potential and parameter solver
bound.py          - Bound state finite-difference solver
continuum.py      - Continuum state shooting method
normalization.py  - Energy normalization for continuum states
cross_section.py  - Dipole elements and cross sections
run.py            - Master script (reads config.txt)
```

## Verification Steps

### Level 1: Module Tests
```bash
python potential.py      # Should print test output
python bound.py
python continuum.py
python normalization.py
python cross_section.py
```

### Level 2: Individual Parts
Edit `config.txt` to run one part:
```ini
[RUN_CONTROL]
solve_gasw_parameters = yes
plot_varying_vgasw = no
...
```
Then: `python run.py`

### Level 3: Check Outputs
```bash
ls results/csv/          # Should see .csv files
ls results/plots/        # Should see .png files
head results/csv/gasw_parameters.csv
```

### Level 4: Physics Validation
- Bound state energy should be ≈ -0.5 a.u. for free hydrogen (A=U=μ=0)
- Wavefunction normalization should be ≈ 1.0
- Cross section should decrease at high energies
- Deeper cages → less negative energies (less bound)

## Physics Background

The total potential is:

```
V(r) = V_Coulomb + V_Gaussian + V_SquareWell

V_Coulomb = -Z exp(-μr) / r          (Debye-screened)
V_Gaussian = A exp(-(r-r_c)²/2σ²)    (Cage potential)
V_SquareWell = -U  for r ∈ [r_c-Δ/2, r_c+Δ/2]
```

Parameters from Saha et al.:
- σ = 1.70 a.u. (Gaussian width)
- r_c = 6.7 a.u. (cage radius)
- Δ = 2.8 a.u. (well width)

## Citation

If you use this code, please cite:
```
[Your publication or arXiv reference]
```

Based on methodology from:
- Saha et al., [Journal Reference]

## License

MIT License - See LICENSE file

## Contact

Issues: https://github.com/yourusername/hydrogen-photoionization/issues