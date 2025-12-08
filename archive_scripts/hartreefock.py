import pyscf
from pyscf import gto, scf

# 1. Define the Helium Atom
mol = gto.M(
    atom = 'He 0 0 0',   # Helium at origin
    basis = 'aug-cc-pvqz', # Very high quality basis set
    spin = 0             # Singlet state
)

# 2. Run Hartree-Fock
mf = scf.RHF(mol)
energy = mf.kernel()

# 3. Extract Ionization Potential (Koopmans' Theorem)
# HOMO energy is roughly -Ip
homo_energy = mf.mo_energy[mf.mo_occ > 0][-1]

print(f"\n--- PySCF RESULTS ---")
print(f"Total HF Energy: {energy:.6f} a.u.")
print(f"Orbital Energy (1s): {homo_energy:.6f} a.u.")
print(f"Ionization Potential: {abs(homo_energy)*27.211:.3f} eV")