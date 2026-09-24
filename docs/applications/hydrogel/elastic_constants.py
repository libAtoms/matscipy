"""
Elastic constants calculation for hydrogel on diamond lattice using matscipy.

This example creates a hydrogel network on a diamond lattice and computes
the elastic constants by fitting stress-strain relationships.

Physical parameters (in LJ units where kT = 1):
    - Kuhn length b = 1.0
    - Chain length N = 50 monomers
    - Monomer volume v0 = 4*pi/3 (sphere of diameter b)
    - Flory-Huggins parameter chi = 0.5 (moderate solvent quality)
    - Coordination number = 4 (diamond lattice)

The crosslinker spacing is chosen as sqrt(N)*b ~ 7.07, which gives
chains at their equilibrium (Gaussian) extension.
"""

import numpy as np
from ase.build import bulk
from ase.optimize import FIRE

from matscipy.calculators.hydrogel import Hydrogel
from matscipy.elasticity import Voigt_6x6_to_cubic, fit_elastic_constants
from matscipy.molecules import Molecules
from matscipy.neighbours import neighbour_list


def create_diamond_hydrogel(N=50, b=1.0, n_cells=2):
    """Create a diamond lattice hydrogel.

    Parameters
    ----------
    N : int
        Number of monomers per chain
    b : float
        Kuhn length
    n_cells : int
        Number of unit cells in each direction

    Returns
    -------
    atoms : ase.Atoms
        ASE Atoms object with the hydrogel structure
    molecules : matscipy.molecules.Molecules
        Bond topology
    """
    # Equilibrium end-to-end distance
    R0 = np.sqrt(N) * b

    # Diamond lattice constant (nearest neighbor distance = a*sqrt(3)/4)
    # We want nearest neighbor distance = R0
    a = R0 * 4 / np.sqrt(3)

    # Create diamond lattice
    atoms = bulk("C", "diamond", a=a, cubic=True)
    atoms = atoms.repeat((n_cells, n_cells, n_cells))

    # Set masses (LJ units)
    atoms.set_masses(np.ones(len(atoms)))

    # Create bonds between nearest neighbors
    # Nearest neighbor distance in diamond is a*sqrt(3)/4 = R0
    nn_dist = a * np.sqrt(3) / 4
    i_p, j_p = neighbour_list("ij", atoms, nn_dist * 1.1)

    # Keep only unique bonds (i < j)
    mask = i_p < j_p
    bonds = np.column_stack([i_p[mask], j_p[mask]])

    molecules = Molecules(bonds_connectivity=bonds)

    return atoms, molecules


def main():
    # Physical parameters
    N = 50  # Monomers per chain
    b = 1.0  # Kuhn length
    v0 = 4 * np.pi / 3  # Monomer volume
    chi = 0.5  # Flory-Huggins parameter
    coord = 4  # Coordination number

    # Derived parameters
    R0 = np.sqrt(N) * b  # RMS end-to-end distance of a free chain
    L0 = (N - 1) * b  # Contour length
    rc = 3 * R0  # Cutoff radius

    print("=========================================")
    print("Hydrogel Elastic Constants Calculation")
    print("=========================================")
    print()
    print("Physical parameters:")
    print(f"  Kuhn length b = {b}")
    print(f"  Chain length N = {N}")
    print(f"  Flory parameter chi = {chi}")
    print(f"  Coordination = {coord}")
    print(f"  RMS end-to-end distance R0 = {R0:.4f}")
    print(f"  Contour length L0 = {L0:.4f}")
    print(f"  Cutoff radius rc = {rc:.4f}")

    # Create the hydrogel
    atoms, molecules = create_diamond_hydrogel(N=N, b=b, n_cells=2)

    print("\nSystem:")
    print(f"  {len(atoms)} crosslinkers")
    print(f"  {len(molecules.bonds)} bonds")
    print(
        f"  Box size: {atoms.cell[0, 0]:.4f} x {atoms.cell[1, 1]:.4f} x "
        f"{atoms.cell[2, 2]:.4f}"
    )

    # Create calculator
    calc = Hydrogel(
        cutoff=rc,
        chain_monomers=N,
        kuhn_length=b,
        monomer_volume=v0,
        flory_chi=chi,
        coordination=coord,
        molecules=molecules,
    )
    atoms.calc = calc

    # Initial state
    energy = atoms.get_potential_energy()
    stress = atoms.get_stress()
    print("\nInitial state:")
    print(f"  Energy = {energy:.4f}")
    print(f"  Stress (Voigt): {stress}")

    # Minimize energy with box relaxation
    print("\nMinimizing energy...")
    optimizer = FIRE(atoms)
    optimizer.run(fmax=1e-6, steps=500)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    max_force = np.max(np.abs(forces))

    print("\nAfter minimization:")
    print(f"  Energy = {energy:.4f}")
    print(f"  Max force = {max_force:.6f}")
    print(f"  Stress (Voigt): {stress}")

    # Compute elastic constants by stress-strain fitting
    print("\n=========================================")
    print("Computing Elastic Constants")
    print("=========================================")

    # Use cubic symmetry since diamond has cubic symmetry
    # Small strain amplitude and more steps for accuracy
    C, C_err = fit_elastic_constants(
        atoms,
        symmetry="cubic",
        N_steps=5,
        delta=1e-4,
        optimizer=FIRE,
        fmax=1e-6,
        verbose=True,
    )

    # Extract cubic elastic constants
    try:
        C11, C12, C44 = Voigt_6x6_to_cubic(C)

        print("\n=========================================")
        print("Results (cubic elastic constants)")
        print("=========================================")
        print(f"C11 = {C11:.6f} kT/b³")
        print(f"C12 = {C12:.6f} kT/b³")
        print(f"C44 = {C44:.6f} kT/b³")

        # Derived moduli
        bulk_modulus = (C11 + 2 * C12) / 3
        shear_modulus_voigt = C44
        shear_modulus_reuss = (C11 - C12) / 2

        print(f"\nBulk modulus K = {bulk_modulus:.6f} kT/b³")
        print(f"Shear modulus G (Voigt, C44) = {shear_modulus_voigt:.6f} kT/b³")
        print(
            f"Shear modulus G' (Reuss, (C11-C12)/2) = {shear_modulus_reuss:.6f} kT/b³"
        )

        if abs(C11 - C12) > 1e-10:
            poisson = C12 / (C11 + C12)
            print(f"Poisson ratio nu = {poisson:.4f}")

            # Zener anisotropy ratio
            anisotropy = 2 * C44 / (C11 - C12)
            print(f"\nZener anisotropy A = 2*C44/(C11-C12) = {anisotropy:.4f}")
            print("(A = 1 for an isotropic material)")

    except ValueError as e:
        print(f"\nCould not extract cubic constants: {e}")
        print("\nFull 6x6 elastic tensor:")
        print(C)

    # Compare with theoretical estimate for rubber elasticity
    print("\n=========================================")
    print("Theoretical Comparison")
    print("=========================================")

    # Rubber elasticity theory: G ~ n*kT where n is crosslink density
    # For diamond lattice: n = 8/V_cell crosslinkers, each with 4/2=2 effective chains
    V_cell = atoms.get_volume() / (2**3)  # Volume per unit cell
    n_chains = 8 * 2 / V_cell  # Chain density (chains per volume)

    # Phantom network theory: G = (1 - 2/f) * n * kT = 0.5 * n * kT for f=4
    G_phantom = 0.5 * n_chains

    # Affine network theory: G = n * kT
    G_affine = n_chains

    print(f"Chain density n = {n_chains:.6f}")
    print(f"Phantom network theory: G = {G_phantom:.6f} kT/b³")
    print(f"Affine network theory: G = {G_affine:.6f} kT/b³")

    print("\n=========================================")
    print("Calculation complete.")
    print("=========================================")


if __name__ == "__main__":
    main()
