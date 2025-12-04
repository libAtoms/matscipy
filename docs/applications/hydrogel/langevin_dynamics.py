"""
Hydrogel Langevin dynamics simulation using matscipy.

This example creates a hydrogel network on a diamond lattice and runs
molecular dynamics with a Langevin thermostat using ASE.

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
from ase import units
from ase.build import bulk
from ase.io import write
from ase.md.langevin import Langevin

from matscipy.calculators.hydrogel import Hydrogel
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
    R0 = np.sqrt(N) * b  # Equilibrium end-to-end distance
    L0 = (N - 1) * b  # Contour length
    rc = 3 * R0  # Cutoff radius

    print("Hydrogel parameters:")
    print(f"  N = {N} monomers")
    print(f"  b = {b} (Kuhn length)")
    print(f"  R0 = {R0:.4f} (equilibrium distance)")
    print(f"  L0 = {L0:.4f} (contour length)")
    print(f"  rc = {rc:.4f} (cutoff)")

    # Create the hydrogel
    atoms, molecules = create_diamond_hydrogel(N=N, b=b, n_cells=2)

    print("\nSystem:")
    print(f"  {len(atoms)} crosslinkers")
    print(f"  {len(molecules.bonds)} bonds")

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

    # Initial energy
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    max_force = np.max(np.abs(forces))

    print("\nInitial state:")
    print(f"  Energy = {energy:.4f}")
    print(f"  Max force = {max_force:.4f}")

    # Minimize energy
    from ase.optimize import FIRE

    optimizer = FIRE(atoms)
    optimizer.run(fmax=0.01, steps=100)

    energy = atoms.get_potential_energy()
    print("\nAfter minimization:")
    print(f"  Energy = {energy:.4f}")

    # Save initial structure
    write("hydrogel_initial.xyz", atoms)

    # Set up Langevin dynamics
    # In LJ units, kT = 1 corresponds to T = 1
    T = 0.1  # Temperature (kT units)
    dt = 0.005  # Timestep
    friction = 0.01  # Friction coefficient

    # Initialize velocities
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    MaxwellBoltzmannDistribution(atoms, temperature_K=T / units.kB)

    # Create Langevin dynamics object
    dyn = Langevin(
        atoms, timestep=dt * units.fs, temperature_K=T / units.kB, friction=friction
    )

    # Set up trajectory output
    traj = []

    def save_state():
        traj.append(atoms.copy())

    # Equilibration
    print("\nStarting equilibration...")
    n_equil = 1000
    for i in range(n_equil // 100):
        dyn.run(100)
        save_state()
        if (i + 1) % 10 == 0:
            energy = atoms.get_potential_energy()
            temp = atoms.get_kinetic_energy() / (1.5 * len(atoms))
            print(f"  Step {(i+1)*100}: E = {energy:.4f}, T = {temp:.4f}")

    # Production run
    print("\nStarting production run...")
    n_prod = 5000
    energies = []
    temperatures = []

    for i in range(n_prod // 100):
        dyn.run(100)
        save_state()
        energy = atoms.get_potential_energy()
        temp = atoms.get_kinetic_energy() / (1.5 * len(atoms))
        energies.append(energy)
        temperatures.append(temp)

        if (i + 1) % 10 == 0:
            print(f"  Step {n_equil + (i+1)*100}: E = {energy:.4f}, T = {temp:.4f}")

    # Save trajectory
    write("hydrogel_trajectory.xyz", traj)

    # Statistics
    print("\nSimulation statistics:")
    print(f"  Mean energy = {np.mean(energies):.4f} +/- {np.std(energies):.4f}")
    print(
        f"  Mean temperature = {np.mean(temperatures):.4f} +/- "
        f"{np.std(temperatures):.4f}"
    )

    # Save final structure
    write("hydrogel_final.xyz", atoms)
    print("\nSimulation complete. Files written:")
    print("  hydrogel_initial.xyz")
    print("  hydrogel_trajectory.xyz")
    print("  hydrogel_final.xyz")


if __name__ == "__main__":
    main()
