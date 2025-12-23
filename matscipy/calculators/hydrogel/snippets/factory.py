
from matscipy.calculators.hydrogel import Hydrogel
from matscipy.elasticity import Voigt_6x6_to_cubic, fit_elastic_constants
from matscipy.molecules import Molecules
from matscipy.neighbours import neighbour_list

def create_diamond_hydrogel(crosslink_spacing, n_cells=2):
    """Create a diamond lattice hydrogel.

    Parameters
    ----------
    crosslink_spacing : float
        Distance between crosslinkers
    n_cells : int
        Number of unit cells in each direction

    Returns
    -------
    atoms : ase.Atoms
        ASE Atoms object with the hydrogel structure
    molecules : matscipy.molecules.Molecules
        Bond topology
    """


    # Diamond lattice constant (nearest neighbor distance = a*sqrt(3)/4)
    # We want nearest neighbor distance = R0
    a = crosslink_spacing * 4 / np.sqrt(3)

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