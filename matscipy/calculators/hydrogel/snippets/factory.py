
from matscipy.calculators.hydrogel import Hydrogel
from matscipy.elasticity import Voigt_6x6_to_cubic, fit_elastic_constants
from matscipy.molecules import Molecules
from matscipy.neighbours import neighbour_list
import numpy as np
from ase.build import bulk
from ase.build import graphene


def create_diamond_lattice(crosslink_spacing, n_cells=2):
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

def create_graphite_lattice(crosslink_spacing, n_cells=2, truly_2d=True, vacuum=10.0):
    """Create a 2D graphene (graphite monolayer) lattice.

    Parameters
    ----------
    crosslink_spacing : float
        Distance between crosslinkers (nearest neighbor distance in graphene)
    n_cells : int
        Number of unit cells in each direction (x, y)
    truly_2d : bool
        If True, set periodic boundary conditions only in x,y (not z)
        If False, use 3D periodic with vacuum in z-direction
    vacuum : float
        Vacuum space in z-direction (Angstrom), only used if truly_2d=False

    Returns
    -------
    atoms : ase.Atoms
        ASE Atoms object with the 2D graphene structure
    molecules : matscipy.molecules.Molecules
        Bond topology
    """

    # Create graphene sheet with the desired nearest neighbor distance
    # In graphene, the lattice parameter 'a' relates to nearest neighbor distance as: nn_dist = a / sqrt(3)
    # So a = crosslink_spacing * sqrt(3)
    a = crosslink_spacing * np.sqrt(3)
    
    if truly_2d:
        # Create truly 2D system with no vacuum and 2D periodic boundary conditions
        atoms = graphene(a=a, size=(n_cells, n_cells, 1), vacuum=0.0)
        
        # Set periodic boundary conditions: True for x,y and False for z
        atoms.pbc = [True, True, False]
        
        # Make the z-dimension of the cell very small (just enough to contain atoms)
        cell = atoms.cell.copy()
        cell[2, 2] = 1.0  # Set z-dimension to 1 Angstrom
        atoms.set_cell(cell)
        
    else:
        # Create 3D periodic system with vacuum in z-direction
        atoms = graphene(a=a, size=(n_cells, n_cells, 1), vacuum=vacuum)

    # Set masses (LJ units)
    atoms.set_masses(np.ones(len(atoms)))

    # Create bonds between nearest neighbors
    # Nearest neighbor distance in graphene
    nn_dist = crosslink_spacing
    i_p, j_p = neighbour_list("ij", atoms, nn_dist * 1.1)

    # Keep only unique bonds (i < j)
    mask = i_p < j_p
    bonds = np.column_stack([i_p[mask], j_p[mask]])

    molecules = Molecules(bonds_connectivity=bonds)

    return atoms, molecules