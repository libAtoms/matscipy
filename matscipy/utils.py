import warnings
import numpy as np

def validate_cubic_cell(a, symbol="w", axes=None, crystalstructure=None, pbc=True):
    '''
    Provide uniform interface for generating rotated atoms objects through two main methods:

    For lattice constant + symbol + crystalstructure, build a valid cubic bulk rotated into the frame defined by axes
    For cubic bulk atoms object + crystalstructure, validate structure matches expected cubic bulk geometry, and rotate to frame defined by axes

    a: float or ase.atoms
        EITHER lattice constant (in A), or the cubic bulk structure
    symbol: str
        Elemental symbol, passed to relevant crystalstructure generator when a is float
    axes: np.array
        Axes transform to apply to bulk cell
    crystalstructure: str
        Base Structure of bulk system, currently supported: fcc, bcc, diamond
    pbc: list of bool
        Periodic Boundary Conditions in x, y, z
    '''

    from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic, Diamond
    from ase.atoms import Atoms
    from ase.build import cut, rotate
    from warnings import warn

    constructors = {
        "fcc": FaceCenteredCubic,
        "bcc": BodyCenteredCubic,
        "diamond": Diamond
    }

    # Choose correct ase.lattice.cubic constructor given crystalstructure
    try:
        if crystalstructure is not None:
            cell_builder = constructors[crystalstructure.lower()]
        else:
            cell_builder = None
    except (KeyError, AttributeError) as e:
        if type(e) == KeyError:
            # KeyError when crystalstructure not a key in constructors
            raise ValueError(f"Crystal Structure {crystalstructure} not in accepted list: {list(constructors.keys())}")
        else:
            # AttributeError when type(crystalstructure) doesn't support .lower() (not a valid input type)
            raise TypeError(f"crystalstructure should be one of {constructors.keys()}")

    if np.issubdtype(type(a), np.floating) or np.issubdtype(type(a), np.integer):
        # Reproduce legacy behaviour with a==alat
        alat = a

        if cell_builder is None:
            raise AssertionError("crystalstructure must be given when 'a' argument is a lattice parameter")
            
        unit_cell = cell_builder(symbol, directions=axes.tolist(),
                                 pbc=pbc,
                                 latticeconstant=alat)
    elif isinstance(a, Atoms):
        # New behaviour for arbitrary cubic unit cells (Zincblende, L12, ...)
        alat = a.cell[0, 0]

        ats = a.copy()

        if crystalstructure is not None:
            # Try to validate that "a" matches expected structure given by crystalstructure
            tol = 1e-3
            ref_ats = cell_builder("C", directions=np.eye(3).astype(int).tolist(),
                                latticeconstant=alat)

            # Check that number of atoms in ats is an integer multiple of ref_ats
            # (ats has to be the cubic primitive, or a supercell of the cubic primitive)
            # Also check that the structure is cubic
            rel_size = len(ats) / len(ref_ats)
            skip_fractional_check = False
            try:
                # Integer multiple of ats test
                assert abs(rel_size - np.floor(rel_size)) < tol

                # Cubic cell test: cell = a*I_3x3
                assert np.allclose(ats.cell[:, :], np.eye(3) * ats.cell[0, 0], atol=tol)
            except AssertionError:
                # Test failed, skip next test + warn user
                skip_fractional_check = True
                
            # Check fractional coords match expectation
            frac_match = True
            sup_size = int(rel_size**(1/3))
            if not skip_fractional_check:
                ref_supercell = ref_ats * (sup_size, sup_size, sup_size)

                try:
                    apos = np.sort(ats.get_scaled_positions(), axis=0)
                    rpos = np.sort(ref_supercell.get_scaled_positions(), axis=0)
                    assert np.allclose(apos, rpos, atol=tol)
                except:
                    frac_match = False

            if not frac_match or skip_fractional_check:
                warn(f"Input bulk does not appear to match bulk {crystalstructure}.", stacklevel=2)

            alat /= sup_size # Account for larger supercells having multiple internal core sites

        ats = cut(ats, a=axes[0, :], b=axes[1, :], c=axes[2, :])
        rotate(ats, ats.cell[0, :].copy(), [1, 0, 0], ats.cell[1, :].copy(), [0, 1, 0])
        unit_cell = ats.copy()

        unit_cell.set_pbc(pbc)
    
    else:
        # "a" variable not valid
        raise TypeError("type(a) is not a float, or an Atoms object")

    return alat, unit_cell

def complete_basis(v1, v2=None, normalise=False, nmax=5, tol=1E-6):
    '''
    Generate a complete (v1, v2, v3) orthogonal basis in 3D from v1 and an optional v2

    (V1, V2, V3) is always right-handed.

    v1: np.array
        len(3) array giving primary axis
    v2: np.array | None
        len(3) array giving secondary axis
        If None, a search for a "sensible" secondary axis will occur, raising a RuntimeError if unsuccessful
         (Searches for vectors perpendicular to v1 with small integer indices)
        If v2 is given and is not orthogonal to v1, raises a warning
    normalise: bool
        return an orthonormal basis, rather than just orthogonal
    nmax: int
        Maximum integer index for v2 search
    tol: float
        Tolerance for checking orthogonality between v1 and v2

    Returns
    -------
    V1, V2, V3: np.arrays
        Complete orthogonal basis, optionally normalised
    '''

    def _v2_search(v1, nmax, tol):
        for i in range(nmax):
            for j in range(-i-1, i+1):
                for k in range(-j-1, j+1):
                    if i==j and j==k and k==0:
                        continue
                    # Try all permutations
                    test_vec = np.array([i, j, k])

                    if np.abs(np.dot(v1, test_vec)) < tol:
                        return test_vec
                    
                    test_vec = np.array([k, i, j])

                    if np.abs(np.dot(v1, test_vec)) < tol:
                        return test_vec
                    
                    test_vec = np.array([j, k, i])

                    if np.abs(np.dot(v1, test_vec)) < tol:
                        return test_vec
                    # No nice integer vector found!
                    raise RuntimeError(f"Could not automatically find an integer basis from basis vector {v1}")

    V1 = np.array(v1).copy().astype(int)

    if v2 is None:
        V2 = _v2_search(v1, nmax, tol)
        V2 *= np.sign(V2[0])
        V2 = V2.astype(int)
    else:
        V2 = np.array(v2).copy().astype(int)

    if np.abs(np.dot(V1, V2)) >= tol:
        # Vector basis is not orthogonal
        msg = f"V2 vector {V2} is not orthogonal to V1 vector {V1}; dot(V1, V2) = {float(np.dot(V1, V2))}"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    
    V3 = np.cross(V1, V2)

    if normalise:
        V1 = V1.astype(np.float64) / np.linalg.norm(V1)
        V2 = V2.astype(np.float64) / np.linalg.norm(V2)
        V3 = V3.astype(np.float64) / np.linalg.norm(V3)
    else:
        # Attempt to improve V3
        # Try to find integer vector in same direction
        # with smaller magnitude
        non_zero = np.abs(V3[V3!=0])
        gcd = np.gcd.reduce(non_zero)
        V3 = V3.astype(int)/ int(gcd)
        V3 = V3.astype(int)

    # If we made a V2 ourselves, enforce a Right-handed coordinate system
    chirality = np.linalg.det(np.array([V1, V2, V3]))

    if chirality < 0:
        # Left Handed coord system
        if v2 is None:
            # We founds V2 ourselves, so is safe to swap
            V3, V2 = V2, V3
        else:
            # V2 was specified, invert V3
            V3 = -V3
    
    return V1, V2, V3


# Get the results of Ovito Common Neighbor Analysis 
# https://www.ovito.org/docs/current/reference/pipelines/modifiers/common_neighbor_analysis.html
# and Identify Diamond modifier
# https://www.ovito.org/docs/current/reference/pipelines/modifiers/identify_diamond.html
# for better visualisation of the dislocation core
# it will be identified as "other" structure type
def get_structure_types(structure, diamond_structure=False):
    """Get the results of Common Neighbor Analysis and 
        Identify Diamond modifiers from Ovito
        (Requires Ovito python module)
    Args:
        structure (ase.atoms): input structure
    Returns:
        atom_labels (array of ints): per atom labels of the structure types
        structure_names (list of strings): names of the structure types
        colors (list of strings): colors of the structure types in hex format
    """
    from ovito.io.ase import ase_to_ovito
    from ovito.modifiers import CommonNeighborAnalysisModifier, IdentifyDiamondModifier
    from ovito.pipeline import StaticSource, Pipeline
    ovito_structure = structure.copy()
    if "fix_mask" in ovito_structure.arrays:
        del ovito_structure.arrays["fix_mask"]
    
    if diamond_structure:
        modifier = IdentifyDiamondModifier()
    else:
        modifier = CommonNeighborAnalysisModifier() 
    
    data = ase_to_ovito(ovito_structure)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(modifier)
    data = pipeline.compute()

    atom_labels = data.particles['Structure Type'].array

    structure_names = [structure.name for structure in modifier.structures]
    colors = [structure.color for structure in modifier.structures]
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in colors] 

    return atom_labels, structure_names, hex_colors