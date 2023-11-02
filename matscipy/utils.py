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
        cell_builder = constructors[crystalstructure.lower()]
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
            
        unit_cell = cell_builder(symbol, directions=axes.tolist(),
                                 pbc=pbc,
                                 latticeconstant=alat)
    elif type(a) == Atoms:
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
