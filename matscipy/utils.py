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


def line_intersect_2D(p1, p2, x1, x2):
    '''
    Test if 2D finite line defined by points p1 & p2 intersects with the finite line defined by x1 & x2.
    Essentially a Python conversion of the ray casting algorithm suggested in: 
    https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon

    Returns True if lines intersect (including if they are collinear)

    p1, p2, x1, x2: np.array
        2D points defining lines
    '''

    # Convert from pointwise p1, p2 form to ax + by + c = 0 form
    a_p = p2[0] - p1[0]
    b_p = p1[1] - p2[1]
    c_p = (p2[0] * p1[1]) - (p1[0] * p2[1])

    # Sub in x1, x2
    d1 = a_p * x1[0] + b_p * x1[1] + c_p
    d2 = a_p * x2[0] + b_p * x2[1] + c_p

    # Both points above the line
    if (d1 > 0 and d2 > 0) return False
    # Both points below the line
    if (d1 < 0 and d2 < 0) return False

    # One point above, one point below the *infinite* line defined by p1, p2
    # Instead of testing for finite line, switch p <-> x and repeat
    # (Both finite lines intersect if both sets of points intersect the infinite
    # lines defined by the other set of points)

    # Convert from pointwise x1, x2 form to ax + by + c = 0 form
    a_x = x2[0] - x1[0]
    b_x = x1[1] - x2[1]
    c_x = (x2[0] * x1[1]) - (x1[0] * x2[1])

    # Sub in p1, p2
    d1 = a_x * p1[0] + b_x * p1[1] + c_x
    d2 = a_x * p2[0] + b_x * p2[1] + c_x

    # Both points above the line
    if (d1 > 0 and d2 > 0) return False
    # Both points below the line
    if (d1 < 0 and d2 < 0) return False

    # 2 possibilities left: the lines intersect at a single point, or the lines are collinear,
    # both result in an intersection, so return True
    return True

def points_in_polygon2D(p, poly_points):
    '''
    Test if points lies within the closed 2D polygon defined by poly_points
    Uses ray casting algorithm, as suggested by:
    https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon

    p: np.array
        (n, 2) array of points to test
    poly_points: np.array
        Points defining the 2D polygon

    Returns
    -------
    mask: np.array
        Boolean mask. True for points inside the poylgon
    '''
    if len(p.shape) == 1:
        # Single point provided
        points = p[np.newaxis, :]
    else:
        points = p

    # Ensure polygon is closed
    if poly_points[0, :] != poly_points[-1, :]:
        poly_points = np.append(poly_points, poly_points[0, :], axis=0)



    npoints = points.shape[0]

    mask = np.zeros(npoints, dtype=bool)

    # Get point that is definitely outside the polygon
    test_point = np.array([2 * np.max(poly_points[:, 0]), 2 * np.max(poly_points[:, 1])])

    for i in range(npoints):
        intersections = 0
        for j in range(poly_points.shape[0] - 1):
            intersections += line_intersect2D(poly_points[j, :], poly_points[j+1, :], test_point, points[i, :])

        if intersections %% 2:
            # Even number of intersections, point is inside polygon
            mask[i] = True
    return mask
