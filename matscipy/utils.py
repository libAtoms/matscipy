import functools
import warnings
import numpy as np
from ase.utils.structure_comparator import SymmetryEquivalenceCheck


class classproperty:
    '''
    Decorator class to replace classmethod property decorators
    '''
    def __init__(self, method):
        self.method = method
        functools.update_wrapper(self, method)
    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self.method(cls)


def validate_cubic_cell(a, symbol="w", axes=None, crystalstructure=None, pbc=True):
    """
    Provide uniform interface for generating rotated atoms objects through two main methods:

    For lattice constant + symbol + crystalstructure, build a valid cubic bulk rotated into the frame defined by axes
    For cubic bulk atoms object + crystalstructure, validate structure matches expected cubic bulk geometry, and rotate to frame defined by axes
    Also, if cubic atoms object is supplied, search for a more condensed representation than is provided by ase.build.cut

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
    """

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
        
        # cut and rotate can result in a supercell structure. Attempt to find smaller repr 
        ats = find_condensed_repr(ats)
        
        unit_cell = ats.copy()

        unit_cell.set_pbc(pbc)
    
    else:
        # "a" variable not valid
        raise TypeError("type(a) is not a float, or an Atoms object")

    return alat, unit_cell


def find_condensed_repr(atoms, precision=2):
    """
    Search for a condensed representation of atoms
    Equivalent to attempting to undo a supercell

    atoms: ase Atoms object
        structure to condense
    precision: int
        Number of decimal places to use in determining whether scaled positions are equal

    returns a condensed copy of atoms, if such a condensed representation is found. Else returns a copy of atoms
    """
    ats = atoms.copy()
    for axis in range(2, -1, -1):
        ats = find_condensed_repr_along_axis(ats, axis, precision)

    return ats

def find_condensed_repr_along_axis(atoms, axis=-1, precision=2):
    """
    Search for a condensed representation of atoms about axis. 
    Essentially an inverse to taking a supercell.

    atoms: ase Atoms object
        structure to condense
    axis: int
        axis about which to find a condensed repr
        axis=-1 essentially will invert a (1, 1, x) supercell
    precision: int
        Number of decimal places to use in determining whether scaled positions are equal

    returns a condensed copy of atoms, if such a condensed representation is found. Else returns a copy of atoms
    """
    comp = SymmetryEquivalenceCheck()

    cart_directions = [
        [1, 2], # axis=0
        [2, 0], # axis=1
        [0, 1] # axis=2
    ]

    dirs = cart_directions[axis]

    ats = atoms.copy()
    ats.wrap()

    # Choose an origin atom, closest to cell origin
    origin_idx = np.argmin(np.linalg.norm(ats.positions, axis=-1))
    
    # Find all atoms which are in line with the origin_idx th atom
    p = np.round(ats.get_scaled_positions(), precision)

    # MIC distance vector from origin atom, in scaled coordinates
    p_diff = (p - p[origin_idx, :]) % 1.0

    matches = np.argwhere((p_diff[:, dirs[0]] < 10**(-precision)) * (p_diff[:, dirs[1]] < 10**(-precision)))[:, 0]
    min_off = np.inf
    ret_struct = ats

    for match in matches:
        if match == origin_idx:
            # skip i=origin_idx
            continue

        # Fractional change in positions
        dz = (p[origin_idx, axis] - p[match, axis]) % 1.0
        # Create a test atoms object and cut cell along in axis
        # Test whether test atoms is equivalent to original structure
        test_ats = ats.copy()
        test_cell = test_ats.cell[:, :].copy()
        test_cell[axis, :] *= dz
        test_ats.set_cell(test_cell)

        sp = np.round(test_ats.get_scaled_positions(wrap=False), precision)

        del_mask = sp[:, axis] > 1.0 - 10**(-precision)
        test_ats = test_ats[~del_mask]

        # Create a supercell of test ats, and see if it matches the original atoms
        test_sup = [1] * 3
        test_sup[axis] = int(1/dz)

        is_equiv = comp.compare(ats.copy(), test_ats * tuple(test_sup))

        if is_equiv:
            if dz < min_off:
                ret_struct = test_ats.copy()
                min_off = dz

    return ret_struct


def complete_basis(v1, v2=None, normalise=False, nmax=5, tol=1E-6):
    """
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
        return an float orthonormal basis, rather than integer orthogonal basis
    nmax: int
        Maximum integer index for v2 search
    tol: float
        Tolerance for checking orthogonality between v1 and v2

    Returns
    -------
    V1, V2, V3: np.arrays
        Complete orthogonal basis, optionally normalised
        dtype of arrays is int with normalise=False, float with normalise=True
    """

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

def line_intersect_2D(p1, p2, x1, x2):
    """
    Test if 2D finite line defined by points p1 & p2 intersects with the finite line defined by x1 & x2.
    Essentially a Python conversion of the ray casting algorithm suggested in: 
    https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon

    Returns True if lines intersect (including if they are collinear)

    p1, p2, x1, x2: np.array
        2D points defining lines
    """

    # Convert from pointwise p1, p2 form to ax + by + c = 0 form
    a_p = p2[1] - p1[1]
    b_p = p1[0] - p2[0]
    c_p = (p2[0] * p1[1]) - (p1[0] * p2[1])

    # Sub in x1, x2
    d1 = (a_p * x1[0]) + (b_p * x1[1]) + c_p
    d2 = (a_p * x2[0]) + (b_p * x2[1]) + c_p

    # Both points above the line
    if (d1 > 0 and d2 > 0):
        return False
    # Both points below the line
    if (d1 < 0 and d2 < 0):
        return False

    # One point above, one point below the *infinite* line defined by p1, p2
    # Instead of testing for finite line, switch p <-> x and repeat
    # (Both finite lines intersect if both sets of points intersect the infinite
    # lines defined by the other set of points)

    # Convert from pointwise x1, x2 form to ax + by + c = 0 form
    a_x = x2[1] - x1[1]
    b_x = x1[0] - x2[0]
    c_x = (x2[0] * x1[1]) - (x1[0] * x2[1])

    # Sub in p1, p2
    d1 = (a_x * p1[0]) + (b_x * p1[1]) + c_x
    d2 = (a_x * p2[0]) + (b_x * p2[1]) + c_x

    # Both points above the line
    if (d1 > 0 and d2 > 0):
        return False
    # Both points below the line
    if (d1 < 0 and d2 < 0): 
        return False

    # 2 possibilities left: the lines intersect at a single point, or the lines are collinear,
    # both result in an intersection, so return True
    return True

def points_in_polygon2D(p, poly_points):
    """
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
    """
    if len(p.shape) == 1:
        # Single point provided
        points = p.copy()[np.newaxis, :]
    else:
        points = p.copy()

    points += 1E-3

    # Ensure polygon is closed
    if np.any(poly_points[0, :] != poly_points[-1, :]):
        poly_points = np.append(poly_points, poly_points[0, :][np.newaxis, :], axis=0)

    npoints = points.shape[0]

    mask = np.zeros(npoints, dtype=bool)

    # Get random point that is definitely outside the polygon
    a, b = 10 + np.random.random(size=2) * 10

    test_point = np.array([a * np.max(poly_points[:, 0]), b * np.max(poly_points[:, 1])])

    for i in range(npoints):
        intersections = 0
        for j in range(poly_points.shape[0] - 1):
            intersections += line_intersect_2D(poly_points[j, :2], poly_points[j+1, :2], test_point, points[i, :2])
        if intersections % 2:
            # Even number of intersections, point is inside polygon
            mask[i] = True
    return mask.astype(bool)

def get_distance_from_polygon2D(test_points:np.array, polygon_points:np.array) -> np.array:
    """
    Get shortest distance between a test point and a polygon defined by polygon_points
        (i.e. the shortest distance between each point and the lines of the polygon)
    
    Uses formula from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points

    test_points: np.array
        2D Points to get distances for
    polygon_points: np.array
        Coordinates defining the points of the polygon
    """

    def get_dist(p, v, w):
        """
        Gets distance between point p, and the line segment defined by v and w
        From https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
        """
        denominator = np.linalg.norm(v - w)

        if denominator == 0.0:
            # point 1 and point 2 are the same, return distance between test_point and poly_point_1
            return np.linalg.norm(p - v)
        
        # Use t on [0, 1] to parameterize moving on the line segment defined by poly points
        t = np.max([0, np.min([1, np.dot(p - v, w - v)/denominator**2])])

        # Closest point on line segment
        projection = v + t * (w - v)

        return np.linalg.norm(p - projection)
        
    distances = np.zeros(test_points.shape[0])

    N = polygon_points.shape[0]
    for i in range(test_points.shape[0]):
        test_point = test_points[i, :]

        distances[i] = min([get_dist(test_point, polygon_points[j, :], polygon_points[(j+1)%N, :]) for j in range(N)])
    
    return distances

def radial_mask_from_polygon2D(test_points:np.array, polygon_points:np.array, radius:float, inner:bool=True) -> np.array:
    """
    Get a boolean mask of all test_points within a radius of any edge of the polygon defined by polygon_points

    test_points: np.array
        2D Points to get mask for
    polygon_points: np.array
        Coordinates defining the points of the polygon
    radius: float
        Radius to use as cutoff for mask
    inner: bool
        Whether test_points inside the polygon should always be masked as True, regardless of radius
    """

    distances = get_distance_from_polygon2D(test_points, polygon_points)

    outer_mask = distances <= radius 

    if inner:
        inner_mask = points_in_polygon2D(test_points, polygon_points)

        full_mask = (inner_mask + outer_mask).astype(bool)
    else:
        full_mask = outer_mask.astype(bool)
    return full_mask