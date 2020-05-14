# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ======================================================================

import itertools

import numpy as np

from ase.data import atomic_numbers

import _matscipy
from _matscipy import first_neighbours

###

def mic(dr, cell, pbc=None):
    """
    Apply minimum image convention to an array of distance vectors.

    Parameters
    ----------
    dr : array_like
        Array of distance vectors.
    cell : array_like
        Simulation cell.
    pbc : array_like, optional
        Periodic boundary conditions in x-, y- and z-direction. Default is to
        assume periodic boundaries in all directions.

    Returns
    -------
    dr : array
        Array of distance vectors, wrapped according to the minimum image
        convention.
    """
    # Check where distance larger than 1/2 cell. Particles have crossed
    # periodic boundaries then and need to be unwrapped.
    rec = np.linalg.inv(cell)
    if pbc is not None:
        rec *= np.array(pbc, dtype=int).reshape(3,1)
    dri = np.round(np.dot(dr, rec))

    # Unwrap
    return dr - np.dot(dri, cell)


def neighbour_list(quantities, atoms=None, cutoff=None, positions=None,
                   cell=None, pbc=None, numbers=None, cell_origin=None):
    """
    Compute a neighbor list for an atomic configuration. Atoms outside periodic
    boundaries are mapped into the box. Atoms outside nonperiodic boundaries
    are included in the neighbor list but the complexity of neighbor list search
    for those can become n^2.

    The neighbor list is sorted by first atom index 'i', but not by second 
    atom index 'j'.

    The neighbour list accepts either an ASE Atoms object or positions and cell
    vectors individually.

    Parameters
    ----------
    quantities : str
        Quantities to compute by the neighbor list algorithm. Each character
        in this string defines a quantity. They are returned in a tuple of
        the same order. Possible quantities are
            'i' : first atom index
            'j' : second atom index
            'd' : absolute distance
            'D' : distance vector
            'S' : shift vector (number of cell boundaries crossed by the bond
                  between atom i and j). With the shift vector S, the
                  distances D between atoms can be computed from:
                  D = a.positions[j]-a.positions[i]+S.dot(a.cell)
    atoms : ase.Atoms
        Atomic configuration. (Default: None)
    cutoff : float or dict
        Cutoff for neighbor search. It can be
            - A single float: This is a global cutoff for all elements.
            - A dictionary: This specifies cutoff values for element
              pairs. Specification accepts element numbers of symbols.
              Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
            - A list/array with a per atom value: This specifies the radius of
              an atomic sphere for each atoms. If spheres overlap, atoms are
              within each others neighborhood.
    positions : array_like
        Atomic positions. (Default: None)
    cell : array_like
        Cell vectors as a 3x3 matrix. (Default: Shrink wrapped cell)
    pbc : array_like
        3-vector containing periodic boundary conditions in all three
        directions. (Default: Nonperiodic box)
    numbers : array_like
        Array containing the atomic numbers.

    Returns
    -------
    i, j, ... : array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a), but the order of (i,j)
        pairs is not guaranteed.

    Examples
    --------
    Examples assume Atoms object *a* and numpy imported as *np*.
    1. Coordination counting:
        i = neighbor_list('i', a, 1.85)
        coord = np.bincount(i)

    2. Coordination counting with different cutoffs for each pair of species
        i = neighbor_list('i', a,
                           {('H', 'H'): 1.1, ('C', 'H'): 1.3, ('C', 'C'): 1.85})
        coord = np.bincount(i)

    3. Pair distribution function:
        d = neighbor_list('d', a, 10.00)
        h, bin_edges = np.histogram(d, bins=100)
        pdf = h/(4*np.pi/3*(bin_edges[1:]**3 - bin_edges[:-1]**3)) * a.get_volume()/len(a)

    4. Pair potential:
        i, j, d, D = neighbor_list('ijdD', a, 5.0)
        energy = (-C/d**6).sum()
        pair_forces = (6*C/d**5  * (D/d).T).T
        forces_x = np.bincount(j, weights=pair_forces[:, 0], minlength=len(a)) - \
                   np.bincount(i, weights=pair_forces[:, 0], minlength=len(a))
        forces_y = np.bincount(j, weights=pair_forces[:, 1], minlength=len(a)) - \
                   np.bincount(i, weights=pair_forces[:, 1], minlength=len(a))
        forces_z = np.bincount(j, weights=pair_forces[:, 2], minlength=len(a)) - \
                   np.bincount(i, weights=pair_forces[:, 2], minlength=len(a))

    5. Dynamical matrix for a pair potential stored in a block sparse format:
        from scipy.sparse import bsr_matrix
        i, j, dr, abs_dr = neighbor_list('ijDd', atoms)
        energy = (dr.T / abs_dr).T
        dynmat = -(dde * (energy.reshape(-1, 3, 1) * energy.reshape(-1, 1, 3)).T).T \
                 -(de / abs_dr * (np.eye(3, dtype=energy.dtype) - \
                   (energy.reshape(-1, 3, 1) * energy.reshape(-1, 1, 3))).T).T
        dynmat_bsr = bsr_matrix((dynmat, j, first_i), shape=(3*len(a), 3*len(a)))

        dynmat_diag = np.empty((len(a), 3, 3))
        for x in range(3):
            for y in range(3):
                dynmat_diag[:, x, y] = -np.bincount(i, weights=dynmat[:, x, y])

        dynmat_bsr += bsr_matrix((dynmat_diag, np.arange(len(a)),
                                  np.arange(len(a) + 1)),
                                 shape=(3 * len(a), 3 * len(a)))


  i_n, j_n, dr_nc, abs_dr_n = neighbour_list('ijDd', atoms, dict)
e_nc = (dr_nc.T/abs_dr_n).T
    D_ncc = -(dde_n * (e_nc.reshape(-1,3,1) * e_nc.reshape(-1,1,3)).T).T
    D_ncc += -(de_n/abs_dr_n * (np.eye(3, dtype=e_nc.dtype) - (e_nc.reshape(-1,3,1) * e_nc.reshape(-1,1,3))).T).T

    D = bsr_matrix((D_ncc, j_n, first_i), shape=(3*nat,3*nat))

    Ddiag_icc = np.empty((nat,3,3))
    for x in range(3):
        for y in range(3):
            Ddiag_icc[:,x,y] = -np.bincount(i_n, weights = D_ncc[:,x,y])

    D += bsr_matrix((Ddiag_icc,np.arange(nat),np.arange(nat+1)), shape=(3*nat,3*nat))

    return D
    """

    if cutoff is None:
        raise ValueError('Please provide a value for the cutoff radius.')

    if atoms is None:
        if positions is None:
            raise ValueError('You provided neither an ASE Atoms object nor '
                             'a positions array.') 
        if cell is None:
            # Shrink wrapped cell
            rmin = np.min(positions, axis=0)
            rmax = np.max(positions, axis=0)
            cell_origin = rmin
            cell = np.diag(rmax - rmin)
        if cell_origin is None:
            cell_origin = np.zeros(3)
        if pbc is None:
            pbc = np.zeros(3, dtype=bool)
        if numbers is None:
            numbers = np.ones(len(positions), dtype=np.int32)
    else:
        if positions is not None:
            raise ValueError('You cannot provide an ASE Atoms object and '
                             'individual position atomic positions at the same '
                             'time.')
        positions = atoms.positions
        if cell_origin is not None:
            raise ValueError('You cannot provide an ASE Atoms object and '
                             'a cell origin at the same time.')
        cell_origin = np.zeros(3)
        if cell is not None:
            raise ValueError('You cannot provide an ASE Atoms object and '
                             'cell vectors at the same time.')
        cell = atoms.cell
        if pbc is not None:
            raise ValueError('You cannot provide an ASE Atoms object and '
                             'separate periodicity information at the same '
                             'time.')
        pbc = atoms.pbc
        if numbers is not None:
            raise ValueError('You cannot provide an ASE Atoms object and '
                             'separate atomic numbers at the same time.')
        numbers = atoms.numbers.astype(np.int32)

    if isinstance(cutoff, dict):
        maxel = np.max(numbers)
        _cutoff = np.zeros([maxel+1, maxel+1], dtype=float)
        for (el1, el2), c in cutoff.items():
            try:
                el1 = atomic_numbers[el1]
            except:
                pass
            try:
                el2 = atomic_numbers[el2]
            except:
                pass
            if el1 < maxel+1 and el2 < maxel+1:
                _cutoff[el1, el2] = c
                _cutoff[el2, el1] = c
    else:
        _cutoff = cutoff

    return _matscipy.neighbour_list(quantities, cell_origin, cell,
                                    np.linalg.inv(cell.T), pbc, positions,
                                    _cutoff, numbers)


def find_indices_of_reversed_pairs(i_n, j_n, abs_dr_n):
    """Find neighbor list indices where reversed pairs are stored

    Given list of identifiers of neighbor atoms `i_n` and `j_n`,
    determines the list of indices `reverse` into the neighbor list
    where each pair is reversed, i.e. `i_n[reverse[n]]=j_n[n]` and
    `j_n[reverse[n]]=i_n[n]` for each index `n` in the neighbor list
    
    In the case of small periodic systems, one needs to be careful, because
    the same pair may appear more than one time, with different pair
    distances. Therefore, the pair distance must be taken into account.

    We assume that there is in fact one reversed pair for every pair.
    However, we do not check this assumption in order to avoid overhead.

    Parameters
    ----------
    i_n : array_like
       array of atom identifiers
    j_n : array_like
       array of atom identifiers
    abs_dr_n : array_like
        pair distances

    Returns
    -------
    reverse : numpy.ndarray
        array of indices into i_n and j_n
    """
    sorted_1 = np.lexsort(keys=(abs_dr_n, i_n, j_n))
    sorted_2 = np.lexsort(keys=(abs_dr_n, j_n, i_n))
    #np.testing.assert_equal(j_n[sorted_1], i_n[sorted_2])
    #np.testing.assert_equal(i_n[sorted_1], j_n[sorted_2])
    #np.testing.assert_equal(abs_dr_n[sorted_1], abs_dr_n[sorted_2])
    #print(np.c_[i_n[sorted_2], j_n[sorted_2], abs_dr_n[sorted_2], 
    #            i_n[sorted_1], j_n[sorted_1], abs_dr_n[sorted_1]])
    tmp2 = np.arange(i_n.size)[sorted_2]
    tmp1 = np.arange(i_n.size)[sorted_1]
    # np.arange(i_n.size) are indices into the neighbor list, so
    #  * the nth element in tmp1 is the index where i,j was before reordering with sorted_1
    #  * the nth element in tmp2 is the index where j,i was before reordering with sorted_2
    reverse = np.empty(i_n.size, dtype=i_n.dtype)
    reverse[tmp1] = tmp2
    return reverse


def find_common_neighbours(i_n, j_n,  nat):
    """Find common neighbors of pairs of atoms

    For each pair ``(i1, j1)`` in the neighbor list, find all other pairs
    ``(i2, j1)`` which share the same ``j1``. This includes ``(i1,j1)``
    itself. In this way, create a list with ``n`` blocks of rows, where ``n``
    is the length of the neighbor list. All rows in a block have the same
    ``j1``. Each row corresponds to one triplet ``(i1, j1 ,i2)``. The number
    of rows in the block is equal to the total number of neighbors of ``j1``.

    Parameters
    ----------
    i_n : array_like
       array of atom identifiers
    j_n : array_like
       array of atom identifiers
    nat: int
        number of atoms

    Returns
    -------
    cnl_i1_i2: array
        atom numbers i1 and i2
    cnl_j1: array
        shared neighbor of i1 and i2
    nl_index_i1_j1: array
        index in the neighbor list of pair i1, j1
    nl_index_i2_j1: array
        index in the neighbor list of pair i2, j1

    Examples
    --------

    Accumulate random numbers for pairs with common neighbors:

    >>> import numpy as np
    >>> import matscipy
    >>> from ase.lattice.cubic import FaceCenteredCubic
    >>> from matscipy.neighbours import neighbour_list, find_common_neighbours
    >>> cutoff = 6.5
    >>> atoms = FaceCenteredCubic('Cu', size=[4, 4, 4])
    >>> nat = len(atoms.numbers)
    >>> print(nat)
    256
    >>> i_n, j_n, dr_nc, abs_dr_n = neighbour_list('ijDd', atoms, cutoff)
    >>> print(i_n.shape)
    (22016,)
    >>> cnl_i1_i2, cnl_j1, nl_index_i1_j1, nl_index_i2_j1 = find_common_neighbours(i_n, j_n, nat)
    >>> print(cnl_i1_i2.shape)
    (1893376, 2)
    >>> unique_pairs_i1_i2, bincount_bins = np.unique(cnl_i1_i2, axis=0, return_inverse=True) 
    >>> print(unique_pairs_i1_i2.shape)
    (65536, 2)
    >>> tmp = np.random.rand(cnl_i1_i2.shape[0])
    >>> my_sum = np.bincount(bincount_bins, weights=tmp, minlength=unique_pairs_i1_i2.shape[0]) 
    >>> print(my_sum.shape)
    (65536,)

    """
    # Create a copy of the neighbor list which is sorted by j_n, e.g.
    # +---------------+    +---------------+
    # | sorted by i_n |    | sorted by j_n |
    # +=======+=======+    +=======+=======+
    # | i_n   | j_n   |    | i_n   | j_n   |
    # +-------+-------+    +-------+-------+
    # | 1     | 2     |    | 2     | 1     |
    # +-------+-------+    +-------+-------+
    # | 1     | 95    |    | 4     | 1     |
    # +-------+-------+    +-------+-------+
    # | 2     | 51    |    | 81    | 2     |
    # +-------+-------+    +-------+-------+
    # | 2     | 99    |    | 12    | 2     |
    # +-------+-------+    +-------+-------+
    # | 2     | 1     |    | 6     | 2     |
    # +-------+-------+    +-------+-------+
    # | 3     | 78    |    | 143   | 3     |
    # +-------+-------+    +-------+-------+
    # | ...   | ...   |    | ...   | ...   |
    # +-------+-------+    +-------+-------+
    j_order = np.argsort(j_n)
    i_n_2 = i_n[j_order]
    j_n_2 = j_n[j_order]
    # Find indices in the copy where contiguous blocks with same j_n_2 start
    first_j = first_neighbours(nat, j_n_2) 
    num_rows_per_j = first_j[j_n+1] - first_j[j_n] 
    num_rows_cnl = np.sum(num_rows_per_j)

    # The common neighbor information could be stored as
    # a 2D array. However, multiple 1D arrays are likely
    # better for performance (fewer cache misses later).
    nl_index_i1_j1 = np.empty(num_rows_cnl, dtype=i_n.dtype)
    cnl_j1 = np.empty(num_rows_cnl, dtype=i_n.dtype)
    nl_index_i2_j1 = np.empty(num_rows_cnl, dtype=i_n.dtype)
    cnl_i1_i2 = np.empty((num_rows_cnl, 2), dtype=i_n.dtype)

    block_start = np.r_[0, np.cumsum(num_rows_per_j)]
    slice_for_j1 = {j1: slice(first_j[j1], first_j[j1+1]) for j1 in np.arange(nat)}
    for block_number, (i1, j1) in enumerate(zip(i_n, j_n)):
        slice1 = slice(block_start[block_number], block_start[block_number+1])
        slice2 = slice_for_j1[j1] 
        nl_index_i1_j1[slice1] = block_number
        cnl_j1[slice1] = j1 
        nl_index_i2_j1[slice1] = j_order[slice2]
        cnl_i1_i2[slice1, 0] = i1
        cnl_i1_i2[slice1, 1] = i_n_2[slice2]
    return cnl_i1_i2, cnl_j1, nl_index_i1_j1, nl_index_i2_j1


