#
# Copyright 2014-2015, 2017-2019, 2021 Lars Pastewka (U. Freiburg)
#           2020 Jonas Oldenstaedt (U. Freiburg)
#           2020 Wolfram G. Nöhring (U. Freiburg)
#           2019 Jan Griesser (U. Freiburg)
#           2015 James Kermode (Warwick U.)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
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
#

import itertools as it
import typing as ts
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import ase
from ase.data import atomic_numbers
from ase.geometry import find_mic

import _matscipy
from _matscipy import first_neighbours, get_jump_indicies  # noqa
from .molecules import Molecules


class Neighbourhood(ABC):
    """Abstract class defining a neighbourhood of atoms (pairs, triplets)."""

    def __init__(self, atom_types=None):
        """Initialize with atoms and optional atom types."""
        self.atom_type = atom_types \
            if atom_types is not None else lambda i: np.asanyarray(i)

    @abstractmethod
    def get_pairs(self, atoms: ase.Atoms, quantities: str, cutoff=None):
        """Return requested data on pairs."""

    @abstractmethod
    def get_triplets(self,
                     atoms: ase.Atoms,
                     quantities: str,
                     neighbours=None,
                     cutoff=None,
                     full_connectivity=False):
        """Return requested data on triplets."""

    @staticmethod
    def make_result(quantities, connectivity, D, d, S,
                    accepted_quantities) -> ts.List:
        """Construct result list."""
        if not set(quantities) <= set(accepted_quantities):
            unknowns = set(quantities) - set(accepted_quantities)
            raise ValueError(f"Unknown requested quantities {unknowns}")

        e_size = connectivity.shape[1]
        quantities_map = {
            idx: connectivity[:, i]
            for i, idx in enumerate("ijk"[:e_size])
        }
        quantities_map.update({'d': d, 'D': D})

        res = [quantities_map[data] for data in quantities]

        if len(res) == 1:
            return res[0]
        return res

    @staticmethod
    def compute_distances(
        atoms: ase.Atoms,
        connectivity: np.ndarray,
        indices: ts.List[int],
    ) -> ts.Tuple[np.ndarray, np.ndarray]:
        """Return distances and vectors for connectivity."""
        n_nuplets = connectivity.shape[0]
        dim = atoms.positions.shape[1]

        positions = [atoms.positions[col] for col in connectivity.T]
        D = np.zeros((n_nuplets, len(indices), dim))
        d = np.zeros((n_nuplets, len(indices)))

        if positions:
            for i, idx in enumerate(indices):
                D[:, i, :], d[:, i] = \
                    find_mic(positions[idx[1]] - positions[idx[0]],
                             atoms.cell, atoms.pbc)

            if connectivity.shape[1] == 3:
                for i, idx in enumerate(indices):
                    D[:, i, :] = \
                        (positions[idx[1]] - positions[idx[0]])
                    d[:, i] = np.linalg.norm(D[:, i], axis=-1)
        return D.squeeze(), d.squeeze()

    def connected_triplets(self, atoms: ase.Atoms, pair_list, triplet_list,
                           nb_pairs):
        i_p, j_p = pair_list
        ij_t, ik_t, jk_t = triplet_list
        first_p = first_neighbours(nb_pairs, ij_t)

        all_ij_pairs = []
        all_ijm_types = []
        all_ijn_types = []

        for pair_im, pair_in in zip(ij_t, ik_t):
            pairs_ij = ik_t[first_p[pair_im]:first_p[pair_im + 1]]
            all_ij_pairs.append(pairs_ij[(pairs_ij != pair_im)
                                         & (pairs_ij != pair_in)])

            all_ijm_types.append(
                self.find_triplet_types(atoms, i_p[pair_im], j_p[pairs_ij],
                                        j_p[pair_im]))

            all_ijn_types.append(
                self.find_triplet_types(atoms, i_p[pair_in], j_p[pairs_ij],
                                        j_p[pair_in]))

        return all_ij_pairs, all_ijm_types, all_ijn_types

    def triplet_to_numbers(self, atoms: ase.Atoms, i, j, k):
        ids = map(np.asarray, (i, j, k))
        max_size = max(map(len, ids))

        full_ids = np.empty((3, max_size), ids[0].dtype)

        for idx, id in enumerate(ids):
            full_ids[idx, :] = id
        return (atoms.numbers[i] for i in full_ids)

    def find_triplet_types(self, atoms: ase.Atoms, i, j, k):
        """Return triplet types from atom ids."""
        return self.triplet_type(*self.triplet_to_numbers(atoms, i, j, k))


class CutoffNeighbourhood(Neighbourhood):
    """Class defining neighbourhood based on proximity."""

    def __init__(self,
                 atom_types=None,
                 pair_types=None,
                 triplet_types=None,
                 cutoff: ts.Union[float, dict] = None):
        """Initialize with atoms, atom types, pair types and cutoff.

        Parameters
        ----------
        atom_types : ArrayLike
            atom types array
        pair_types : function of 2 atom type arrays
            maps 2 atom types array to an array of pair types
        cutoff : float or dict
            Cutoff for neighbor search. It can be
                - A single float: This is a global cutoff for all elements.
                - A dictionary: This specifies cutoff values for element
                pairs. Specification accepts element numbers of symbols.
                Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
                - A list/array with a per atom value: This specifies the radius
                of an atomic sphere for each atoms. If spheres overlap, atoms
                are within each others neighborhood.
        """
        super().__init__(atom_types)
        self.pair_type = (pair_types if pair_types is not None else
                          lambda i, j: np.ones_like(i))
        self.triplet_type = (triplet_types if triplet_types is not None else
                             lambda i, j, k: np.ones_like(i))
        self.cutoff = cutoff

    def get_pairs(self, atoms: ase.Atoms, quantities: str, cutoff=None):
        """Return pairs and quantities from conventional neighbour list."""
        if cutoff is None:
            cutoff = self.cutoff
        return neighbour_list(quantities, atoms, cutoff)

    def get_triplets(self,
                     atoms: ase.Atoms,
                     quantities: str,
                     neighbours=None,
                     cutoff=None):
        """Return triplets and quantities from conventional neighbour list."""
        if cutoff is None:
            cutoff = self.cutoff

        full_connectivity = 'k' in quantities

        if neighbours is None:
            i_p, j_p, d_p, D_p = neighbour_list("ijdD", atoms, cutoff)
        else:
            i_p, j_p, d_p, D_p = neighbours

        first_n = first_neighbours(len(atoms), i_p)

        # Getting all references in pair list
        ij_t, ik_t, jk_t = triplet_list(first_n, d_p, cutoff, i_p, j_p)
        connectivity = np.array([ij_t, ik_t, jk_t]).T

        if full_connectivity and np.any(jk_t == -1):
            raise ValueError("Cutoff is too small for complete "
                             "triplet connectivity")

        D, d = None, None

        # If any distance is requested, compute distances vectors and norms
        # Distances are computed from neighbour list
        if "d" in quantities or "D" in quantities:
            D = np.zeros((len(ij_t), 3, 3))
            D[:, 0] = D_p[ij_t]  # i->j
            D[:, 1] = D_p[ik_t]  # i->k
            D[:, 2] = D[:, 1] - D[:, 0]  # j->k

            d = np.linalg.norm(D, axis=-1)  # distances

        return self.make_result(
            quantities, connectivity, D, d, None, accepted_quantities="ijkdD")


class MolecularNeighbourhood(Neighbourhood):
    """Class defining neighbourhood based on molecular connectivity."""

    def __init__(self, molecules: Molecules, atom_types=None):
        """Initialze with atoms and molecules."""
        super().__init__(atom_types)
        self.molecules = molecules
        self.cutoff = np.inf

    @property
    def molecules(self):
        """Molecules instance that defines neighbourhood."""
        return self._molecules

    @molecules.setter
    def molecules(self, molecules):
        """Create full connectivity when assigning new molecules."""
        self._molecules = molecules

        # Get ij + ji pairs and ijk + kji angles to mimic the cutoff behavior
        self.connectivity = {
            "bonds": self.double_connectivity(molecules.bonds),
            "angles": self.double_connectivity(molecules.angles),
        }

        # Add pairs from the angle connectivity with negative types
        # This way they should be ignored for the pair potentials
        if molecules.angles.size > 0:
            self.complete_connectivity(
                typeoffset=-(np.max(molecules.angles["type"]) + 1))
        else:
            self.triplet_list = np.zeros([0, 3], dtype=np.int32)

    @property
    def pair_type(self):
        """Map atom types to pair types."""
        return lambda ti_p, tj_p: self.connectivity["bonds"]["type"]

    @property
    def triplet_type(self):
        """Map atom types to triplet types."""
        return lambda ti_p, tj_p, tk_p: self.connectivity["angles"]["type"]

    @staticmethod
    def double_connectivity(connectivity: np.ndarray) -> np.ndarray:
        """Sort and stack connectivity + reverse connectivity."""
        c = np.zeros(2 * len(connectivity), dtype=connectivity.dtype)
        c["type"].reshape(2, -1)[:] = connectivity["type"]
        c_fwd, c_bwd = np.split(c["atoms"], 2)
        c_fwd[:] = connectivity["atoms"]
        c_bwd[:] = connectivity["atoms"][:, ::-1]
        return c

    def complete_connectivity(self, typeoffset: int = 0):
        """Add angles to pair connectivity."""
        bonds, angles = self.connectivity["bonds"], self.connectivity["angles"]

        permutations = list(
            it.combinations(range(angles["atoms"].shape[1]), 2))
        e = len(permutations)
        n, nn = len(bonds), e * len(angles)

        new_bonds = np.zeros(n + nn, dtype=bonds.dtype)

        # Copying bonds connectivity and types
        new_bonds[:n] = bonds
        new_bonds["type"][n:].reshape(e, -1)[:] = angles["type"]
        new_bonds["type"][n:] += typeoffset

        for arr, permut in zip(
                np.split(new_bonds["atoms"][n:], e), permutations):
            arr[:] = angles["atoms"][:, permut]

        # Construct unique bond list and triplet_list
        self.connectivity["bonds"], indices_r = \
            np.unique(new_bonds, return_inverse=True)

        # Need to sort after all the shenanigans
        idx = np.argsort(self.connectivity["bonds"]["atoms"][:, 0])
        self.connectivity["bonds"][:] = self.connectivity["bonds"][idx]

        # To construct triplet references (aka ij_t, ik_t and jk_t):
        #   - revert sort operation
        #   - apply reverse unique operatation
        #   - take only appended values
        #   - reshape
        #   - re-sort so that ij_t is sorted
        r_idx = np.zeros_like(idx, dtype=np.int32)
        r_idx[idx] = np.arange(len(idx))  # revert sort
        self.triplet_list = r_idx[indices_r][n:].reshape(e, -1).T

        idx = np.argsort(self.triplet_list[:, 0])  # sort ij_t
        self.triplet_list = self.triplet_list[idx]

    def get_pairs(self, atoms: ase.Atoms, quantities: str, cutoff=None):
        """Return pairs and quantities from connectivities."""
        D, d = None, None

        connectivity = self.connectivity["bonds"]["atoms"].astype(np.int32)

        # If any distance is requested, compute distances vectors and norms
        if "d" in quantities or "D" in quantities:
            D, d = self.compute_distances(atoms, connectivity, [(0, 1)])

        return self.make_result(
            quantities, connectivity, D, d, None, accepted_quantities="ijdD")

    def get_triplets(self,
                     atoms: ase.Atoms,
                     quantities: str,
                     neighbours=None,
                     cutoff=None):
        """Return triplets and quantities from connectivities."""
        D, d = None, None

        # Need to reorder connectivity for distances
        bonds = self.connectivity["bonds"]["atoms"]
        connectivity = np.array([
            bonds[self.triplet_list[:, i], j]
            for i, j in [(0, 0), (0, 1), (1, 1)]
        ]).T

        # If any distance is requested, compute distances vectors and norms
        if "d" in quantities or "D" in quantities:
            #           i  j    i  k    j  k
            indices = [(0, 1), (0, 2), (1, 2)]  # defined in Jan's paper
            D, d = self.compute_distances(atoms, connectivity, indices)

        # Returning triplet references in bonds list
        connectivity = self.triplet_list
        return self.make_result(
            quantities, connectivity, D, d, None, accepted_quantities="ijkdD")

    def find_triplet_types(self, atoms: ase.Atoms, i, j, k):
        triplet_numbers = self.triplet_to_numbers(atoms, i, j, k)
        connectivity_numbers = atoms.numbers[self.connectivity["angles"]
                                             ["atoms"]]
        unique_numbers, indices = np.unique(
            connectivity_numbers, return_index=True, axis=0)
        unique_types = self.connectivity["angles"]["type"][indices]

        all_types = np.zeros(len(triplet_numbers), dtype=np.int32)

        for i in range(all_types.shape[0]):
            all_types[i] = unique_types[
                np.argwhere(np.all(
                    np.equal(unique_numbers, triplet_numbers[i]), axis=1))
            ]

        return all_types


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
        rec *= np.array(pbc, dtype=int).reshape(3, 1)
    dri = np.round(np.dot(dr, rec))

    # Unwrap
    return dr - np.dot(dri, cell)


def neighbour_list(quantities,
                   atoms=None,
                   cutoff=None,
                   positions=None,
                   cell=None,
                   pbc=None,
                   numbers=None,
                   cell_origin=None):
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
            raise ValueError(
                'You cannot provide an ASE Atoms object and '
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

    if isinstance(cutoff, defaultdict):
        _cutoff = cutoff.default_factory()

    elif isinstance(cutoff, dict):
        maxel = np.max(numbers)
        _cutoff = np.zeros([maxel + 1, maxel + 1], dtype=float)
        for (el1, el2), c in cutoff.items():
            try:
                el1 = atomic_numbers[el1]
            except:
                pass
            try:
                el2 = atomic_numbers[el2]
            except:
                pass
            if el1 < maxel + 1 and el2 < maxel + 1:
                _cutoff[el1, el2] = c
                _cutoff[el2, el1] = c
    else:
        _cutoff = cutoff

    try:
        return _matscipy.neighbour_list(quantities, cell_origin, cell,
                                        np.linalg.inv(cell.T), pbc, positions,
                                        _cutoff, numbers)
    except ValueError as e:
        if str(e) == "object of too small depth for desired array":
            raise TypeError(f"cutoff of invalid type {type(_cutoff)}")
        raise e


def triplet_list(first_neighbours,
                 abs_dr_p=None,
                 cutoff=None,
                 i_p=None,
                 j_p=None):
    """
    Compute a triplet list for an atomic configuration. The triple list is a
    mask that can be applied to the corresponding neighbour list to mask
    triplet properties.
    The triplet list accepts an first_neighbour array (generated by
    first_neighbours) as input.

    Parameters
    ----------
    first_neighbours : array
        adresses of the first time an atom occours in the neighour list

    Returns
    -------
    ij_t, ik_t : array
        lists of adresses that form triples in the pair lists
    jk_t : array (if and only if i_p, j_p, first_i != None)
        list of pairs jk that connect each triplet ij, ik
        between atom j and k

    Example
    -------
    i_n, j_n, abs_dr_p = neighbour_list('ijd', atoms=atoms, cutoff=cutoff)

    first_i = np.array([0, 2, 6, 10], dtype='int32')
    a = triplet_list(first_i, [2.2]*9+[3.0], 2.6)

    # one may obtain first_ij by using
    find_ij = first_neighbours(len(i_p), ij_t)
    # or (slower but less parameters and more general,
    # i.e for every ordered list)
    first_ij = get_jump_indicies(ij_t)

    """
    if not (abs_dr_p is None or cutoff is None):
        res = _matscipy.triplet_list(first_neighbours, abs_dr_p, cutoff)
    else:
        res = _matscipy.triplet_list(first_neighbours)
    # TODO: should be wrapped in c and be independet of i_n
    # and j_n as of j_n is sorted; related issue #50
    # add some tests!!!
    if not (i_p is None or j_p is None or first_neighbours is None):
        ij_t, ik_t = res
        jk_t = -np.ones(len(ij_t), dtype='int32')
        for t, (ij, ik) in enumerate(zip(ij_t, ik_t)):
            for i in np.arange(first_neighbours[j_p[ij]],
                               first_neighbours[j_p[ij] + 1]):
                if i_p[i] == j_p[ij] and j_p[i] == j_p[ik]:
                    jk_t[t] = i
                    break
        return ij_t, ik_t, jk_t
    else:
        return res


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
    # np.testing.assert_equal(j_n[sorted_1], i_n[sorted_2])
    # np.testing.assert_equal(i_n[sorted_1], j_n[sorted_2])
    # np.testing.assert_equal(abs_dr_n[sorted_1], abs_dr_n[sorted_2])
    # print(np.c_[i_n[sorted_2], j_n[sorted_2], abs_dr_n[sorted_2],
    #            i_n[sorted_1], j_n[sorted_1], abs_dr_n[sorted_1]])
    tmp2 = np.arange(i_n.size)[sorted_2]
    tmp1 = np.arange(i_n.size)[sorted_1]
    # np.arange(i_n.size) are indices into the neighbor list, so
    #  * the nth element in tmp1 is the index where i,j was before reordering with sorted_1
    #  * the nth element in tmp2 is the index where j,i was before reordering with sorted_2
    reverse = np.empty(i_n.size, dtype=i_n.dtype)
    reverse[tmp1] = tmp2
    return reverse


def find_common_neighbours(i_n, j_n, nat):
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
    first_j = _matscipy.first_neighbours(nat, j_n_2)
    num_rows_per_j = first_j[j_n + 1] - first_j[j_n]
    num_rows_cnl = np.sum(num_rows_per_j)

    # The common neighbor information could be stored as
    # a 2D array. However, multiple 1D arrays are likely
    # better for performance (fewer cache misses later).
    nl_index_i1_j1 = np.empty(num_rows_cnl, dtype=i_n.dtype)
    cnl_j1 = np.empty(num_rows_cnl, dtype=i_n.dtype)
    nl_index_i2_j1 = np.empty(num_rows_cnl, dtype=i_n.dtype)
    cnl_i1_i2 = np.empty((num_rows_cnl, 2), dtype=i_n.dtype)

    block_start = np.r_[0, np.cumsum(num_rows_per_j)]
    slice_for_j1 = {
        j1: slice(first_j[j1], first_j[j1 + 1])
        for j1 in np.arange(nat)
    }
    for block_number, (i1, j1) in enumerate(zip(i_n, j_n)):
        slice1 = slice(block_start[block_number],
                       block_start[block_number + 1])
        slice2 = slice_for_j1[j1]
        nl_index_i1_j1[slice1] = block_number
        cnl_j1[slice1] = j1
        nl_index_i2_j1[slice1] = j_order[slice2]
        cnl_i1_i2[slice1, 0] = i1
        cnl_i1_i2[slice1, 1] = i_n_2[slice2]
    return cnl_i1_i2, cnl_j1, nl_index_i1_j1, nl_index_i2_j1
