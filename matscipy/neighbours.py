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


def neighbour_list(quantities, a, cutoff, *args):
    """
    Compute a neighbour list for an atomic configuration.

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
                  distances d between can be computed as:
                  D = a.positions[j]-a.positions[i]+S.dot(a.cell)
    a : ase.Atoms
        Atomic configuration.
    cutoff : float or array_like
        Cutoff for neighbour search. If square array is given, then different
        cutoffs will be used for individual bonds. Square array contains
        pair-wise cutoffs for a given species, given by the *numbers* parameter.
    numbers : array_like, optional
        Atomic numbers or similar identifiers for elements. Used for cutoff
        lookup.

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
        i = neighbour_list('i', a, 1.85)
        coord = np.bincount(i)

    2. Coordination counting with different cutoffs for each pair of species
       (Assumes that species are Carbon=6 and Hydrogen=1)
        i = neighbour_list('i', a, [[1.1, 1.3], [1.3, 1.85]], (a.numbers-1)//5)
        coord = np.bincount(i)

    3. Pair distribution function:
        d = neighbour_list('d', a, 10.00)
        h, bin_edges = np.histogram(d, bins=100)
        pdf = h/(4*np.pi/3*(bin_edges[1:]**3 - bin_edges[:-1]**3)) * a.get_volume()/len(a)

    4. Pair potential:
        i, j, d, D = neighbour_list('ijdD', a, 5.0)
        energy = (-C/d**6).sum()
        pair_forces = (6*C/d**5  * (D/d).T).T
        forces_x = np.bincount(j, weights=pair_forces[:, 0], minlength=len(a)) - \
                   np.bincount(i, weights=pair_forces[:, 0], minlength=len(a))
        forces_y = np.bincount(j, weights=pair_forces[:, 1], minlength=len(a)) - \
                   np.bincount(i, weights=pair_forces[:, 1], minlength=len(a))
        forces_z = np.bincount(j, weights=pair_forces[:, 2], minlength=len(a)) - \
                   np.bincount(i, weights=pair_forces[:, 2], minlength=len(a))
    """

    return _matscipy.neighbour_list(quantities, a.cell,
                                    np.linalg.inv(a.cell.T), a.pbc,
                                    a.positions, cutoff, *args)

