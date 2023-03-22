#
# Copyright 2015 Lars Pastewka (U. Freiburg)
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

import itertools

import numpy as np

import ase

from matscipy.neighbours import first_neighbours, neighbour_list

###

def hydrogenate(a, cutoff, bond_length, b=None, mask=[True, True, True],
                exclude=None, vacuum=None):
    """
    Hydrogenate a slab of material at its periodic boundary conditions.
    Boundary conditions are turned into nonperiodic.

    Parameters
    ----------
    a : ase.Atoms
        Atomic configuration.
    cutoff : float
        Cutoff for neighbor counting.
    bond_length : float
        X-H bond length for hydrogenation.
    b : ase.Atoms, optional
        If present, this is the configuration to hydrogenate. Number of atoms
        must be identical to a object. All bonds present in a but not present
        in b will be hydrogenated in b.
    mask : list of bool
        Cartesian directions which to hydrogenate, only if b argument is not
        given.
    exclude : array_like
        Boolean array masking atoms to be excluded from hydrogenation.
    vacuum : float, optional
        Add this much vacuum after hydrogenation.

    Returns
    -------
    a : ase.Atoms
        Atomic configuration of the hydrogenated slab.
    """
    if b is None:
        b = a.copy()
        b.set_pbc(np.logical_not(mask))

    if exclude is None:
        exclude = np.zeros(len(a), dtype=bool)

    i_a, j_a, D_a, d_a = neighbour_list('ijDd', a, cutoff)
    i_b, j_b = neighbour_list('ij', b, cutoff)

    firstneigh_a = first_neighbours(len(a), i_a)
    firstneigh_b = first_neighbours(len(b), i_b)

    coord_a = np.bincount(i_a, minlength=len(a))
    coord_b = np.bincount(i_b, minlength=len(b))

    hydrogens = []
    # Surface atoms have coord_a != coord_b. Those need hydrogenation
    for k in np.arange(len(a))[np.logical_and(coord_a!=coord_b,
                                              np.logical_not(exclude))]:
        l1_a = firstneigh_a[k]
        l2_a = firstneigh_a[k+1]
        l1_b = firstneigh_b[k]
        l2_b = firstneigh_b[k+1]
        n_H = 0
        for l_a in range(l1_a, l2_a):
            assert i_a[l_a] == k
            bond_exists = False
            for l_b in range(l1_b, l2_b):
                assert i_b[l_b] == k
                if j_a[l_a] == j_b[l_b]:
                    bond_exists = True
            if not bond_exists:
                # Bond existed before cut
                hydrogens += [b[k].position+bond_length*D_a[l_a]/d_a[l_a]]
                n_H += 1
        assert n_H == coord_a[k]-coord_b[k]

    if hydrogens == []:
        raise RuntimeError('No Hydrogen created.')

    b += ase.Atoms(['H']*len(hydrogens), hydrogens)

    if vacuum is not None:
        axis=[]
        for i in range(3):
            if mask[i]:
                axis += [i]
        b.center(vacuum, axis=axis)

    return b
