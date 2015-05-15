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

import ase

from matscipy.neighbours import first_neighbours, neighbour_list

###

def hydrogenate(a, cutoff, bond_length, mask, vacuum=None):
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
    mask : list of bool
        Cartesian directions which to hydrogenate.
    vacuum : float, optional
        Add this much vacuum after hydrogenation.

    Returns
    -------
    a : ase.Atoms
        Atomic configuration of the hydrogenated slab.
    """
    b = a.copy()
    b.set_pbc(np.logical_not(mask))

    i_a, j_a, D_a, d_a = neighbour_list('ijDd', a, cutoff)
    i_b, j_b = neighbour_list('ij', b, cutoff)

    firstneigh_a = first_neighbours(len(a), i_a)
    firstneigh_b = first_neighbours(len(b), i_b)

    coord_a = np.bincount(i_a, minlength=len(a))
    coord_b = np.bincount(i_b, minlength=len(b))

    hydrogens = []
    # Surface atoms have coord_a != coord_b. Those need hydrogenation
    for k in np.arange(len(a))[coord_a!=coord_b]:
        l_a = firstneigh_a[k]
        l_b = firstneigh_b[k]
        while l_a < len(i_a) and i_a[l_a] == k:
            if l_b < len(i_b) and i_a[l_a] == i_b[l_b] and j_a[l_a] == j_b[l_b]:
                l_a += 1
                l_b += 1
            else:
                # Bond existed before cut
                hydrogens += [a[k].position+bond_length*D_a[l_a]/d_a[l_a]]
                l_a += 1

    if hydrogens == []:
        raise RuntimeError('Non Hydrogens created.')

    b += ase.Atoms(['H']*len(hydrogens), hydrogens)

    if vacuum is not None:
        axis=[]
        for i in range(3):
            if mask[i]:
                axis += [i]
        b.center(vacuum, axis=axis)

    return b
