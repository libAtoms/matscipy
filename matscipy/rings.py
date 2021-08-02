#
# Copyright 2014-2015 Lars Pastewka (U. Freiburg)
#           2014 James Kermode (Warwick U.)
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

from matscipy.neighbours import neighbour_list
import _matscipy 

###

def ring_statistics(a, cutoff, maxlength=-1):
    """
    Compute number of shortest path rings in sample.
    See: D.S. Franzblau, Phys. Rev. B 44, 4925 (1991)

    Parameters
    ----------
    a : ase.Atoms
        Atomic configuration.
    cutoff : float
        Cutoff for neighbor counting.
    maxlength : float, optional
        Maximum ring length. Search for rings will stop at this length. This
        is useful to speed up calculations for large systems.

    Returns
    -------
    ringstat : array
        Array with number of shortest path rings.
    """
    i, j, r = neighbour_list('ijD', a, cutoff)
    d = _matscipy.distances_on_graph(i, j)
    return _matscipy.find_sp_rings(i, j, r, d, maxlength)
