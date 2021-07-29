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

import numpy as np

###

from _matscipy import count_islands, count_segments, distance_map

###

# Stencils for determining nearest-neighbor relationships on a square grid
nn_stencil = [(1,0),(0,1),(-1,0),(0,-1)]
nnn_stencil = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

###

def outer_perimeter(c, stencil=nn_stencil):
    """
    Return a map where the outer perimeter is marked with the patch number.

    Parameters
    ----------
    c : array_like
        2D map of islands.

    Returns
    -------
    c : array
        2D map of island perimeters.
    """
    
    c_nearby = np.zeros_like(c, dtype=bool)
    for dx, dy in stencil:
        tmp = c.copy()
        if dx != 0:
            tmp = np.roll(tmp, dx, 0)
        if dy != 0:
            tmp = np.roll(tmp, dy, 1)
        c_nearby = np.logical_or(c_nearby, tmp)
    return np.logical_and(np.logical_not(c), c_nearby)


def inner_perimeter(patch_ids, stencil=nn_stencil):
    """
    Return a map where the inner perimeter is marked with the patch number

    Parameters
    ----------
    c : array_like
        2D map of islands.

    Returns
    -------
    c : array
        2D map of island perimeters.
    """
    
    c = outer_perimeter(patch_ids == 0, stencil)
    return c*patch_ids


def island_areas(island_ids):
    """
    Return a list containing island areas for each island number

    Parameters
    ----------
    c : array_like
        2D map of island ids.
  
    Returns
    -------
    areas : array
        Array containing the area for each island id.
    """
    
    return np.bincount(island_ids.reshape((-1,)))[1:]
