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

import numpy as np

import _matscipy

###

def angle_distribution(i, j, dr, nbins, *args):
    """
    Compute a bond angle distribution from a neighbour list.

    Parameters
    ----------
    i, j, dr : array_like
        Neighbour list, including list of distance vectors.
    nbins : int
        Number of bins for bond angle histogram.
    cutoff : float, optional
        Bond length cutoff, i.e. consider only bonds shorter than this length.
    """
    return _matscipy.angle_distribution(np.asarray(i), np.asarray(j),
                                        np.asarray(dr), nbins, *args)
