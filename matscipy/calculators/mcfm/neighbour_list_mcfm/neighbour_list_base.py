#
# Copyright 2021 Lars Pastewka (U. Freiburg)
#           2018 Jacek Golebiowski (Imperial College London)
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


class NeighbourListBase(object):
    """Interface for the neighbour list.
    mcfm module can use any neighbour list object as long
    as it provides the implementation of the two routines below.
    """

    def update(self, atoms):
        """Make sure the list is up to date. If clled for the first
        time, build the list

        Parameters
        ----------
        atoms : ase.Atoms
            atoms to initialize the list from

        Returns
        -------
        bool
            True of the update was sucesfull
        """
        raise NotImplementedError("Must implement this function!")

    def get_neighbours(self, a):
        """Return neighbors of atom number a.

        A list of indices to neighboring atoms is
        returned.

        Parameters
        ----------
        a : int
            atomic index


        Returns
        -------
        np.array
            array of neighbouring indices

        """
        raise NotImplementedError("Must implement this function!")
