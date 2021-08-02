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


class ClusterData(object):
    """Class for storing cluster data

    Attributes
    ----------
    forces : np.array
        Atomic forces
    mark : list
        Marks assigning atoms as:
            1: core QM region
            2: buffer region
            3: terminal atoms (final atom included in the buffer region)
            4: additional terminal atoms
            5: Hydrogens used ot terminate cut-off bonds
    qm_list : list
        list of inner QM atoms
    """

    def __init__(self, nAtoms, mark=None, qm_list=None, forces=None):
        if len(mark) != nAtoms:
            raise ValueError(
                "mark length not compatible with atoms length in this ClusterData object")
        if np.shape(forces) != (nAtoms, 3):
            raise ValueError(
                "forces shape not compatible with atoms length in this ClusterData object")

        self.forces = forces
        self.mark = mark
        self.qm_list = qm_list
        self.nClusterAtoms = None

    def __str__(self):
        return str(self.mark)
