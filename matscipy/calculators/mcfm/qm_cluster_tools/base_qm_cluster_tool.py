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


class BaseQMClusterTool(object):
    """Class that hold all the functions common to
    qm clustering objects"""

    def __init__(self, mediator):
        """Class that hold all the functions common to
        qm clustering objects

        Parameters
        ----------
        mediator : matscipy.calculators.mcfm.QMCluster
            class responsible for managing the QM clusters in the simulation
        """
        self.mediator = mediator
        self.verbose = mediator.verbose

    def find_neighbours(self, atoms, index):
        """Find the neighbours of atom i using self.neighbour_list
        returns a list of [heavy_neighbours, hydrogen_neighbours]

        Parameters
        ----------
        atoms : ase.Atoms object
            structure in which it is necessary to find the neighbours
        index : int
            atomic index

        Returns
        -------
        list
            non-hydrogen neighbours
        list
            hydrogen neighbours
        """

        neighbours = self.mediator.neighbour_list.get_neighbours(index)

        heavy_n = []
        hydro_n = []
        for arr_i, atom_i in enumerate(neighbours):
            if (atoms.numbers[atom_i] == 1):
                hydro_n.append(atom_i)
            else:
                heavy_n.append(atom_i)

        return [heavy_n, hydro_n]

    def hydrogenate_cluster(self, atoms, cluster):
        """Add neigoburing hydrogens to a cluster composed of heavy ions
        The input should be a set representing heavy ions in a cluster
        This functions operates on sets

        Parameters
        ----------
        atoms : ase.Atoms object
            structure in which it is necessary to find the neighbours
        cluster :ase.Atoms object
            sub-structure of the larger struct that needs its dangling
            bonds hydrogenated

        Returns
        -------
        ase.Atoms
            The original cluster but now hydrogenated
        """

        for atom_id in cluster.copy():
                # find_neighbours returns a list where
                # [0] - heavy neighbours
                # [1] - hydrogen neighbours
            cluster |= set(self.find_neighbours(atoms, atom_id)[1])

        return cluster
