from __future__ import absolute_import, division, print_function, unicode_literals
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
