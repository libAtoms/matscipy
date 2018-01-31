from __future__ import absolute_import, division, print_function, unicode_literals
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
