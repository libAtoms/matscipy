from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from math import sqrt
from .neighbour_list_base import NeighbourListBase
from ....neighbours import neighbour_list as mspy_nl


class NeighbourListMCFM(NeighbourListBase):
    """Neighbor list object. Wrapper aroud matscipy.neighbour_list

    atoms: ase.Atoms
        Atomic configuration.
    cutoffs: float or dict
        Cutoff for neighbour search. If single float is given, a global cutoff
        is used for all elements. A dictionary specifies cutoff for element
        pairs. Specification accepts element numbers of symbols.
        Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
    skin: float
        If no atom has moved more than the skin-distance since the
        last call to the ``update()`` method, then the neighbor list
        can be reused.  This will save some expensive rebuilds of
        the list, but extra neighbors outside the cutoff will be
        returned.
    hysteretic_break_factor: float
        If atoms are connected, the link will break only of they move apart
        further than cutoff * hysteretic_break_factor

    """

    def __init__(self, atoms, cutoffs, skin=0.3, hysteretic_break_factor=1):

        self.cutoffs = cutoffs.copy()
        self.cutoffs_hysteretic = cutoffs.copy()
        if hysteretic_break_factor > 1:
            self.do_hysteretic = True
            for key in self.cutoffs_hysteretic:
                self.cutoffs_hysteretic[key] *= hysteretic_break_factor
        else:
            self.do_hysteretic = False

        self.skin = skin
        self.nupdates = 0

        # Additional data
        self.neighbours = [np.zeros(0) for idx in range(len(atoms))]
        self.old_neighbours = [[] for idx in range(len(atoms))]

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
        if self.nupdates == 0:
            self.do_update(atoms)
            return True
        elif ((self.pbc != atoms.get_pbc()).any() or
              (self.cell != atoms.get_cell()).any() or
              ((self.positions - atoms.get_positions())**2).sum(1).max() >
                self.skin**2):
            self.do_update(atoms)
            return True

        return False

    def do_update(self, atoms):
        """Build the neighbour list based on pairwise distances.

        Parameters
        ----------
        atoms : ase.Atoms
            atoms to initialize the list from

        Raises
        ------
        ValueError
            Must specify cutoff radii for all atoms
        """
        self.positions = atoms.get_positions()
        self.pbc = atoms.get_pbc()
        self.cell = atoms.get_cell()

        shorti, shortj = mspy_nl(str("ij"), atoms, self.cutoffs)

        new_neighbours = [[] for idx in range(len(atoms))]
        for idx in range(len(shorti)):
            new_neighbours[shorti[idx]].append(shortj[idx])

        if self.do_hysteretic:
            longi, longj = mspy_nl(str("ij"), atoms, self.cutoffs_hysteretic)

            for idx in range(len(longi)):
                # Split for profiling
                previously_connected = longj[idx] in self.old_neighbours[longi[idx]]
                not_added = longj[idx] not in new_neighbours[longi[idx]]
                if previously_connected and not_added:
                    new_neighbours[longi[idx]].append(longj[idx])

            self.old_neighbours = new_neighbours

        for idx in range(len(new_neighbours)):
            self.neighbours[idx] = np.asarray(list(new_neighbours[idx]))

        self.nupdates += 1

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

        Raises
        ------
        RuntimeError
            Must update the list at least once!

        """

        if self.nupdates == 0:
            raise RuntimeError("Must update the list at least once!")

        return self.neighbours[a]
