from __future__ import absolute_import, division, print_function, unicode_literals
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
