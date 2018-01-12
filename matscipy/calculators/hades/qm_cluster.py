from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from .qmClusterModules.qm_flagging_module import QMFlaggingModule
from .qmClusterModules.qm_clustering_module import QMClusteringModule


class QMcluster(object):
    """This is a class responsible for managing the QM clusters in the simulation.

    It acts as a mediator between
    -----------------------------
        QM flagginig module
        QM clustering module
        neighbour list
    and contains interface to those classes

    Attributes
    ----------
    clustering_module : hades.qmClusterModule.QMClusteringModule
        module responsible for carving a qm cluster
    flagging_module : hades.qmClusterModule.QMFlaggingModule
        module responsible for flagging atoms
    neighbour_list : hades.NeighborListBase
        object holding the neighbour list
    verbose : int
        Set verbosity level
    """

    def __init__(self, special_atoms_list=[], verbose=0):
        """This is a class responsible for managing the QM clusters in the simulation.

        It acts as a mediator between
        -----------------------------
            QM flagginig module
            QM clustering module
            neighbour list
        and contains interface to those classes

        Parameters
        ----------
        special_atoms_list : list of ints (atomic indices)
            In case a group of special atoms are specified (special molecule),
            If one of these atoms is in the buffer region, the rest are also added to it.
        verbose : int
            verbosity level to be passed to other objects
        """
        self.flagging_module = None
        self.clustering_module = None
        self.neighbour_list = None

        self.special_atoms_list = special_atoms_list
        self.verbose = verbose

    def attach_neighbour_list(self, neighbour_list):
        """attach a neighbour list"""
        self.neighbour_list = neighbour_list

    def attach_flagging_module(self, **kwargs):
        """Initialize and attach hades.QMFlaggingModule
        The function calls the class initializer with given parameters"""
        self.flagging_module = QMFlaggingModule(mediator=self, **kwargs)

    def attach_clustering_module(self, **kwargs):
        """Initialize and attach hades.QMClusteringModule
        The function calls the class initializer with given parameters"""
        self.clustering_module = QMClusteringModule(mediator=self, **kwargs)

    def reset_energized_list(self):
        """Reset old_energized_atoms list in flaggingModule to facilitate
        MCFM potential warmup"""
        self.flagging_module.old_energized_list = []

    def update_qm_region(self, *args, **kwargs):
        """Interface to
        self.flagging_module.update_qm_region(self,
                                             atoms,
                                             potential_energies=None,
                                             )"""
        return self.flagging_module.update_qm_region(*args, **kwargs)

    def carve_cluster(self, *args, **kwargs):
        """Interface to
        self.clustering_module.carve_cluster(self,
                                            atoms,
                                            core_qm_list,
                                            buffer_hops=10)"""
        return self.clustering_module.carve_cluster(*args, **kwargs)
