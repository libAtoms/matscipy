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

from .qm_cluster_tools.qm_flagging_tool import QMFlaggingTool
from .qm_cluster_tools.qm_clustering_tool import QMClusteringTool


class QMCluster(object):
    """This is a class responsible for managing the QM clusters in the simulation.

    It acts as a mediator between
    -----------------------------
        QM flagginig module
        QM clustering module
        neighbour list
    and contains interface to those classes

    Attributes
    ----------
    clustering_module : matscipy.calculators.mcfm.qmClusterModule.QMClusteringTool
        module responsible for carving a qm cluster
    flagging_module : matscipy.calculators.mcfm.qmClusterModule.QMFlaggingTool
        module responsible for flagging atoms
    neighbour_list : matscipy.calculators.mcfm.neighbour_list_mcfm.NeighborListBase
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
        """Initialize and attach matscipy.calculators.mcfm.QMFlaggingTool
        The function calls the class initializer with given parameters"""
        self.flagging_module = QMFlaggingTool(mediator=self, **kwargs)

    def attach_clustering_module(self, **kwargs):
        """Initialize and attach matscipy.calculators.mcfm.QMClusteringTool
        The function calls the class initializer with given parameters"""
        self.clustering_module = QMClusteringTool(mediator=self, **kwargs)

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
