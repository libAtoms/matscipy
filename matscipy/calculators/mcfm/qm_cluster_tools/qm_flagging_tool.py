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

from .base_qm_cluster_tool import BaseQMClusterTool


class QMFlaggingTool(BaseQMClusterTool):
    """This class is responsible for flagging atoms
    that move out of their equilibrium"""

    def __init__(self, mediator=None, qm_flag_potential_energies=None,
                 small_cluster_hops=3, only_heavy=False, ema_parameter=0.1, energy_cap=None,
                 energy_increase=1):
        """This class is responsible for flagging atoms
        that move out of their equilibrium

        Parameters
        ----------
        mediator : matscipy.calculators.mcfm.QMCluster
            class responsible for managing the QM clusters in the simulation
        qm_flag_potential_energies : np.array
            threshholds for flagging indivual atoms.
            The diensions are (nAtoms, 2) where:
                column 1: threshold to enter the QM regios
                column 2: threshold to stay the QM region
        small_cluster_hops : int
            Each flagged atom and atoms around it within small_cluster_hops neighbour hops
            will generate a single cluster, clusters are later joined.
        only_heavy : bool
            If True, only consider non-hydrogen atoms in cluster expansion.
            Hydrogens are added later
        ema_parameter : float
            parameter lambda in the exponential mean average calculation
        energy_cap : float
            if not None, cap potential energy per atom at this value
        energy_increase : int
            Multiplier for potential energy per atom, used to scale it for convininece

        """
        # Initialize the QMClusterObject with a mediator
        super(QMFlaggingTool, self).__init__(mediator)

        try:
            self.qm_flag_potential_energies = qm_flag_potential_energies
        except AttributeError:
            raise AttributeError("QM flag PE/force tolerance must be defined")

        self.small_cluster_hops = small_cluster_hops
        self.only_heavy = only_heavy
        self.ema_parameter = ema_parameter
        self.energy_cap = energy_cap
        self.energy_increase = energy_increase

        self.qm_atoms_list = []
        self.old_energized_list = []

        self.verbose = 0

    def get_energized_list(self, atoms, data_array, property_str, hysteretic_tolerance):
        """Produce a list of atoms that are ot be flagged as a QM region
        based on the properties given in the array according to the
        tolerance given.

        Parameters
        ----------
        atoms : ase.Atoms
            Whole structure
        data_array : array
            an array of per atom data providing information
        property_str : str
            name of th property so that it can be stored in atoms.properties.
        hysteretic_tolerance : array
            Threshholds for flagging indivual atoms.
            The diensions are (nAtoms, 2) where:
                column 1: threshold to enter the QM regios
                column 2: threshold to stay the QM region

        Returns
        -------
        list
            List of flagged atoms
        """

        # ------ Update EPA
        update_avg_property_per_atom(atoms, data_array, property_str, self.ema_parameter)

        avg_property_per_atom = atoms.arrays[property_str]

        tolerance = np.zeros(len(atoms)) + hysteretic_tolerance[:, 0]
        tolerance[self.old_energized_list] = hysteretic_tolerance[self.old_energized_list, 1]
        energized_mask = np.greater_equal(avg_property_per_atom, tolerance)
        energized_list = np.arange(len(atoms))[energized_mask]

        return energized_list

    def create_cluster_around_atom(self, atoms, atom_id, hydrogenate=False):
        """Carve a cluster around the atom with atom_id
        This function operates on sets and returns a set

        Parameters
        ----------
        atoms : ase.Atoms
            Whole structure
        atom_id : int
            Atomic index
        hydrogenate : bool
            If true, hydrogenate the resulting structure

        Returns
        -------
        list
            atoms in the new cluster
        """
        cluster_set = set([atom_id])
        edge_neighbours = set([atom_id])

        for i in range(self.small_cluster_hops):
            new_neighbours = set()
            # For each atom in edge neighbours list, expand the list
            for index in edge_neighbours:
                new_neighbours |= set(self.find_neighbours(atoms, index)[0])
            # Remove atoms already in the qm list
            edge_neighbours = new_neighbours - cluster_set
            # Make a union of the sets
            cluster_set = cluster_set | edge_neighbours

        # ----- If specified, add hydrogens ot the cluster
        if hydrogenate:
            self.hydrogenate_cluster(atoms, cluster_set)
        return cluster_set

    def join_clusters(self, verbose=False):
        """This function will join the clusters if they overlap
        Input is an array of sets each representing individual
        small cluster

        Parameters
        ----------
        verbose : bool
            Print messages during calculation
        """

        i = 0
        # Iterate over the whole list C taking into account that it might get
        # throughout the loop
        while (i < len(self.qm_atoms_list)):

            # Iterate over the sets taking into account that C can change
            # Do not repeat pairise disjointment checks
            # i.e. for a list of sets [A, B, C, D]
            # first loop included checks A-B, A-C, A-D (pairs 0 - 1:3)
            # Then make sure the second only does B-C, B-D (pairs 1 - 2:3)
            for j in range(i + 1, len(self.qm_atoms_list)):
                if verbose is True:
                    print(i, j, self.qm_atoms_list[i], self.qm_atoms_list[j],
                          not set.isdisjoint(self.qm_atoms_list[i], self.qm_atoms_list[j]))

                if not set.isdisjoint(self.qm_atoms_list[i], self.qm_atoms_list[j]):
                    # If intersection detected, unify sets
                    self.qm_atoms_list[i] |= self.qm_atoms_list[j]
                    # Then delete the second set to avoid duplicates
                    # Then restart the j loop to see if now, any set
                    # has an intersection with the new union
                    del self.qm_atoms_list[j]
                    i -= 1

                    if verbose is True:
                        for entry in self.qm_atoms_list:
                            print(entry)
                    break
            i += 1

    def expand_cluster(self, special_atoms_list):
        """Include extra atoms in the cluster.

        If one of the special atoms is included in one of the clusters,
        add all other special atoms to this cluster

        Parameters
        ----------
        special_atoms_list : list
            list of the special atoms
        """

        for specialMolecule in special_atoms_list:
            specialMoleculeSet = set(specialMolecule)
            for clusterIndex in range(len(self.qm_atoms_list)):
                if (not specialMoleculeSet.isdisjoint(self.qm_atoms_list[clusterIndex])):
                    self.qm_atoms_list[clusterIndex] |= specialMoleculeSet

    def update_qm_region(self, atoms,
                         potential_energies=None,
                         ):
        """Update the QM region while the simulation is running

        Parameters
        ----------
        atoms : ase.Atoms
            whole structure
        potential_energies : array
            Potential energy per atom

        Returns
        -------
        list of lists of ints
            list of individual clusters as lists of atoms
        """
        # Make sure the right atoms object is in

        # ------ Increase the energy by a common factor - makes it more readable in some cases
        if (self.energy_increase is not None):
            potential_energies *= self.energy_increase

        # ------ Cap maximum energy according to the flag
        if (self.energy_cap is not None):
            np.minimum(potential_energies, self.energy_cap, potential_energies)

        # ------ Get the energized atoms list
        flagged_atoms_dict = {}

        flagged_atoms_dict["potential_energies"] = self.get_energized_list(atoms,
                                                                           potential_energies,
                                                                           "avg_potential_energies",
                                                                           self.qm_flag_potential_energies)

        energized_set = set()
        for key in flagged_atoms_dict:
            energized_set = set(flagged_atoms_dict[key]) | energized_set
        energized_list = list(energized_set)
        self.old_energized_list = list(energized_list)

        if (len(energized_list) != 0):
            self.mediator.neighbour_list.update(atoms)

        # TODO if energized list include the whole system just pass it along
        for array_i, atom_i in enumerate(energized_list):
            energized_list[array_i] = self.create_cluster_around_atom(atoms, atom_i, hydrogenate=False)

        self.qm_atoms_list = energized_list
        if (len(self.qm_atoms_list) > 0):
            self.join_clusters()
            self.expand_cluster(self.mediator.special_atoms_list)
            self.join_clusters()

        if self.only_heavy is False:
            for index in range(len(self.qm_atoms_list)):
                self.qm_atoms_list[index] = self.hydrogenate_cluster(atoms, self.qm_atoms_list[index])

        self.qm_atoms_list = list(map(list, self.qm_atoms_list))
        return self.qm_atoms_list
        # print "QM cluster", self.qm_atoms_list


def exponential_moving_average(oldset, newset=None, ema_parameter=0.1):
    """Apply the exponential moving average to the given array

    Parameters
    ----------
    oldset : array
        old values
    newset : array
        new data set
    ema_parameter : float
        parameter lambda
    """
    if newset is None:
        pass
    else:
        oldset *= (1 - ema_parameter)
        oldset += ema_parameter * newset


def update_avg_property_per_atom(atoms, data_array, property_str, ema_parameter):
    """Update the per atom property using running avarages
    and store it in atoms.properties[property_str]

    Parameters
    ----------
    atoms : ase.Atoms
        structure that need updated values
    data_array : array
        data that need to be attached to atoms
    property_str : str
        key for structure properties dictionary
    ema_parameter : float
        Coefficient for the Exponential Moving Average
    """

    # Abbreviations
    # ppa - (property per atom
    # appa - average property per atom

    ppa = data_array

    # ------ Get average ppa
    if (property_str in atoms.arrays):
        exponential_moving_average(atoms.arrays[property_str],
                                   ppa, ema_parameter)
    else:
        atoms.arrays[property_str] = ppa.copy()
