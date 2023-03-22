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
from ase import Atom


class QMClusteringTool(BaseQMClusterTool):
    """This class is responsible for carving and hydrogenating
    a qm cluster"""

    def __init__(self, mediator=None, double_bonded_atoms_list=[]):
        """This class is responsible for carving and hydrogenating
        a qm cluster

        Parameters
        ----------
        mediator : matscipy.calculators.mcfm.QMCluster
            class responsible for managing the QM clusters in the simulation
        double_bonded_atoms_list : list
            list of doubly bonded atoms, needed for double hydrogenation.
        """
        # Initialize the QMClusterObject with a mediator
        super(QMClusteringTool, self).__init__(mediator)

        self.double_bonded_atoms_list = double_bonded_atoms_list

    def create_buffer_region(self, atoms, qm_atoms_list, buffer_hops=10):
        """GIven a list of active QM atoms, returns a list containing buffer atoms indices

        Parameters
        ----------
        atoms : ase.Atoms
            whole structure
        qm_atoms_list : list of ints (atomic indexes)
            List of atoms in the inner QM region
        buffer_hops : int
            Expand the cluster by this many neighbour hops to create the buffer

        Returns
        -------
        list of ints (atomic indexes)
            buffer_list: List of atoms in the buffer region
        list of ints (atomic indexes)
            terminal_atoms: List of atoms from the buffer region that are on the verge of it
        list of ints (atomic indexes)
            cutoff_atoms_list: List of atoms that are not in the buffer but are bonded to the
            atoms in the buffer.
        """
        innerQM_region_set = set(qm_atoms_list)
        edge_neighbours = set(qm_atoms_list)
        terminal_atoms = []

        if len(qm_atoms_list) == len(atoms):
            return [], [], []

        for i in range(buffer_hops):
            new_neighbours = set()
            # For each atom in edge neighbours list, expand the list
            for index in edge_neighbours:
                new_neighbours |= set(self.find_neighbours(atoms, index)[0])
            # Remove atoms already in the qm list
            edge_neighbours = new_neighbours - innerQM_region_set

            # If the cluster is still growing, update the list of atoms at the edge
            if len(edge_neighbours) > 0:
                terminal_atoms = edge_neighbours
            # Make a union of the sets
            innerQM_region_set |= edge_neighbours

        # GO throught the loop one more time to find the last atoms not in the cluster
        new_neighbours = set()
        # For each atom in edge neighbours list, expand the list
        for index in edge_neighbours:
            new_neighbours |= set(self.find_neighbours(atoms, index)[0])
        # Remove atoms already in the qm list
        cutoff_atoms_set = new_neighbours - innerQM_region_set
        cutoff_atoms_list = list(cutoff_atoms_set)

        # Create buffer list
        innerQM_region_set -= set(qm_atoms_list)
        buffer_list = list(innerQM_region_set)
        terminal_atoms = list(terminal_atoms)
        return buffer_list, terminal_atoms, cutoff_atoms_list

    def carve_cluster(self, atoms, core_qm_list, buffer_hops=10):
        """Create a cluster with the list as core atoms, returns an ase.Atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            whole structure
        core_qm_list : list of ints (atomic indexes)
            Indexes of atoms in the core QM region
        buffer_hops : int
            Expand the cluster by this many neighbour hops to create the buffer

        Returns
        -------
        ase.Atoms
            atoms object representing the QM cluster including
            the inner region and the buffer
        """

        # Define some lists
        total_supplementary_terminal_atoms = []

        # Buffer atoms - the buffer region
        # terminal_atoms_list - last atoms in the buffer region, bonded to the atoms not in the buffer
        # cutoff_atoms_list - atoms to be changed into hydrogen (cut-off from the buffer)
        # total_supplementary_terminal_atoms - atoms added to the buffer to make clusters more physical

        self.print_message("Creating buffer list", 1)
        buffer_list, terminal_atoms_list, cutoff_atoms_list =\
            self.create_buffer_region(atoms, core_qm_list, buffer_hops=buffer_hops)

        # If a spcial atoms was to be cut-off, add all of them to the cluster
        specialFlag = True
        while (specialFlag):
            specialFlag = False
            for specialMolecule in self.mediator.special_atoms_list:
                # Only operate on the special molecule if a part of it is inside the buffer
                if not any([specialAtom in buffer_list for specialAtom in specialMolecule]):
                    continue

                for specialAtomIndex in specialMolecule:
                    if (specialAtomIndex in cutoff_atoms_list):
                        self.include_special_atom(specialAtomIndex,
                                                  atoms,
                                                  buffer_list,
                                                  terminal_atoms_list,
                                                  cutoff_atoms_list)
                        # If at least one atom was added, cotinue to the next loop
                        specialFlag = True

        # Complete aromatic rings, repeat the process untill no more atoms are added
        iterMax = 10
        for i in range(iterMax):
            completeFlag = self.complete_aromatic_rings(
                atoms, buffer_list, terminal_atoms_list, cutoff_atoms_list,
                total_supplementary_terminal_atoms)
            if (not completeFlag):
                break

        # Create joint list. Buffer list is the original buffer while the
        # supplementary list is composed of new additions
        self.print_message("Creating joint listt", 10)
        if len(buffer_list) > 0:
            total_list = core_qm_list + buffer_list + total_supplementary_terminal_atoms
        else:
            total_list = core_qm_list

        # Add missing hydrogens (all routines operate only on heavy atoms)
        total_set = set(total_list)
        self.hydrogenate_cluster(atoms, total_set)
        total_list = list(total_set)

        self.print_message("finished adding atoms", 10)
        self.print_message("Buffer complete, creating cluster from mark", 1)

        atomic_cluster = self.create_cluster_from_marks(atoms, total_list)
        atomic_cluster.info["no_quantum_atoms"] = len(atomic_cluster)

        # Add properties for core region and buffer
        atomic_cluster.arrays["cluster_mark"] = np.zeros(len(atomic_cluster), dtype=int)
        atomic_cluster.arrays["cluster_mark"] += 5

        for i in range(len(atomic_cluster)):
            if atomic_cluster.arrays["orig_index"][i] in total_supplementary_terminal_atoms:
                atomic_cluster.arrays["cluster_mark"][i] = 4

            elif atomic_cluster.arrays["orig_index"][i] in terminal_atoms_list:
                atomic_cluster.arrays["cluster_mark"][i] = 3

            elif atomic_cluster.arrays["orig_index"][i] in buffer_list:
                atomic_cluster.arrays["cluster_mark"][i] = 2

            elif atomic_cluster.arrays["orig_index"][i] in core_qm_list:
                atomic_cluster.arrays["cluster_mark"][i] = 1

        # Change the cut-off atoms into hydrogens
        self.print_message("Change cutoff atoms into hydrogens", 1)
        if (len(cutoff_atoms_list) > 0) and(len(terminal_atoms_list) > 0):
            terminal_atoms_list = list(set(terminal_atoms_list))
            self.hydrogenate_dangling_bonds(terminal_atoms_list, cutoff_atoms_list, atomic_cluster, atoms)

        self.print_message("Center the atomic_cluster and remove PBC's", 1)
        # Center the cluster and remove PBC's
        # atomic_cluster.positions += np.array([0, 0, 20])
        atomic_cluster.wrap()
        atomic_cluster.center(vacuum=30)
        atomic_cluster.pbc = np.array([0, 0, 0], dtype=bool)

        self.print_message("Fished!", 1)

        return atomic_cluster

    def hydrogenate_dangling_bonds(self, terminal_atoms_list, cutoff_atoms_list, atomic_cluster, atoms):
        """Change atoms that were cut-off into hydrogens

        Parameters
        ----------
        terminal_atoms_list : list of ints (atomic indexes)
            last atoms in the buffer region, bonded to the atoms not in the buffer
        cutoff_atoms_list : list of ints (atomic indexes)
            atoms to be changed into hydrogen, first atoms not in the buffer
        atomic_cluster : ase.Atoms
            QM region structure (with core and buffer atoms)
        atoms : ase.Atoms
            whole structure
        """

        pos = atoms.get_positions()
        # Change cutoff list into a numpy array
        cutoff_atoms_list = np.asarray(cutoff_atoms_list)
        for tAI in terminal_atoms_list:
            # Check if any of the cut off atoms are neighbours of the terminal atom
            cutoff_neighs = [item for item in self.mediator.neighbour_list.get_neighbours(
                tAI) if (item in cutoff_atoms_list)]
            # Iterate over all cut-off atoms that are neighburs of tAI to
            # Effectively loop over all cut bonds
            for cAI in cutoff_neighs:
                if ((cAI in self.double_bonded_atoms_list) and (tAI in self.double_bonded_atoms_list)):
                    self.replace_double_bond(tAI, cAI, atomic_cluster, atoms, pos)
                else:
                    self.replace_single_bond(tAI, cAI, atomic_cluster, atoms, pos)

    def replace_single_bond(self,
                            terminal_atom_index,
                            cutoff_atom_index,
                            atomic_cluster,
                            atoms,
                            atomic_positions):
        """Replace a cut-off atom with a single hydrogen

        Parameters
        ----------
        terminal_atoms_list : list of ints (atomic indexes)
            last atoms in the buffer region, bonded to the atoms not in the buffer
        cutoff_atoms_list : list of ints (atomic indexes)
            atoms to be changed into hydrogen, first atoms not in the buffer
        atomic_cluster : ase.Atoms
            QM region structure (with core and buffer atoms)
        atoms : ase.Atoms
            whole structure
        atomic_positions : np.array
            Positions of atoms in the whole structure (copy of the atoms.positions)
        """
        vector = atomic_positions[cutoff_atom_index] - atomic_positions[terminal_atom_index]
        # Make the bond approximately 1 angstrom
        vector /= np.linalg.norm(vector)
        vector *= 1
        # Add a hydrogen instead of the cutoff atom
        pos = atomic_positions[terminal_atom_index] + vector
        cutoff_hydro = Atom(symbol=1, position=pos, charge=0.1)
        atomic_cluster.append(cutoff_hydro)
        atomic_cluster.arrays["orig_index"][len(atomic_cluster) - 1] = len(atoms) + 1
        atomic_cluster.arrays["cluster_mark"][len(atomic_cluster) - 1] = 6

    def replace_double_bond(self,
                            terminal_atom_index,
                            cutoff_atom_index,
                            atomic_cluster,
                            atoms,
                            atomic_positions):
        """Replace a cut-off atom with two hydrogens

        Parameters
        ----------
        terminal_atoms_list : list of ints (atomic indexes)
            last atoms in the buffer region, bonded to the atoms not in the buffer
        cutoff_atoms_list : list of ints (atomic indexes)
            atoms to be changed into hydrogen, first atoms not in the buffer
        atomic_cluster : ase.Atoms
            QM region structure (with core and buffer atoms)
        atoms : ase.Atoms
            whole structure
        atomic_positions : np.array
            Positions of atoms in the whole structure (copy of the atoms.positions)
        """
        # Find a vector to from the terminal atom to the cutoff atom
        vector = atomic_positions[cutoff_atom_index] - atomic_positions[terminal_atom_index]

        # ------ Find the displacement between two hydrogens
        # Find two closest neighbours of the cut-off atom
        neighbours = np.asarray(self.find_neighbours(atoms, terminal_atom_index)[0])
        dispVectors = atomic_positions[neighbours] - atomic_positions[terminal_atom_index]
        distances = np.sum(np.square(dispVectors), axis=1)
        closeNeighbours = np.argsort(distances)
        closeNeighbours = neighbours[closeNeighbours][: 2]

        # Find the vectors to those two atoms
        a1 = atomic_positions[terminal_atom_index] - atomic_positions[closeNeighbours[0]]
        a2 = atomic_positions[terminal_atom_index] - atomic_positions[closeNeighbours[1]]

        # Find the cross product of a1 and a2 thus finding a vector perpendicular to
        # the plane they define
        aPerp = np.cross(a1, a2)
        aPerp /= np.linalg.norm(aPerp)
        aPerp *= 2

        # Create two vectors, the initial displacement +/- the perpendicular vector
        vector1 = vector + aPerp
        vector2 = vector - aPerp

        # Make the bonds approximately 1 angstrom
        vector1 /= np.linalg.norm(vector1)
        vector2 /= np.linalg.norm(vector2)

        # Add a hydrogen instead of the cutoff atom
        pos = atomic_positions[terminal_atom_index] + vector1
        cutoff_hydro = Atom(symbol=1, position=pos, charge=0.1)
        atomic_cluster.append(cutoff_hydro)
        atomic_cluster.arrays["orig_index"][len(atomic_cluster) - 1] = len(atoms) + 1
        atomic_cluster.arrays["cluster_mark"][len(atomic_cluster) - 1] = 6

        pos = atomic_positions[terminal_atom_index] + vector2
        cutoff_hydro = Atom(symbol=1, position=pos, charge=0.1)
        atomic_cluster.append(cutoff_hydro)
        atomic_cluster.arrays["orig_index"][len(atomic_cluster) - 1] = len(atoms) + 1
        atomic_cluster.arrays["cluster_mark"][len(atomic_cluster) - 1] = 6

    def include_special_atom(self,
                             specialAtomIndex,
                             atoms,
                             buffer_list,
                             terminal_atoms_list,
                             cutoff_atoms_list):
        """Add a special atom to the buffer and update indexes.
        In case a group of special atoms are specified (special molecule),
        If one of these atoms is in the buffer regios, the rest are also added to it.

        Parameters
        ----------
        specialAtomIndex : int (atomic index)
            Add a specified atoms to the buffer
        atoms : ase.Atoms
            whole structure
        buffer_list : list of ints (atomic indexes)
            List of atoms in the buffer region
        terminal_atoms_list : list of ints (atomic indexes)
            last atoms in the buffer region, bonded to the atoms not in the buffer
        cutoff_atoms_list : list of ints (atomic indexes)
            atoms to be changed into hydrogen, first atoms not in the buffer
        """

        buffer_list.append(specialAtomIndex)
        terminal_atoms_list.append(specialAtomIndex)
        cutoff_atoms_list.remove(specialAtomIndex)

        # ------ Add new cutoff atoms
        specialAtomNeighbours = self.find_neighbours(atoms, specialAtomIndex)[0]
        for neighIndex in specialAtomNeighbours:
            if (neighIndex not in buffer_list) and (neighIndex not in cutoff_atoms_list):
                cutoff_atoms_list.append(neighIndex)

    def complete_aromatic_rings(self, atoms, buffer_list,
                                terminal_atoms_list, cutoff_atoms_list,
                                total_supplementary_terminal_atoms):
        """Check if a terminal atom is not connected ot two atoms at once
        If it is, add it. This deals with aromatic ring structures

        Parameters
        ----------
        atoms : ase.Atoms
            whole structure
        buffer_list : list of ints (atomic indexes)
            List of atoms in the buffer region
        terminal_atoms_list : list of ints (atomic indexes)
            last atoms in the buffer region, bonded to the atoms not in the buffer
        cutoff_atoms_list : list of ints (atomic indexes)
            atoms to be changed into hydrogen, first atoms not in the buffer
        total_supplementary_terminal_atoms : NONE
            atoms added to the buffer to make clusters more physical.
            Example: last atom in the aromnatic ring

        Returns
        -------
        bool
            Return True if any atoms were added
        """

        supplementary_terminal_atoms = []

        # Buffer atoms - the buffer region
        # terminal_atoms_list - last atoms in the buffer region, bonded to the atoms not in the buffer
        # cutoff_atoms_list - atoms to be changed into hydrogen (cut-off from the buffer)
        # supplementary_terminal_atoms - atoms added to the buffer to make clusters more physical

        self.print_message("Completing destroyed rings", 1)
        for index, cI in enumerate(cutoff_atoms_list):
            msg = "Working on atom {0} with number {1}".format(cI, atoms.numbers[cI])
            self.print_message(msg, 100)

            # Check if a cutoff atom has more than 1 neighbour in terminal atoms list
            neighs = self.find_neighbours(atoms, cI)[0]
            cutoff_atom_neighs = [item for item in neighs if item in (terminal_atoms_list + buffer_list)]
            # If two or more, add it
            if (len(cutoff_atom_neighs) >= 2):
                supplementary_terminal_atoms.append(cI)
                self.print_message("Adding {0} to supplementary index".format(cI), 10)
        self.print_message("Finished adding atoms.", 10)

        # Return False if no atoms were added
        if (len(supplementary_terminal_atoms) == 0):
            return False

        # Keep track of all the additions to the supplementary atoms
        total_supplementary_terminal_atoms += supplementary_terminal_atoms
        terminal_atoms_list += total_supplementary_terminal_atoms
        if (self.verbose >= 10):
            print("Added", len(supplementary_terminal_atoms), "Atoms!")
        if (self.verbose >= 100):
            print("Added list:", supplementary_terminal_atoms)

        # Find new cutoff atoms
        if (self.verbose >= 10):
            print("Finding new cutoff atoms")

        cutoff_atoms_list[:] = []
        outer_qm_list = buffer_list + total_supplementary_terminal_atoms
        for eqI in terminal_atoms_list:
            new_cutoff_atoms = [item for item in self.find_neighbours(atoms, eqI)[0] if
                                (item not in outer_qm_list)]
            cutoff_atoms_list += new_cutoff_atoms

        # Get rid of duplicates
        cutoff_atoms_list[:] = list(set(cutoff_atoms_list))
        terminal_atoms_list[:] = list(set(terminal_atoms_list))

        # Return True if any atoms were added:
        return True

    def create_cluster_from_marks(self, atoms, select_list):
        """Return an ase.Atoms object containing selected atoms from
        a larger structure

        Parameters
        ----------
        atoms : ase.Atoms
            whole structure
        select_list : list of ints (atomic indexes)
            List of atoms to include in the new structure

        Returns
        -------
        ase.Atoms
            Structure composed of selected atoms
        """

        if (len(select_list) > len(atoms)):
            select_list = np.unique(select_list)

        pbc = atoms.get_pbc()
        cell = atoms.get_cell()
        cluster = atoms.__class__(cell=cell, pbc=pbc)

        cluster.arrays = {}
        # for name, a in atoms.arrays.items():
        #     cluster.arrays[name] = a[select_list].copy()

        for name in ["numbers", "positions"]:
            cluster.arrays[name] = atoms.arrays[name][select_list].copy()

        cluster.arrays["orig_index"] = np.asarray(select_list, dtype=int)

        return cluster

    def print_message(self, message, limit=100):
        """Print a message if the calculators verbosity level is above the
        given threshold

        Parameters
        ----------
        message : str
            The message to be printed
        limit : int
            the verbosity threshold for this mesage
        """
        if (self.verbose >= limit):
            print(message)
