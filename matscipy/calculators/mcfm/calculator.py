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

from ase.calculators.calculator import Calculator
from .mcfm_parallel import mcfm_parallel_control as mcfm_parallel_control
from .cluster_data import ClusterData
import ase.io


class MultiClusterForceMixingPotential(Calculator):
    """Subclass of ASE.Calculator facilitating a support for multiple
    QM clusters. It utilizes the given classical_calculator and qm_calculator to initialize
    an instace of ForceMixingPotential

    Extends:
        Calculator

    Variables:
        implemented_properties {list} -- ["energy", "forces", "potential_energies", "stress"]
        all_changes {list} -- ['positions', 'numbers', 'cell', 'pbc']
    """

    implemented_properties = ["energy", "forces", "potential_energies", "stress"]
    all_changes = ['positions', 'numbers', 'cell', 'pbc']

    def __init__(self, atoms=None, classical_calculator=None, qm_calculator=None,
                 qm_cluster=None, forced_qm_list=None, change_bonds=True,
                 calculate_errors=False, calculation_always_required=False,
                 buffer_hops=10, verbose=0, enable_check_state=True):
        """Initialize a generic ASE potential without any calculating power,
        This is only to have access to the necessary functions, all the
        evaluations will be performes in self.mm_pot and self.qm_calculator

        Parameters
        ----------
        atoms : ASE.atoms
            atoms object
        classical_calculator : AE.calculator
            classical calculator
        qm_calculator : ASE.calculator
            qm calculator
        qm_cluster : matscipy.calculators.mcfm.qm_cluster
            flagging/cluster carving utility
        forced_qm_list : list
            add this list to enforce a set of atoms for qm treatment
        change_bonds : bool
            call the classical potential to update topology
        calculate_errors : bool
            evaluate errors after each step
        calculation_always_required : bool
            as name
        buffer_hops : int
            number of neighbours hop used to construct the core QM region
        verbose : int
            For now verbose levels are:
            0 - nothing is printed
            1 - More data is added to Atoms object
            10 - Calculate steps are listed
            100 - Information about specific QM clusters
            (the default is 0)
        enable_check_state : bool
            Save the atoms after each evaluation to enable meth::check_state
        """

        # Set the verbose status
        self.verbose = verbose
        self.debug_cluster_carving = False

        # Set storing atoms - slows down evaluation but enables check_state funtion
        self.enable_check_state = enable_check_state

        # Flag for warmup
        self.warmup = False

        # Init ASE calculator as a parent class
        self._calc_args = {}
        self._default_properties = []
        self.calculation_always_required = calculation_always_required
        Calculator.__init__(self)

        # If an atoms objct has been specified, attach a copy to the calculator to facilitate
        # the proper use of meth:check_state()
        if atoms is not None:
            self.atoms = atoms.copy()
            atoms.set_calculator(self)

        # Set some flags and values
        self.errors = {}
        self.calculate_errors = calculate_errors
        self.change_bonds = change_bonds
        self.buffer_hops = buffer_hops
        self.conserve_momentum = False
        self.long_range_weight = 0.0
        self.doParallel = True

        # Flag for QM debugging
        self.debug_qm_calculator = False

        # Set the cluster carving object
        self.qm_cluster = qm_cluster
        if forced_qm_list is None:
            self.forced_qm_list = None
        else:
            self.forced_qm_list = [forced_qm_list]
        self.cluster_list = []
        self.cluster_data_list = None

        # Set qm and mm calculators
        self.classical_calculator = classical_calculator
        self.qm_calculator = qm_calculator

        # Set up writing clusters
        self.clusterDebug_cluster_atoms = None
        if (self.verbose >= 100):
            self.debug_cluster_carving = True
            self.clusterDebug_cluster_atoms = open("clusterDebug_cluster_atoms.xyz", "w")
        self.clusterDebug_full_structure = None
        if (self.verbose >= 100):
            self.debug_cluster_carving = True
            self.clusterDebug_full_structure = open("clusterDebug_full_structure.xyz", "w")

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        """Calculate selected properties of the given Atoms object.
        Initially, a classical potential is called to evaluate potential
        energies for each atom and afterwards a qm_cluster object is employed
        to analyze them. If no atom is flagged for QM treatment, classical forces
        are returned. In case some atoms are flagged for QM treatment
        each Qm cluster is independently send to a qmmm potential to evaluate
        more accurate forces. The results of qmmm evaluations are used to modify
        the classical forces and the final array is given

        results are
        -----------
        energy = potential energy from classical evaluation
        potential energies = pot energy per atom from classical evaluation
        forces = classical or qmmm forces depending on whether any atoms are flagged

        Parameters
        ----------
        atoms : ASE.atoms
            atoms object
        properties : list
            properties ot evaluate
        system_changes : TYPE
            changes in the system

        Raises
        ------
        AttributeError
            Must provide an atoms object
        """

        Calculator.calculate(self, atoms=atoms, properties=properties, system_changes=system_changes)
        # Check for atoms and if present wrap them to their cell
        if atoms is None:
            raise AttributeError("No atoms object provided")
        else:
            pass

        self.print_message("\nCalculation with MCFM potential!", limit=10)
        self.print_message("Calculating parameters using classical potential", limit=10)
        forces, potential_energy, potential_energies =\
            self.produce_classical_results(atoms=atoms)

        self.results["classical_forces"] = forces.copy()

        self.print_message(
            "Update the qm_cluster based on potential energies obtained from mm_pot calculation", limit=10)

        # If in warmup mode, do not use the clusters
        self.produce_qm_clusters(atoms,
                                 potential_energies=potential_energies)

        if (self.warmup):
            self.print_message("Warmup mode: not using any clusters", limit=10)
            self.cluster_list = []
            self.qm_cluster.reset_energized_list()
        self.print_message("Cluster list", limit=100)
        self.print_message(self.cluster_list, limit=100)

        # Perform parallel QM calculations on each cluster
        cluster_data_list = np.empty(len(self.cluster_list), dtype=object)
        if (len(cluster_data_list) > 0):
            if (self.doParallel):
                mcfm_parallel_control.get_cluster_data(atoms=atoms,
                                                       clusterData=cluster_data_list,
                                                       mcfm_pot=self)
            else:
                for i, cluster in enumerate(self.cluster_list):
                    # Evaluate qm cluster
                    cluster_data_list[i] = self.evaluate_qm_cluster_serial(
                        atoms=atoms, cluster=cluster, clusterNumber=i)

        # ------ Attach the cluster data list
        self.cluster_data_list = cluster_data_list

        # Stitch the forces using data from cluster_data_list
        forces = self.combine_qm_mm_forces(atoms=atoms,
                                           forces=forces,
                                           cluster_data_list=cluster_data_list)

        # Create the full QM list and QM mask
        full_qm_atoms_list = [item for sublist in self.cluster_list for item in sublist]
        full_qm_atoms_mask = np.zeros(len(atoms), dtype=bool)
        full_qm_atoms_mask[full_qm_atoms_list] = True

        # If the potential can update topology, do it.
        do_bond_change = (not (len(full_qm_atoms_list) == 0)) and\
            (self.change_bonds is True) and\
            hasattr(self.classical_calculator, "update_topology")
        if do_bond_change:
            self.print_message("Updating Topology!", limit=10)
            self.classical_calculator.update_topology(full_qm_atoms_list)

        # Mark atoms that are treated quantm mmechanically for more comprehensive ooutput
        self.attach_hybrid_data(
            atoms=atoms, full_qm_atoms_mask=full_qm_atoms_mask, cluster_data=cluster_data_list)

        # Compute stress
        if "stress" in properties:
            self.results["stress"] = self.compute_stress(atoms, forces)

        # Attach the updated version of atoms so that check-state would work properly.
        if self.enable_check_state:
            self.atoms = atoms.copy()

        self.results["forces"] = forces
        self.results["potential_energy"] = potential_energy
        self.results["energy"] = potential_energy
        self.results["potential_energies"] = potential_energies

        if (self.calculate_errors):
            self.evaluate_errors(atoms=atoms)

    def produce_classical_results(self, atoms=None):
        """Call the classical potential ot obtain forces, potential energy
        and potential energies per atom

        Parameters
        ----------
        atoms : ASE.atoms
            atoms object

        Returns
        -------
        forces : np.array
            Atomic forces
        potential_energy : np.array
            Potential energy of the system
        potential_energies : np.array
            Per atom potential energies
        """
        # Get forces
        forces = self.classical_calculator.get_forces(atoms)

        # Get potential energies
        if "potential_energies" in self.classical_calculator.results:
            potential_energies = self.classical_calculator.results["potential_energies"]
        else:
            potential_energies = self.classical_calculator.get_potential_energies(atoms)

        # Get total potential energy
        # (summing over individual contributions isusually faster then a calculation)
        if "energy" in self.classical_calculator.results:
            potential_energy = self.classical_calculator.results["energy"]
        elif "potential_energy" in self.classical_calculator.results:
            potential_energy = self.classical_calculator.results["potential_energy"]
        else:
            potential_energy = potential_energies.sum()

        return forces, potential_energy, potential_energies

    def produce_qm_clusters(self, atoms,
                            potential_energies=None):
        """Update qm clusters based on potential energies per atom

        Parameters
        ----------
        atoms : ASE.atoms
            atoms object
        potential_energies : np.array
            Per atom potential energies
        """
        if self.forced_qm_list is None:
            # Use the newly calculated values to find the quantum mechanical regions
            self.cluster_list = self.qm_cluster.update_qm_region(atoms,
                                                                 potential_energies=potential_energies
                                                                 )
        else:
            if len(self.forced_qm_list) == 0:
                self.cluster_list = []
            else:
                self.cluster_list = self.forced_qm_list
        # Safeguard against empty clusters
        self.cluster_list = [item for item in self.cluster_list if len(item) > 0]

    def evaluate_qm_cluster_serial(self, atoms=None, cluster=None, clusterNumber=0):
        """Evaluate forces for a single QM cluster given the buffer hops

        Parameters
        ----------
        atoms : ASE.atoms
            atoms object
        cluster : list
            list of core qm atoms
        clusterNumber : int
            cluster number

        Returns
        -------
        Cluster : cluster_data
            object with forces
                        qm_atoms mark
                        core qm list
        """
        self.print_message("Evaluating cluster", limit=100)
        self.print_message(cluster, limit=100)

        # Create and evaluate hybrid cluster
        self.print_message("Creating cluster", limit=10)

        atomic_cluster = self.qm_cluster.carve_cluster(atoms, cluster, buffer_hops=self.buffer_hops)
        self.print_message("Size of the atomic cluster: " + str(len(atomic_cluster)), limit=10)

        # Debug cluster carving by printing the structures and clusters
        if (self.debug_cluster_carving):
            self.print_message("Writing cluster to file", limit=10)
            extension = "_" + str(clusterNumber + 1) + ".xyz"
            ase.io.write("cluster" + extension, atomic_cluster, format="xyz")
            ase.io.write("cluster_ext" + extension, atomic_cluster, format="extxyz")
            ase.io.write("structure.xyz", atoms, format="xyz")
            ase.io.write("structure_ext.xyz", atoms, format="extxyz", write_results=False)

            if (self.clusterDebug_cluster_atoms is not None):
                ase.io.write(self.clusterDebug_cluster_atoms, atomic_cluster,
                             format="extxyz", write_results=False)
            if (self.clusterDebug_full_structure is not None):
                ase.io.write(self.clusterDebug_full_structure, atoms,
                             format="extxyz", write_results=False)

        self.print_message("Evaluating", limit=100)
        qm_forces_array = self.qm_calculator.get_forces(atomic_cluster)

        self.print_message("qmmm pot cluster, " + str(len(atomic_cluster)) + " atoms long", limit=100)
        self.print_message(atomic_cluster.arrays["orig_index"], limit=100)

        # Create a cluster data object with relevan values
        mark = np.zeros(len(atoms), dtype=int)
        full_qm_forces = np.zeros((len(atoms), 3))

        for i in range(atomic_cluster.info["no_quantum_atoms"]):
            orig_index = atomic_cluster.arrays["orig_index"][i]
            full_qm_forces[orig_index, :] = qm_forces_array[i, :]
            mark[orig_index] = atomic_cluster.arrays["cluster_mark"][i]

        cluster_data = ClusterData(len(atoms), mark, cluster, full_qm_forces)

        # Try to add additional details to the cluster data
        qm_charges = np.zeros(len(atoms))
        if (self.debug_qm_calculator):
            try:
                # print('eigenvalues cluster:')
                # self.qm_calculator.print_eigenvalues(scope=15, offset=0)

                qm_charges -= 10
                atomic_cluster.arrays["qm_charges"] = self.qm_calculator.results["charges"].copy()
                atomic_cluster.arrays["qm_forces"] = self.qm_calculator.results["forces"].copy()

                ase.io.write("structure_ext.xyz", atoms, format="extxyz", write_results=False)
                ase.io.write("cluster_ext" + extension, atomic_cluster,
                             format="extxyz", write_results=False)

                for i in range(atomic_cluster.info["no_quantum_atoms"]):
                    orig_index = atomic_cluster.arrays["orig_index"][i]
                    qm_charges[orig_index] = self.qm_calculator.results["charges"][i]

            except KeyError:
                pass

        cluster_data.qm_charges = qm_charges
        cluster_data.nClusterAtoms = len(atomic_cluster)
        return cluster_data

    def combine_qm_mm_forces(self, atoms=None, forces=None, cluster_data_list=None):
        """This combines QM and MM forces

        Parameters
        ----------
        atoms : ASE.atoms
            atoms object
        forces : np.array
            atomic forces
        cluster_data_list : list of matscipy.calculators.mcfm.ClusterData
            information about the clusters

        Returns
        -------
        forces : np.array
            atomic forces
        """
        if (self.verbose >= 1):
            atoms.arrays["classical_forces"] = forces.copy()
        self.raw_qm_cluster_forces = np.zeros_like(forces)

        # If any clusters present, combine QM/MM forces, otherwise just pass forces along
        if ((self.long_range_weight > 0.0) and (len(cluster_data_list) > 0)):
            self.print_message(
                "Splitting forces into bonding/longRange and combining with QM if needed.", limit=10)

            # Obtain bonding forces
            forcesLR = self.classical_calculator.get_pairwise_forces(atoms)
            # Calculate long range forces
            # forcesLR = forces - forcesB

            # Replace bonding forces with QM forces for QM atoms only
            for cluster_data in cluster_data_list:
                for aI in cluster_data.qm_list:

                    # Combine short ranged forces with a fraction of the long range ones
                    # The long range forces are there for stability, should not affect the dynamics much.
                    forces[aI, :] = cluster_data.forces[aI, :] + forcesLR[aI, :] * self.long_range_weight
                    self.raw_qm_cluster_forces[aI, :] = cluster_data.forces[aI, :]

            # Add long range forces to the output
            if (self.verbose >= 1):
                atoms.arrays["Long_range_forces"] = forcesLR * self.long_range_weight

        elif (len(cluster_data_list) > 0):
            self.print_message("Combining QM and MM forces.", limit=10)
            for cluster_data in cluster_data_list:
                for aI in cluster_data.qm_list:
                    forces[aI, :] = cluster_data.forces[aI, :]
                    self.raw_qm_cluster_forces[aI, :] = cluster_data.forces[aI, :]

        if (self.verbose >= 1):
            atoms.arrays["raw_qm_forces"] = self.raw_qm_cluster_forces

        if (self.conserve_momentum) and ((len(cluster_data_list) > 0)):
            avg_force = forces.mean(axis=0)
            forces -= avg_force

        return forces

    def attach_hybrid_data(self, atoms=None, full_qm_atoms_mask=None, cluster_data=None):
        """Mark atoms that are treated quantm mmechanically
        for more comprehensive ooutput

        Parameters
        ----------
        atoms : ASE.atoms
            atoms object
        full_qm_atoms_mask : list
            list of all qm atoms
        cluster_data_list : list of matscipy.calculators.mcfm.ClusterData
            information about the clusters
        """

        # Store infrmation on individual clusters in atoms file
        atoms.arrays["hybrid_clusters"] = np.zeros(len(atoms))

        index = 0
        for cluster in self.cluster_list:
            # Safeguard against empty, nested lists
            if len(cluster) == 0:
                continue
            atoms.arrays["hybrid_clusters"][cluster] = index + 1
            atoms.arrays["cluster_marks_" + str(index + 1)] = cluster_data[index].mark.copy()

            # Add information about qm caharges
            if (self.verbose >= 1):
                atoms.arrays["qm_charges_clus_" + str(index + 1)] = cluster_data[index].qm_charges.copy()

            index += 1

    def evaluate_errors(self, atoms=None, heavy_only=False, r_force=None):
        """Use the forces and reference forces to get errors on hybrid atom
        force evaluations

        Parameters
        ----------
        atoms : ASE.atoms
            atoms object
        heavy_only : bool
            Do not evaluate errors on hydrogens
        r_force : np.array
            array with reference forces
        """

        # Create the full QM list and QM mask
        full_qm_atoms_list = [item for sublist in self.cluster_list for item in sublist]
        full_qm_atoms_mask = np.zeros(len(atoms), dtype=bool)
        full_qm_atoms_mask[full_qm_atoms_list] = True

        forces = self.raw_qm_cluster_forces
        if (r_force is None):
            if len(full_qm_atoms_list) > 0:
                r_force = self.qm_calculator.get_forces(atoms)
            else:
                r_force = forces

        atoms.arrays["reference_qm_force"] = r_force.copy()
        atoms.arrays["simulation_force"] = self.results["forces"].copy()
        atoms.arrays["qmmm_raw_force"] = self.raw_qm_cluster_forces.copy()

        try:
            if (len(self.qm_calculator.results["charges"]) == len(atoms) and (self.verbose >= 1)):
                atoms.arrays["reference_qm_charges"] = self.qm_calculator.results["charges"].copy()
        except KeyError:
            pass

        # Calculate errors for the QM regions
        # Only evaluate errors on heavy atoms if flag is set
        if (heavy_only is True):
            for i in range(len(full_qm_atoms_mask)):
                if (atoms.numbers[i] == 1):
                    full_qm_atoms_mask[i] = False

        if (full_qm_atoms_mask.sum() > 0):
            f_errorFull = r_force - forces
            f_error = np.linalg.norm(abs(f_errorFull), ord=2, axis=1)

            f_errFull = np.zeros((len(atoms), 3))
            f_errFull[full_qm_atoms_mask] = f_errorFull[full_qm_atoms_mask]
            f_err = np.zeros(len(atoms))
            f_err[full_qm_atoms_mask] = f_error[full_qm_atoms_mask]

            # Calculate if the errors are dumping or extrcting energy from the system
            # According to E = Force * velocity * timestep
            # Taking timestep as constant = 1
            try:
                energyChange = np.einsum("ij, ij -> i", f_errFull, atoms.arrays["momenta"])
                totalEChange = np.sum(energyChange)

                totalEnergyVector = np.einsum("ij, ij -> i", forces[full_qm_atoms_mask],
                                              atoms.arrays["momenta"][full_qm_atoms_mask])
                totalEnergy = np.sum(totalEnergyVector)

                totalEChange /= totalEnergy
            except KeyError:
                totalEChange = 0

            # Get the relative force error
            cumulative_forces = np.linalg.norm(forces, ord=2, axis=1)
            cumulative_forces = np.mean(cumulative_forces[full_qm_atoms_mask])
            relative_error = np.divide(f_err, cumulative_forces)

            max_relative_error = relative_error[full_qm_atoms_mask].max()
            rms_relative_error = np.sqrt(np.mean(np.square(relative_error[full_qm_atoms_mask])))

            max_absolute_error = f_err[full_qm_atoms_mask].max()
            rms_absolute_error = np.sqrt(np.mean(np.square(f_err[full_qm_atoms_mask])))

            # Provide max and RMS relative error
            # print "\tRMS of absolute errors: %0.5f, MAX absolute error: %0.5f" %
            # (rms_absolute_error, max_absolute_error)

        else:
            f_err = np.zeros(len(atoms))
            f_errFull = np.zeros((len(atoms), 3))
            relative_error = np.zeros(len(atoms))
            max_absolute_error = 0
            rms_absolute_error = 0
            max_relative_error = 0
            rms_relative_error = 0
            totalEChange = 0

        try:
            self.errors["Cumulative fError vector"] += f_errFull
            self.errors["Cumulative energy change"] += totalEChange
        except KeyError:
            self.errors["Cumulative fError vector"] = f_errFull
            self.errors["Cumulative energy change"] = totalEChange
        self.errors["vector force error"] = f_errFull
        self.errors["Cumulative fError vector length"] =\
            np.linalg.norm(abs(self.errors["Cumulative fError vector"]), ord=2, axis=1)
        self.errors["energy Change"] = totalEChange

        self.errors["absolute force error"] = f_err
        self.errors["relative force error"] = relative_error
        self.errors["max absolute error"] = max_absolute_error
        self.errors["rms absolute error"] = rms_absolute_error
        self.errors["max relative error"] = max_relative_error
        self.errors["rms relative error"] = rms_relative_error
        self.errors["rms force"] = np.sqrt(np.mean(np.square(forces)))
        self.errors["no of QM atoms"] = full_qm_atoms_mask.sum()

        # Add the relative error th the atoms object for visualization
        atoms.arrays["relative_Ferror"] = self.errors["relative force error"].copy()
        atoms.arrays["absolute_Ferror"] = self.errors["absolute force error"].copy()

        # #Calculate errors of the full system
        # f_err = np.linalg.norm(abs(r_force - forces), ord = 2, axis = 1)
        # relative_error = np.divide(f_err,
        #                     np.linalg.norm(forces, ord = 2, axis = 1))
        # max_relative_error = relative_error.max()
        # rms_relative_error = np.sqrt(np.mean(np.square(relative_error)))

        # max_absolute_error = f_err.max()
        # rms_absolute_error = np.sqrt(np.mean(np.square(f_err)))

        # self.errors_full = {}
        # self.errors_full["absolute force error"] = f_err
        # self.errors_full["relative force error"] = relative_error
        # self.errors_full["max absolute error"] = max_absolute_error
        # self.errors_full["rms absolute error"] = rms_absolute_error
        # self.errors_full["max relative error"] = max_relative_error
        # self.errors_full["rms relative error"] = rms_relative_error
        # self.errors_full["rms force"] = np.sqrt(np.mean(np.square(forces)))

    def set_qm_atoms(self, qm_list, atoms=None):
        """Force a certian set of clusters for qmmm evaluation,
        If forced_qm_list is assigned, the cluster list is not updated
        throughout the run

        Parameters
        ----------
        qm_list : list
            list of atoms
        atoms : ASE.atoms
            atoms object
        """
        if qm_list is None:
            self.forced_qm_list = None
        else:
            self.forced_qm_list = [qm_list]

    def compute_stress(self, atoms, forces):
        """Compute total stresses using viral theorem.
        WARNING: only works for non-PBC structures

        the formula for stress evaluation is
        ------------------------------------
        Sij = sum_k (m_k v_ik v_jk)/ volume + sum_k (r_ik f_jk)/volume
        m: mass
        v: velocity
        r: position
        f: force
        where i,j are taken from {x, y, z}
        and sum_k represents a sum over all atoms

        Parameters
        ----------
        atoms : ASE.atoms
            atoms object
        forces : np.array
            atomic forces

        Returns
        -------
        stress : np.array
            stress tensof in matrix notation
        """

        stress_mat = np.zeros((3, 3))
        stress = np.zeros(6)

        vol = atoms.get_volume()
        velo = atoms.get_velocities()
        mom = atoms.get_momenta()
        pos = atoms.get_positions()
        f = forces

        for i in range(3):
            for j in range(3):
                stress_mat[i, j] = - np.dot(pos[:, i], f[:, j]) / vol

        stress[0] = stress_mat[0, 0]
        stress[1] = stress_mat[1, 1]
        stress[2] = stress_mat[2, 2]
        stress[3] = stress_mat[1, 2]
        stress[4] = stress_mat[0, 2]
        stress[5] = stress_mat[0, 1]

        return stress

    def print_message(self, message, limit=100):
        """Print a message if the calculators verbosity level is above the
        given threshold

        For now verbose levels are
        --------------------------
        0 - nothing is printed
        1 - Message when meth::calculate is called
        10 - Calculate steps are listed
        100 - Information about specific QM clusters
        (the default is 0)

        Parameters
        ----------
        message : str
            The message to be printed
        limit : int
            the verbosity threshold for this mesage
        """
        if (self.verbose >= limit):
            print(message)
