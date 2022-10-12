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

import random
import timeit
import os
import ase.io
from ..cluster_data import ClusterData


random.seed(123)


def worker_populate_cluster_data(rank, size,
                                 nProcLocal=None,
                                 atomic_cluster=None,
                                 clusterIndexes=None,
                                 nAtoms=None,
                                 qmCalculator=None,
                                 sharedList=None,
                                 debug_qm_calculator=False):
    """Function to calcuate total energy with TB

    Parameters
    ----------
    rank : int
        process number
    size : int
        total number of processes
    nProcLocal : int
        number of CPUS to be used for this calculation
    atomic_cluster : ASE.atoms
        Stucture on which ot perform the evaluation
    clusterIndexes : np.array
        list with indexes of different cluster atoms
    nAtoms : int
        number of atoms in the cluster
    qmCalculator : ASE.calculator
        calculator to be used for the evaluation
    sharedList : list
        mp shared list used ot store output data
    debug_qm_calculator : bool
        run the simulation in debug mode
    """

    # ------ MultiProcessing library pickes all objects and
    # ------ each workr thread recieves a copy

    # If a caluclator has the options, set parallel parameters
    try:
        # Create a new calculation seed
        qmCalculator.calculationSeed = str(int(random.random() * 1e7)) + str(rank)
        # Set OMP values for the potential
        qmCalculator.omp_set_threads = True
        qmCalculator.omp_num_threads = nProcLocal
    except AttributeError:
        pass

    # Create a cluster data object with relevan values
    mark = np.zeros(nAtoms, dtype=int)
    full_qm_forces = np.zeros((nAtoms, 3))

    if (debug_qm_calculator):
        ase.io.write("cluster_ext_" + str(rank) + ".xyz", atomic_cluster, format="extxyz")

    # ------ Run the calculation
    if (debug_qm_calculator):
        print("Starting evaluation of cluster %d :parallelTime" %
              (rank))
    t0 = timeit.default_timer()
    qm_forces_array = qmCalculator.get_forces(atomic_cluster)
    t1 = timeit.default_timer()
    if (debug_qm_calculator):
        print("Time taken for cluster %d: %.7e  w %d atoms :parallelTime" %
              (rank, (t1 - t0), len(atomic_cluster)))

    for i in range(atomic_cluster.info["no_quantum_atoms"]):
        orig_index = atomic_cluster.arrays["orig_index"][i]
        full_qm_forces[orig_index, :] = qm_forces_array[i, :]
        mark[orig_index] = atomic_cluster.arrays["cluster_mark"][i]

    cluster_data = ClusterData(nAtoms, mark, clusterIndexes, full_qm_forces)
    # Try to add additional details to the cluster data
    cluster_data.nClusterAtoms = len(atomic_cluster)
    qm_charges = np.zeros(nAtoms) - 10
    try:
        for i in range(atomic_cluster.info["no_quantum_atoms"]):
            orig_index = atomic_cluster.arrays["orig_index"][i]
            qm_charges[orig_index] = qmCalculator.results["charges"][i]
    except KeyError:
        pass
    cluster_data.qm_charges = qm_charges
    sharedList[rank] = cluster_data

    if (debug_qm_calculator):
        try:
            atomic_cluster.arrays["qm_charges"] = qmCalculator.results["charges"].copy()
            atomic_cluster.arrays["qm_forces"] = qmCalculator.results["forces"].copy()
        except KeyError:
            pass
        ase.io.write("cluster_ext_" + str(rank) + ".xyz", atomic_cluster, format="extxyz")
