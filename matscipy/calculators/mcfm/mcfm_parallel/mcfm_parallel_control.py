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

import os
import time
import multiprocessing as mp
from . import mcfm_parallel_worker as mpw


def get_cluster_data(atoms=None,
                     clusterData=None,
                     mcfm_pot=None):
    """Obtain a list of cluster data with calculations being done in parallel

    Parameters
    ----------
    atoms : ase.Atoms
        atoms object representing the structure
    clusterData : list
        List of empty objects to be filled with clusterData instances
    mcfm_pot : matscipy.calculators.mcfm.MultiClusterForceMixing
        qmmm potential
    """
    # number of porcessors
    try:
        nProc = int(os.environ["OMP_NUM_THREADS"])
    except KeyError:
        nProc = mp.cpu_count() / 2

    # number of threads - number of clusters
    numThreads = len(mcfm_pot.cluster_list)

    # In case there are not enough cpu's,
    # have the number of processes artificially increased
    if (numThreads > nProc):
        nProc = numThreads

    # Create atomic clusters and evaluate their sizes
    atomicClustersList = []

    for cluster in mcfm_pot.cluster_list:
        atomicCluster = mcfm_pot.qm_cluster.carve_cluster(atoms,
                                                          cluster,
                                                          buffer_hops=mcfm_pot.buffer_hops)
        atomicClustersList.append(atomicCluster)

    # ------ Evaluate work balancing
    valenceElectrons = [np.sum(np.abs(item.numbers - 2)) for item in atomicClustersList]
    fractionWorkloadPerCluster = [(item ** 2) for item in valenceElectrons]
    totalWorkload = sum(fractionWorkloadPerCluster)
    fractionWorkloadPerCluster = [item / totalWorkload for item in fractionWorkloadPerCluster]

    nProcPerCluster = [1 for item in atomicClustersList]
    leftoverProcs = nProc - sum(nProcPerCluster)
    # Distribute leftoverProcs
    for i in range(numThreads):
        nProcPerCluster[i] += int(fractionWorkloadPerCluster[i] * leftoverProcs)

    # Disribute leftover procs (if any)
    leftoverProcs = nProc - sum(nProcPerCluster)
    running = True
    while running:
        for i in np.argsort(fractionWorkloadPerCluster)[::-1]:
            if (leftoverProcs <= 0):
                running = False
                break
            nProcPerCluster[i] += 1
            leftoverProcs -= 1

    if (mcfm_pot.debug_qm_calculator):
        print(fractionWorkloadPerCluster, nProcPerCluster, ":parallelTime")

    # Set up the Manager
    mpManager = mp.Manager()
    sharedList = mpManager.list(list(range(numThreads)))

    # Setup a list of processes that we want to run
    processes = []
    for rank in range(numThreads):
        p = mp.Process(target=mpw.worker_populate_cluster_data,
                       name=None,
                       args=(rank, numThreads),
                       kwargs=dict(nProcLocal=nProcPerCluster[rank],
                                   atomic_cluster=atomicClustersList[rank],
                                   clusterIndexes=mcfm_pot.cluster_list[rank],
                                   nAtoms=len(atoms),
                                   qmCalculator=mcfm_pot.qm_calculator,
                                   sharedList=sharedList,
                                   debug_qm_calculator=mcfm_pot.debug_qm_calculator))
        processes.append(p)

    # Run processes
    for p in processes:
        p.start()
        # Each QM calculation takes between 1 and 100s so the wait shouldnt affect the performance,
        # It helps prevent any I/O clashes when setting up the simulations
        time.sleep(1e-3)

    # Exit the completed processes
    for p in processes:
        p.join()

    # Extract results
    for index in range(len(clusterData)):
        clusterData[index] = sharedList[index]
