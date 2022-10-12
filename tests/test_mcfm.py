#
# Copyright 2020-2021 Lars Pastewka (U. Freiburg)
#           2018 Jacek Golebiowski (Imperial College London)
#           2018 golebiowski.j@gmail.com
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

import ase.io
from math import exp, sqrt
from ase.calculators.calculator import Calculator

from matscipy.calculators.mcfm.neighbour_list_mcfm.neighbour_list_mcfm import NeighbourListMCFM
from matscipy.calculators.mcfm.qm_cluster import QMCluster
from matscipy.calculators.mcfm.calculator import MultiClusterForceMixingPotential

import unittest
import matscipytest


###

clustering_ckeck = [[9, 10, 11, 12, 13, 14, 15, 16, 17, 19]]
clustering_marks_check = np.array([2, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2,
                                   2, 2, 2, 2, 3, 0, 3, 0])
neighbour_list_check = [
    np.array([4, 2, 1], dtype=int),
    np.array([0], dtype=int),
    np.array([0], dtype=int),
    np.array([7, 4, 5], dtype=int),
    np.array([3, 0], dtype=int),
    np.array([3], dtype=int),
    np.array([10, 7, 8], dtype=int),
    np.array([6, 3], dtype=int),
    np.array([6], dtype=int),
    np.array([13, 11, 10], dtype=int),
    np.array([9, 6], dtype=int),
    np.array([9], dtype=int),
    np.array([16, 13, 14], dtype=int),
    np.array([12, 9], dtype=int),
    np.array([12], dtype=int),
    np.array([19, 17, 16], dtype=int),
    np.array([15, 12], dtype=int),
    np.array([15], dtype=int),
    np.array([20, 22, 19], dtype=int),
    np.array([18, 15], dtype=int),
    np.array([18], dtype=int),
    np.array([25, 23, 22], dtype=int),
    np.array([21, 18], dtype=int),
    np.array([21], dtype=int),
    np.array([28, 26, 25], dtype=int),
    np.array([24, 21], dtype=int),
    np.array([24], dtype=int),
    np.array([29, 28], dtype=int),
    np.array([27, 24], dtype=int),
    np.array([27], dtype=int),
]


def load_atoms(filename):
    """Load atoms from the file"""
    atoms = ase.io.read(filename, index=0, format="extxyz")
    atoms.arrays["atomic_index"] = np.arange(len(atoms))
    return atoms


def create_neighbour_list(atoms):
    """Create the neighbour list"""
    hysteretic_cutoff_break_factor = 3
    cutoffs = dict()
    c_helper = dict(H=0.7,
                    C=1,
                    N=1,
                    O=1)

    for keyi in c_helper:
        for keyj in c_helper:
            cutoffs[(keyi, keyj)] = c_helper[keyi] + c_helper[keyj]

    neighbour_list = NeighbourListMCFM(atoms,
                                       cutoffs,
                                       skin=0.3,
                                       hysteretic_break_factor=hysteretic_cutoff_break_factor)
    neighbour_list.update(atoms)
    return neighbour_list


class MorsePotentialPerAtom(Calculator):
    """Morse potential.

    Default values chosen to be similar as Lennard-Jones.
    """

    implemented_properties = ['energy', 'forces', 'potential_energies']
    default_parameters = {'epsilon': 1.0,
                          'rho0': 6.0,
                          'r0': 1.0}
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):

        Calculator.calculate(self, atoms, properties, system_changes)
        epsilon = self.parameters.epsilon
        rho0 = self.parameters.rho0
        r0 = self.parameters.r0

        positions = self.atoms.get_positions()
        energy = 0.0
        energies = np.zeros(len(self.atoms))
        forces = np.zeros((len(self.atoms), 3))

        preF = 2 * epsilon * rho0 / r0
        for i1, p1 in enumerate(positions):
            for i2, p2 in enumerate(positions[:i1]):
                diff = p2 - p1
                r = sqrt(np.dot(diff, diff))
                expf = exp(rho0 * (1.0 - r / r0))
                energy += epsilon * expf * (expf - 2)
                energies[i1] += epsilon * expf * (expf - 2)
                energies[i2] += epsilon * expf * (expf - 2)
                F = preF * expf * (expf - 1) * diff / r
                forces[i1] -= F
                forces[i2] += F
        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['potential_energies'] = energies


def create_mcfm_potential(atoms,
                          classical_calculator=None,
                          qm_calculator=None,
                          special_atoms_list=None,
                          double_bonded_atoms_list=None
                          ):
    """Set up a multi cluster force mixing potential with a sensible set of defaults

    Parameters
    ----------
    atoms : ASE.atoms
        atoms object
    classical_calculator : AE.calculator
        classical calculator
    qm_calculator : ASE.calculator
        qm calculator
    special_atoms_list : list of ints (atomic indices)
        In case a group of special atoms are specified (special molecule),
        If one of these atoms is in the buffer region, the rest are also added to it.
    double_bonded_atoms_list : list
        list of doubly bonded atoms for the clustering module,
        needed for double hydrogenation.

    Returns
    -------
    MultiClusterForceMixingPotential
        ase.calculator supporting hybrid simulations.
    """

    if special_atoms_list is None:
        special_atoms_list = [[]]
    if double_bonded_atoms_list is None:
        double_bonded_atoms_list = []

    # Stage 1 - Neighbour List routines
    neighbour_list = create_neighbour_list(atoms)

    # Stage 1 - Set up QM clusters
    qm_flag_potential_energies = np.ones((len(atoms), 2), dtype=float) * 100
    atoms.arrays["qm_flag_potential_energies[in_out]"] = qm_flag_potential_energies.copy()

    qm_cluster = QMCluster(special_atoms_list=special_atoms_list,
                           verbose=0)

    qm_cluster.attach_neighbour_list(neighbour_list)
    qm_cluster.attach_flagging_module(qm_flag_potential_energies=qm_flag_potential_energies,
                                      small_cluster_hops=3,
                                      only_heavy=False,
                                      ema_parameter=0.01,
                                      energy_cap=1000,
                                      energy_increase=1)

    qm_cluster.attach_clustering_module(double_bonded_atoms_list=double_bonded_atoms_list)

    # Stage 1 - Set Up the multi cluster force mixing potential
    mcfm_pot = MultiClusterForceMixingPotential(atoms=atoms,
                                                classical_calculator=classical_calculator,
                                                qm_calculator=qm_calculator,
                                                qm_cluster=qm_cluster,
                                                forced_qm_list=None,
                                                change_bonds=True,
                                                calculate_errors=False,
                                                calculation_always_required=False,
                                                buffer_hops=6,
                                                verbose=0,
                                                enable_check_state=True
                                                )

    mcfm_pot.debug_qm = False
    mcfm_pot.conserve_momentum = False

    # ------ Parallel module makes this simple simulation extremely slow
    # ------ Due ot large overhead. Use only with QM potentials
    mcfm_pot.doParallel = False
    atoms.set_calculator(mcfm_pot)

    return mcfm_pot


###############################################################################
# ------ Actual tests
###############################################################################


class TestMCFM(matscipytest.MatSciPyTestCase):

    def prepare_data(self):
        self.atoms = load_atoms("carbon_chain.xyz")
        self.morse1 = MorsePotentialPerAtom(r0=2, epsilon=2, rho0=6)
        self.morse2 = MorsePotentialPerAtom(r0=2, epsilon=4, rho0=6)
        self.mcfm_pot = create_mcfm_potential(self.atoms,
                                              classical_calculator=self.morse1,
                                              qm_calculator=self.morse2,
                                              special_atoms_list=None,
                                              double_bonded_atoms_list=None)

    def test_neighbour_list(self):
        self.prepare_data()
        nl = create_neighbour_list(self.atoms)
        for idx in range(len(self.atoms)):
            self.assertTrue((neighbour_list_check[idx] == nl.neighbours[idx]).all())

    def test_clustering(self):
        self.prepare_data()
        self.mcfm_pot.qm_cluster.flagging_module.qm_flag_potential_energies[12, :] *= -20
        self.mcfm_pot.get_forces(self.atoms)

        clusters = self.mcfm_pot.cluster_list
        cluster_marks = self.mcfm_pot.cluster_data_list[0].mark

        self.assertTrue(list(cluster_marks) == list(clustering_marks_check))
        for idx in range(len(clusters)):
            self.assertTrue(clusters[idx].sort() == clustering_ckeck[idx].sort())

    def test_forces(self):
        self.prepare_data()
        self.mcfm_pot.qm_cluster.flagging_module.qm_flag_potential_energies[12, :] *= -20

        f = self.mcfm_pot.get_forces(self.atoms)
        cluster = self.mcfm_pot.cluster_list[0]
        fm1 = self.morse1.get_forces(self.atoms)
        fm2 = self.morse2.get_forces(self.atoms)

        for idx in range(len(self.atoms)):
            if idx in cluster:
                self.assertArrayAlmostEqual(f[idx, :], fm2[idx, :])
            else:
                self.assertArrayAlmostEqual(f[idx, :], fm1[idx, :])


###


if __name__ == '__main__':
    unittest.main()


###
