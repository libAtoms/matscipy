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
"""This script sets up a simulation of a 10 monomer polypropylene chain"""

import numpy as np

import ase.io
from utilities import MorsePotentialPerAtom

from matscipy.calculators.mcfm.neighbour_list_mcfm.neighbour_list_mcfm import NeighbourListMCFM
from matscipy.calculators.mcfm.qm_cluster import QMCluster
from matscipy.calculators.mcfm.calculator import MultiClusterForceMixingPotential


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
    ########################################################################
    # Stage 1 - Neighbour List routines
    ########################################################################
    neighbour_list = create_neighbour_list(atoms)

    ########################################################################
    # Stage 1 - Set up QM clusters
    ########################################################################
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
    ########################################################################
    # Stage 1 - Set Up the multi cluster force mixing potential
    ########################################################################

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
    mcfm_pot.conserve_momentum = True

    # ------ Parallel module makes this simple simulation extremely slow
    # ------ Due ot large overhead. Use only with QM potentials
    mcfm_pot.doParallel = False
    atoms.set_calculator(mcfm_pot)

    return mcfm_pot


def main():
    atoms = load_atoms("structures/carbon_chain.xyz")
    nl = create_neighbour_list(atoms)

    for idx in range(15):
        print("Atom: {}, neighbours: {}".format(idx, nl.neighbours[idx]))

    print("[")
    for idx in range(len(atoms)):
        print("np.{},".format(repr(nl.neighbours[idx])))
    print("]")

    morse1 = MorsePotentialPerAtom(r0=2, epsilon=2, rho0=6)
    morse2 = MorsePotentialPerAtom(r0=2, epsilon=4, rho0=6)
    mcfm_pot = create_mcfm_potential(atoms,
                                     classical_calculator=morse1,
                                     qm_calculator=morse2,
                                     special_atoms_list=None,
                                     double_bonded_atoms_list=None)


if (__name__ == "__main__"):
    main()
