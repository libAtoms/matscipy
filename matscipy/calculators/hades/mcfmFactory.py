"""Set up a multi cluster force mixing potential with a sensible set of defaults"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from .neighbour_list.neighbour_list import NeighborListHysteretic
from .qm_cluster import QMcluster
from .multi_cluster_force_mixing import MultiClusterForceMixingPotential


def create_mcfm_potential(atoms,
                          classical_calculator=None,
                          qm_calculator=None,
                          bondfile_name=None,
                          special_atoms_list=[[]],
                          double_bonded_atoms_list=[]
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
    bondfile_name: str
        Name of the lammps structure file if a list is to be build from it, None if
        it is to be build from atomic positions
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

    ########################################################################
    # Stage 1 - Neighbour List routines
    ########################################################################

    hysteretic_cutoff_break_factor = np.ones(len(atoms))
    cutoffs = np.ones(len(atoms))
    maxNeighbours = np.ones(len(atoms), dtype=int)

    for i in range(len(atoms)):
        # ------ Hydrogen
        if atoms.numbers[i] == 1:
            cutoffs[i] *= 0.7
            hysteretic_cutoff_break_factor[i] *= 3.0
            maxNeighbours[i] *= 8
        # ------ Carbon
        elif atoms.numbers[i] == 6:
            cutoffs[i] *= 1.0
            hysteretic_cutoff_break_factor[i] *= 5.0
            maxNeighbours[i] *= 8
        # ------ Nitrogen
        elif atoms.numbers[i] == 7:
            cutoffs[i] *= 1.0
            hysteretic_cutoff_break_factor[i] *= 5.0
            maxNeighbours[i] *= 8
        # ------ Oxygen
        elif atoms.numbers[i] == 8:
            cutoffs[i] *= 1.0
            hysteretic_cutoff_break_factor[i] *= 5.0
            maxNeighbours[i] *= 8
        # ------ Other
        else:
            cutoffs[i] *= 1
            hysteretic_cutoff_break_factor[i] *= 5.0
            maxNeighbours[i] *= 8

    neighbour_list = NeighborListHysteretic(atoms,
                                            cutoffs,
                                            skin=0.3,
                                            sorted=False,
                                            self_interaction=False,
                                            bothways=True,
                                            max_neighbours=maxNeighbours,
                                            hysteretic_break_factor=hysteretic_cutoff_break_factor,
                                            bondfile_name=bondfile_name)

    neighbour_list.update(atoms)
    ########################################################################
    # Stage 1 - Set up QM clusters
    ########################################################################

    qm_flag_potential_energies = np.ones((len(atoms), 2), dtype=float) * 10000

    atoms.arrays["qm_flag_potential_energies[in_out]"] = qm_flag_potential_energies.copy()

    qm_cluster = QMcluster(special_atoms_list=special_atoms_list,
                           verbose=0)

    qm_cluster.attach_neighbour_list(neighbour_list)
    qm_cluster.attach_flagging_module(qm_flag_potential_energies=qm_flag_potential_energies,
                                      small_cluster_hops=3,
                                      only_heavy=False,
                                      ema_parameter=0.01,
                                      energy_cap=1000,
                                      energy_increase=100)

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
                                                enable_check_state=False
                                                )

    mcfm_pot.long_range_weight = 0.0
    mcfm_pot.debug_tight_binding = False
    mcfm_pot.doParallel = True
    mcfm_pot.conserve_momentum = False

    atoms.set_calculator(mcfm_pot)

    return mcfm_pot
