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
import ase.units as units
import ase.constraints
from ase.optimize import FIRE

from utilities import (MorsePotentialPerAtom, LinearConstraint,
                       trajectory_writer, mcfm_thermo_printstatus, mcfm_error_logger)
from simulation_setup import load_atoms, create_neighbour_list, create_mcfm_potential

from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


def get_dynamics(atoms, mcfm_pot, T=300):
    # ------ Set up logfiles
    traj_interval = 10
    thermoPrintstatus_log = open("log_thermoPrintstatus.log", "w")
    outputTrajectory = open("log_trajectory.xyz", "w")
    mcfmError_log = open("log_mcfmError.log", "w")

    # ------ Let optimiser use the base potential and relax the per atom energies
    mcfm_pot.qm_cluster.flagging_module.ema_parameter = 0.1
    mcfm_pot.qm_cluster.flagging_module.qm_flag_potential_energies =\
        np.ones((len(atoms), 2), dtype=float) * 1001

    # ------ Minimize positions
    opt = FIRE(atoms)
    optimTrajectory = open("log_optimizationTrajectory.xyz", "w")
    opt.attach(trajectory_writer, 1, atoms, optimTrajectory, writeResults=True)
    opt.run(fmax=0.05, steps=1000)

    # ------ Define ASE dyamics
    sim_T = T * units.kB
    MaxwellBoltzmannDistribution(atoms, 2 * sim_T)

    timestep = 5e-1 * units.fs
    friction = 1e-2
    dynamics = Langevin(atoms, timestep, sim_T, friction, fixcm=False)
    dynamics.attach(mcfm_thermo_printstatus, 100, 100, dynamics,
                    atoms, mcfm_pot, logfile=thermoPrintstatus_log)
    dynamics.attach(trajectory_writer, traj_interval, atoms, outputTrajectory, writeResults=False)
    dynamics.attach(mcfm_error_logger, 100, 100, dynamics, atoms, mcfm_pot, logfile=mcfmError_log)
    return dynamics


def main():
    atoms = load_atoms("structures/carbon_chain.xyz")

    morse1 = MorsePotentialPerAtom(r0=3, epsilon=2, rho0=6)
    morse2 = MorsePotentialPerAtom(r0=3, epsilon=4, rho0=6)
    mcfm_pot = create_mcfm_potential(atoms,
                                     classical_calculator=morse1,
                                     qm_calculator=morse2,
                                     special_atoms_list=None,
                                     double_bonded_atoms_list=None)

    dynamics = get_dynamics(atoms, mcfm_pot, T=100)
    # ------ Add constraints
    fixed = 29
    moving = 1
    direction = np.array([0, 0, 1])
    velocity = 5e-4 * units.Ang

    c_fixed = ase.constraints.FixAtoms([fixed])
    c_relax = ase.constraints.FixAtoms([moving])
    c_moving = LinearConstraint(moving, direction, velocity)

    atoms.set_constraint([c_fixed, c_moving])

    # ------ Adjust flagging energies
    mcfm_pot.qm_cluster.flagging_module.ema_parameter = 0.003
    mcfm_pot.qm_cluster.flagging_module.qm_flag_potential_energies =\
        np.ones((len(atoms), 2), dtype=float)

    mcfm_pot.qm_cluster.flagging_module.qm_flag_potential_energies[:, 0] *= -5
    mcfm_pot.qm_cluster.flagging_module.qm_flag_potential_energies[:, 1] *= -6
    atoms.arrays["qm_flag_potential_energies[in_out]"] =\
        mcfm_pot.qm_cluster.flagging_module.qm_flag_potential_energies

    # ------ Run dynamics
    dynamics.run(14000)
    atoms.set_constraint([c_fixed, c_relax])
    dynamics.run(3000)


if (__name__ == "__main__"):
    main()
