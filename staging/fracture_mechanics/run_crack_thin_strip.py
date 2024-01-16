#
# Copyright 2016 Punit Patel (Warwick U.)
#           2014 James Kermode (Warwick U.)
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

# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
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
# ======================================================================

"""
Script to run classical molecular dynamics for a crack slab,
incrementing the load in small steps until fracture starts.

James Kermode <james.kermode@kcl.ac.uk>
August 2013
"""

import numpy as np

import ase.io
import ase.units as units

from ase.constraints import FixAtoms
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.netcdftrajectory import NetCDFTrajectory

from matscipy.fracture_mechanics.crack import (get_strain,
                                               get_energy_release_rate,
                                               ConstantStrainRate,
                                               find_tip_stress_field)
import sys
sys.path.insert(0, '.')
import params

# ********** Read input file ************

print 'Loading atoms from file "crack.xyz"'
atoms = ase.io.read('crack.xyz')

orig_height = atoms.info['OrigHeight']
orig_crack_pos = atoms.info['CrackPos'].copy()

# ***** Setup constraints *******

top = atoms.positions[:, 1].max()
bottom = atoms.positions[:, 1].min()
left = atoms.positions[:, 0].min()
right = atoms.positions[:, 0].max()

# fix atoms in the top and bottom rows
fixed_mask = ((abs(atoms.positions[:, 1] - top) < 1.0) |
              (abs(atoms.positions[:, 1] - bottom) < 1.0))
fix_atoms = FixAtoms(mask=fixed_mask)
print('Fixed %d atoms\n' % fixed_mask.sum())

# Increase epsilon_yy applied to all atoms at constant strain rate

strain_atoms = ConstantStrainRate(orig_height,
                                  params.strain_rate*params.timestep)

atoms.set_constraint(fix_atoms)

atoms.set_calculator(params.calc)

# ********* Setup and run MD ***********

# Set the initial temperature to 2*simT: it will then equilibriate to
# simT, by the virial theorem
MaxwellBoltzmannDistribution(atoms, 2.0*params.sim_T)

# Initialise the dynamical system
dynamics = VelocityVerlet(atoms, params.timestep)

# Print some information every time step
def printstatus():
    if dynamics.nsteps == 1:
        print """
State      Time/fs    Temp/K     Strain      G/(J/m^2)  CrackPos/A D(CrackPos)/A
---------------------------------------------------------------------------------"""

    log_format = ('%(label)-4s%(time)12.1f%(temperature)12.6f'+
                  '%(strain)12.5f%(G)12.4f%(crack_pos_x)12.2f    (%(d_crack_pos_x)+5.2f)')

    atoms.info['label'] = 'D'                  # Label for the status line
    atoms.info['time'] = dynamics.get_time()/units.fs
    atoms.info['temperature'] = (atoms.get_kinetic_energy() /
                                 (1.5*units.kB*len(atoms)))
    atoms.info['strain'] = get_strain(atoms)
    atoms.info['G'] = get_energy_release_rate(atoms)/(units.J/units.m**2)

    crack_pos = find_tip_stress_field(atoms)
    atoms.info['crack_pos_x'] = crack_pos[0]
    atoms.info['d_crack_pos_x'] = crack_pos[0] - orig_crack_pos[0]

    print log_format % atoms.info


dynamics.attach(printstatus)

# Check if the crack has advanced enough and apply strain if it has not
def check_if_crack_advanced(atoms):
    crack_pos = find_tip_stress_field(atoms)

    # strain if crack has not advanced more than tip_move_tol
    if crack_pos[0] - orig_crack_pos[0] < params.tip_move_tol:
        strain_atoms.apply_strain(atoms)

dynamics.attach(check_if_crack_advanced, 1, atoms)

# Save frames to the trajectory every `traj_interval` time steps
trajectory = NetCDFTrajectory(params.traj_file, mode='w')
def write_frame(atoms):
    trajectory.write(atoms)

dynamics.attach(write_frame, params.traj_interval, atoms)

# Start running!
dynamics.run(params.nsteps)
