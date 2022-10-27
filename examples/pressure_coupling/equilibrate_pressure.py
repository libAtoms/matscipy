#
# Copyright 2021 Lars Pastewka (U. Freiburg)
#           2020, 2022 Thomas Reichenbach (Fraunhofer IWM)
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
from ase.io import Trajectory
from ase.units import GPa, fs
import numpy as np
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from matscipy import pressurecoupling as pc
from io import open

# Parameters
dt = 1.0 * fs  # MD time step
C11 = 500.0 * GPa  # material constant
M_factor = 1.0  # scaling factor for lid mass during equilibration,
# 1.0 will give fast equilibration for expensive calculators

Pdir = 2  # index of cell axis along normal pressure is applied
P = 5.0 * GPa  # target normal pressure
v = 0.0  # no sliding yet, only apply pressure
vdir = 0  # index of cell axis along sliding happens
T = 300.0  # target temperature for thermostat
# thermostat is applied in the third direction which
# is neither pressure nor sliding direction and only
# in the middle region between top and bottom.
# This makes sense for small systems which cannot have
# a dedicated thermostat region.

t_langevin = 75.0 * fs  # time constant for Langevin thermostat
gamma_langevin = 1. / t_langevin  # derived Langevin parameter
t_integrate = 1000.0 * fs  # simulation time
steps_integrate = int(t_integrate / dt)  # number of simulation steps

atoms = ASE_ATOMS_OBJECT  # put a specific system here
bottom_mask = BOOLEAN_NUMPY_ARRAY_TRUE_FOR_FIXED_BOTTOM_ATOMS  # adjust
top_mask = BOOLEAN_NUMPY_ARRAY_TRUE_FOR_CONSTRAINT_TOP_ATOMS  # adjust

# save masks for sliding simulations or restart runs
np.savetxt("bottom_mask.txt", bottom_mask)
np.savetxt("top_mask.txt", top_mask)

# set up calculation:
damp = pc.FixedMassCriticalDamping(C11, M_factor)
slider = pc.SlideWithNormalPressureCuboidCell(top_mask, bottom_mask, Pdir,
                                              P, vdir, v, damp)
atoms.set_constraint(slider)
# if we start from local minimum, zero potential energy,
# use double temperature for faster temperature convergence in the beginning:
MaxwellBoltzmannDistribution(atoms, temperature_K=2 * T)
# clear momenta in constraint regions, otherwise lid might run away
atoms.arrays['momenta'][top_mask, :] = 0
atoms.arrays['momenta'][bottom_mask, :] = 0

calc = ASE_CALCULATOR_OBJECT  # put a specific calculator here

atoms.set_calculator(calc)

# only thermalize middle region in one direction
temps = np.zeros((len(atoms), 3))
temps[slider.middle_mask, slider.Tdir] = T
gammas = np.zeros((len(atoms), 3))
gammas[slider.middle_mask, slider.Tdir] = gamma_langevin

integrator = Langevin(atoms, dt, temperature_K=temps,
                      friction=gammas, fixcm=False)
trajectory = Trajectory('equilibrate_pressure.traj', 'w', atoms)
log_handle = open('log_equilibrate.txt',
                  'w', 1, encoding='utf-8')  # 1 means line buffered
logger = pc.SlideLogger(log_handle, atoms, slider, integrator)
# log can be read using pc.SlideLog (see docstring there)
logger.write_header()
integrator.attach(logger)
integrator.attach(trajectory)
integrator.run(steps_integrate)
log_handle.close()
trajectory.close()
