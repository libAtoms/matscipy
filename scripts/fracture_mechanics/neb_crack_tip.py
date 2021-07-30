#
# Copyright 2016, 2021 Lars Pastewka (U. Freiburg)
#           2016 James Kermode (Warwick U.)
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

# USAGE:
#
# Code imports the file 'params.py' from current working directory. params.py
# contains simulation parameters. Some parameters can be omitted, see below.

import os
import sys

import numpy as np

import ase
import ase.constraints
import ase.io
#import ase.optimize.precon

from ase.neb import NEB

from ase.data import atomic_numbers
from ase.units import GPa

import matscipy.fracture_mechanics.crack as crack
from matscipy import parameter
from matscipy.logger import screen
from matscipy.io import savetbl

from setup_crack import setup_crack

from scipy.integrate import cumtrapz

###

import sys
sys.path += ['.', '..']
import params

###

Optimizer = ase.optimize.FIRE
#Optimizer = ase.optimize.precon.LBFGS

# Atom types used for outputting the crack tip position.
ACTUAL_CRACK_TIP = 'Au'
FITTED_CRACK_TIP = 'Ag'

###

logger = screen

###

# Get general parameters

residual_func = parameter('residual_func', crack.displacement_residual)
_residual_func = residual_func

basename = parameter('basename', 'neb')
calc = parameter('calc')
fmax_neb = parameter('fmax_neb', 0.1)
maxsteps_neb = parameter('maxsteps_neb', 100)
Nimages = parameter('Nimages', 7)
k_neb = parameter('k_neb', 1.0)

a, cryst, crk, k1g, tip_x, tip_y, bond1, bond2, boundary_mask, \
    boundary_mask_bulk, tip_mask = setup_crack(logger=logger)

# Deformation gradient residual needs full Atoms object and therefore
# special treatment here.
if _residual_func == crack.deformation_gradient_residual:
    residual_func = lambda r0, crack, x, y, ref_x, ref_y, k, mask=None:\
        _residual_func(r0, crack, x, y, a, ref_x, ref_y, cryst, k,
                       params.cutoff, mask)

# Groups
g = a.get_array('groups')

dirname = os.path.basename(os.getcwd())
initial = ase.io.read('../../initial_state/{0}/initial_state.xyz'.format(dirname))

final_pos = ase.io.read('../../final_state/{0}/final_state.xyz'.format(dirname))
final = initial.copy()
final.set_positions(final_pos.positions)

# remove marker atoms
mask = np.logical_and(initial.numbers != atomic_numbers[ACTUAL_CRACK_TIP],
                      initial.numbers != atomic_numbers[FITTED_CRACK_TIP])

initial = initial[mask]
final = final[mask]

images = [initial] + [initial.copy() for i in range(Nimages-2)] + [final]
neb = NEB(images)
neb.interpolate()

fit_x, fit_y, _ = initial.info['fitted_crack_tip']

initial_tip_x  = []
initial_tip_y = []
for image in neb.images:
    image.set_calculator(calc)
    image.set_constraint(ase.constraints.FixAtoms(mask=boundary_mask))

    # Fit crack tip (again), and get residuals.
    fit_x, fit_y, residuals = \
        crk.crack_tip_position(image.positions[:len(cryst),0],
                               image.positions[:len(cryst),1],
                               cryst.positions[:,0],
                               cryst.positions[:,1],
                               fit_x, fit_y, params.k1*k1g,
                               mask=tip_mask[:len(cryst)],
                               residual_func=residual_func,
                               return_residuals=True)
    initial_tip_x += [fit_x]
    initial_tip_y += [fit_y]
    logger.pr('Fitted crack tip at %.3f %.3f' % (fit_x, fit_y))

nebfile = open('neb-initial.xyz', 'w')
for image in neb.images:
    ase.io.write(nebfile, image, format='extxyz')
nebfile.close()

opt = Optimizer(neb)
opt.run(fmax_neb, maxsteps_neb)

nebfile = open('neb-final.xyz', 'w')
for image in neb.images:
    ase.io.write(nebfile, image, format='extxyz')
nebfile.close()

fit_x, fit_y, _ = initial.info['fitted_crack_tip']
last_a = None
tip_x = []
tip_y = []
work = []
epot_cluster = []
bond_force = []
bond_length = []

for image in neb.images:
    # Bond length.
    dr = image[bond1].position - image[bond2].position
    bond_length += [ np.linalg.norm(dr) ]
    #assert abs(bond_length[-1]-image.info['bond_length']) < 1e-6

    # Get stored potential energy.
    epot_cluster += [ image.get_potential_energy() ]

    forces = image.get_forces(apply_constraint=False)
    df = forces[bond1, :] - forces[bond2, :]
    bond_force += [ 0.5 * np.dot(df, dr)/np.sqrt(np.dot(dr, dr)) ]

    # Fit crack tip (again), and get residuals.
    fit_x, fit_y, residuals = \
        crk.crack_tip_position(image.positions[:len(cryst),0],
                               image.positions[:len(cryst),1],
                               cryst.positions[:,0],
                               cryst.positions[:,1],
                               fit_x, fit_y, params.k1*k1g,
                               mask=tip_mask[:len(cryst)],
                               residual_func=residual_func,
                               return_residuals=True)

    logger.pr('Fitted crack tip at %.3f %.3f' % (fit_x, fit_y))
    tip_x += [fit_x]
    tip_y += [fit_y]

    # Work due to moving boundary.
    if last_a is None:
        work += [ 0.0 ]
    else:
        last_forces = last_a.get_forces(apply_constraint=False)
        # This is the trapezoidal rule.
        work += [ np.sum(0.5 * (forces[g==0,:]+last_forces[g==0,:]) *
                         (image.positions[g==0,:]-last_a.positions[g==0,:])
                          ) ]
    last_a = image
    last_tip_x = fit_x
    last_tip_y = fit_y

epot_cluster = np.array(epot_cluster)-epot_cluster[0]
print('epot_cluster', epot_cluster)

work = np.cumsum(work)
print('work =', work)

tip_x = np.array(tip_x)
tip_y = np.array(tip_y)

# Integrate bond force potential energy
epot = -cumtrapz(bond_force, bond_length, initial=0.0)
print('epot =', epot)

print('epot_cluster + work', epot_cluster + work)

savetbl('{}_eval.out'.format(basename),
        bond_length=bond_length,
        bond_force=bond_force,
        epot=epot,
        epot_cluster=epot_cluster,
        work=work,
        tip_x=tip_x,
        tip_y=tip_y,
        initial_tip_x=initial_tip_x,
        initial_tip_y=initial_tip_y)
