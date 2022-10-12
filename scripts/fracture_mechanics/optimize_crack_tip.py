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
import ase.optimize
from ase.data import atomic_numbers
from ase.units import GPa

import matscipy.fracture_mechanics.crack as crack
from matscipy import parameter
from matscipy.logger import screen

from setup_crack import setup_crack

###

import sys
sys.path += ['.', '..']
import params

###

Optimizer = ase.optimize.FIRE

# Atom types used for outputting the crack tip position.
ACTUAL_CRACK_TIP = 'Au'
FITTED_CRACK_TIP = 'Ag'

###

logger = screen

###

a, cryst, crk, k1g, tip_x, tip_y, bond1, bond2, boundary_mask, \
    boundary_mask_bulk, tip_mask = setup_crack(logger=logger)
ase.io.write('notch.xyz', a, format='extxyz')

# Get general parameters

basename = parameter('basename', 'crack_tip')
calc = parameter('calc')
fmax = parameter('fmax', 0.01)

# Get parameter used for fitting crack tip position

residual_func = parameter('residual_func', crack.displacement_residual)
_residual_func = residual_func
tip_tol = parameter('tip_tol', 1e-4)
tip_mixing_alpha = parameter('tip_mixing_alpha', 1.0)
write_trajectory_during_optimization = parameter('write_trajectory_during_optimization', False)

tip_x = (a.positions[bond1, 0] + a.positions[bond2, 0])/2
tip_y = (a.positions[bond1, 1] + a.positions[bond2, 1])/2
logger.pr('Optimizing tip position -> initially centering tip bond. '
          'Tip positions = {} {}'.format(tip_x, tip_y))

# Check if there is a request to restart from a positions file
restart_from = parameter('restart_from', 'N/A')
if restart_from != 'N/A':
    logger.pr('Restarting from {0}'.format(restart_from))
    a = ase.io.read(restart_from)
    # remove any marker atoms
    marker_mask = np.logical_and(a.numbers != atomic_numbers[ACTUAL_CRACK_TIP],
                                 a.numbers != atomic_numbers[FITTED_CRACK_TIP])
    a = a[marker_mask]
    tip_x, tip_y = crk.crack_tip_position(a.positions[:len(cryst),0],
                                          a.positions[:len(cryst),1],
                                          cryst.positions[:,0],
                                          cryst.positions[:,1],
                                          tip_x, tip_y,
                                          params.k1*k1g,
                                          mask=tip_mask[:len(cryst)],
                                          residual_func=residual_func)
    logger.pr('Optimizing tip position -> initially autodetected tip position. '
              'Tip positions = {} {}'.format(tip_x, tip_y))
else:
    tip_x = (a.positions[bond1, 0] + a.positions[bond2, 0])/2
    tip_y = (a.positions[bond1, 1] + a.positions[bond2, 1])/2
    logger.pr('Optimizing tip position -> initially centering tip bond. '
              'Tip positions = {} {}'.format(tip_x, tip_y))


# Assign calculator.
a.set_calculator(calc)

log_file = open('{0}.log'.format(basename), 'w')
if write_trajectory_during_optimization:
    traj_file = ase.io.NetCDFTrajectory('{0}.nc'.format(basename), mode='w',
                                        atoms=a)
    traj_file.write()
else:
    traj_file = None

# Deformation gradient residual needs full Atoms object and therefore
# special treatment here.
if _residual_func == crack.deformation_gradient_residual:
    residual_func = lambda r0, crack, x, y, ref_x, ref_y, k, mask=None:\
        _residual_func(r0, crack, x, y, a, ref_x, ref_y, cryst, k,
                       params.cutoff, mask)



old_x = tip_x
old_y = tip_y
converged = False
while not converged:
    #b = cryst.copy()
    u0x, u0y = crk.displacements(cryst.positions[:,0],
                                 cryst.positions[:,1],
                                 old_x, old_y, params.k1*k1g)
    ux, uy = crk.displacements(cryst.positions[:,0],
                               cryst.positions[:,1],
                               tip_x, tip_y, params.k1*k1g)

    a.set_constraint(None)
    # Displace atom positions
    a.positions[:len(cryst),0] += ux-u0x
    a.positions[:len(cryst),1] += uy-u0y
    a.positions[bond1,0] -= ux[bond1]-u0x[bond1]
    a.positions[bond1,1] -= uy[bond1]-u0y[bond1]
    a.positions[bond2,0] -= ux[bond2]-u0x[bond2]
    a.positions[bond2,1] -= uy[bond2]-u0y[bond2]
    # Set bond length and boundary atoms explicitly to avoid numerical drift
    a.positions[boundary_mask,0] = \
        cryst.positions[boundary_mask_bulk,0] + ux[boundary_mask_bulk]
    a.positions[boundary_mask,1] = \
        cryst.positions[boundary_mask_bulk,1] + uy[boundary_mask_bulk]
    # Fix outer boundary
    a.set_constraint(ase.constraints.FixAtoms(mask=boundary_mask))

    logger.pr('Optimizing positions...')
    opt = Optimizer(a, logfile=log_file)
    if traj_file:
        opt.attach(traj_file.write)
    opt.run(fmax=fmax)
    logger.pr('...done. Converged within {0} steps.' \
             .format(opt.get_number_of_steps()))
    old_x = tip_x
    old_y = tip_y
    tip_x, tip_y = crk.crack_tip_position(a.positions[:len(cryst),0],
                                          a.positions[:len(cryst),1],
                                          cryst.positions[:,0],
                                          cryst.positions[:,1],
                                          tip_x, tip_y,
                                          params.k1*k1g,
                                          mask=tip_mask[:len(cryst)],
                                          residual_func=residual_func)

    dtip_x = tip_x-old_x
    dtip_y = tip_y-old_y
    logger.pr('- Fitted crack tip (before mixing) is at {:3.2f} {:3.2f} '
             '(= {:3.2e} {:3.2e} from the former position).'.format(tip_x, tip_y, dtip_x, dtip_y))
    tip_x = old_x + tip_mixing_alpha*dtip_x
    tip_y = old_y + tip_mixing_alpha*dtip_y
    logger.pr('- New crack tip (after mixing) is at {:3.2f} {:3.2f} '
             '(= {:3.2e} {:3.2e} from the former position).'.format(tip_x, tip_y, tip_x-old_x, tip_y-old_y))
    converged = np.asscalar(abs(dtip_x) < tip_tol and abs(dtip_y) < tip_tol)

# Fit crack tip (again), and get residuals.
fit_x, fit_y, residuals = \
    crk.crack_tip_position(a.positions[:len(cryst),0],
                           a.positions[:len(cryst),1],
                           cryst.positions[:,0],
                           cryst.positions[:,1],
                           tip_x, tip_y, params.k1*k1g,
                           mask=tip_mask[:len(cryst)],
                           residual_func=residual_func,
                           return_residuals=True)

logger.pr('Measured crack tip at %f %f' % (fit_x, fit_y))

b = a.copy()

# The target crack tip is marked by a gold atom.
b += ase.Atom(ACTUAL_CRACK_TIP, (tip_x, tip_y, b.cell.diagonal()[2]/2))
b.info['actual_crack_tip'] = (tip_x, tip_y, b.cell.diagonal()[2]/2)

# The fitted crack tip is marked by a silver atom.
b += ase.Atom(FITTED_CRACK_TIP, (fit_x, fit_y, b.cell.diagonal()[2]/2))
b.info['fitted_crack_tip'] = (fit_x, fit_y, b.cell.diagonal()[2]/2)

b.info['energy'] = a.get_potential_energy()
b.info['cell_origin'] = [0, 0, 0]
ase.io.write('{0}.xyz'.format(basename), b, format='extxyz')

log_file.close()
if traj_file:
    traj_file.close()
