#
# Copyright 2014-2015, 2021 Lars Pastewka (U. Freiburg)
#           2015 James Kermode (Warwick U.)
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
#
# Parameters
# ----------
# calc : ase.Calculator
#     Calculator object for energy and force computation.
# tip_x0 : float
#     Initial x-position of crack tip.
# tip_y0 : float
#     Initial y-position of crack tip.
# tip_dx : array-like
#     Displacement of tip in x-direction during run. x- and y-positions will be
#     optimized self-consistently if omitted.
# tip_dy : array-like
#     Displacement of tip in y-direction during run. Position will be optimized
#     self-consistently if omitted.

import os
import sys

import numpy as np

import ase
import ase.io
import ase.constraints
import ase.optimize
from ase.units import GPa

import matscipy.fracture_mechanics.crack as crack
from matscipy import parameter
from matscipy.logger import screen

from setup_crack import setup_crack

###

sys.path += ['.', '..']
import params

###

# Atom types used for outputting the crack tip position.
ACTUAL_CRACK_TIP = 'Au'
FITTED_CRACK_TIP = 'Ag'

###

logger = screen

###

a, cryst, crk, k1g, tip_x0, tip_y0, bond1, bond2, boundary_mask, \
    boundary_mask_bulk, tip_mask = setup_crack(logger=logger)
ase.io.write('notch.xyz', a, format='extxyz')   

# Global parameters
basename = parameter('basename', 'quasistatic_crack')
calc = parameter('calc')
fmax = parameter('fmax', 0.01)

# Determine simulation control
k1_list = parameter('k1')
old_k1 = k1_list[0]
nsteps = len(k1_list)
tip_dx_list = parameter('tip_dx', np.zeros(nsteps))
tip_dy_list = parameter('tip_dy', np.zeros(nsteps))

# Run crack calculation.
tip_x = tip_x0
tip_y = tip_y0
a.set_calculator(calc)
for i, ( k1, tip_dx, tip_dy ) in enumerate(zip(k1_list, tip_dx_list,
                                               tip_dy_list)):
    logger.pr('=== k1 = {0}*k1g, tip_dx = {1}, tip_dy = {2} ===' \
              .format(k1, tip_dx, tip_dy))
    if tip_dx is None or tip_dy is None:
        #
        # Optimize crack tip position
        #
        old_y = tip_y+1.0
        old_x = tip_x+1.0
        while abs(tip_x-old_x) > 1e-6 and abs(tip_y-old_y) > 1e-6:
            b = cryst.copy()
            ux, uy = crk.displacements(cryst.positions[:,0], cryst.positions[:,1],
                                       tip_x, tip_y, k*k1g)
            b.positions[:,0] += ux
            b.positions[:,1] += uy
            a.set_constraint(None)
            a.positions[boundary_mask] = b.positions[boundary_mask]
            a.set_constraint(ase.constraints.FixAtoms(mask=boundary_mask))
            logger.pr('Optimizing positions...')
            opt = ase.optimize.FIRE(a, logfile=None)
            opt.run(fmax=fmax)
            logger.pr('...done. Converged within {0} steps.' \
                      .format(opt.get_number_of_steps()))
        
            old_x = tip_x
            old_y = tip_y
            tip_x, tip_y = crk.crack_tip_position(a.positions[:,0],
                                                  a.positions[:,1],
                                                  cryst.positions[:,0],
                                                  cryst.positions[:,1],
                                                  tip_x, tip_y, k*k1g,
                                                  mask=mask)
    else:
        #
        # Do not optimize tip position.
        #
        tip_x = tip_x0+tip_dx
        tip_y = tip_y0+tip_dy
        logger.pr('Setting crack tip position to {0} {1}' \
                  .format(tip_x, tip_y))
        # Scale strain field and optimize crack
        a.set_constraint(None)
        x, y = crk.scale_displacements(a.positions[:len(cryst),0],
                                       a.positions[:len(cryst),1],
                                       cryst.positions[:,0],
                                       cryst.positions[:,1],
                                       old_k1, k1)
        a.positions[:len(cryst),0] = x
        a.positions[:len(cryst),1] = y
        # Optimize atoms in center
        a.set_constraint(ase.constraints.FixAtoms(mask=boundary_mask))
        logger.pr('Optimizing positions...')
        opt = ase.optimize.FIRE(a)
        opt.run(fmax=fmax)
        logger.pr('...done. Converged within {0} steps.' \
                  .format(opt.get_number_of_steps()))
        old_k1 = k1

    # Output optimized configuration ot file. Include the mask array in
    # output, so we know which atoms were used in fitting.
    a.set_array('atoms_used_for_fitting_crack_tip', tip_mask)

    # Output a file that contains the target crack tip (used for
    # the displacmenets of the boundary atoms) and the fitted crack tip
    # positions. The target crack tip is marked by a Hydrogen atom.
    b = a.copy()
    b += ase.Atom(ACTUAL_CRACK_TIP, (tip_x, tip_y, b.cell[2, 2]/2))

    # Measure the true (fitted) crack tip position.
    try:
        measured_tip_x, measured_tip_y = \
            crk.crack_tip_position(a.positions[:,0], a.positions[:,1],
                                   cryst.positions[:,0], cryst.positions[:,1],
                                   tip_x, tip_y, k1*k1g, mask=tip_mask)
        measured_tip_x %= a.cell[0][0]
        measured_tip_y %= a.cell[0][0]
    except:
        measured_tip_x = 0.0
        measured_tip_y = 0.0

    # The fitted crack tip is marked by a Helium atom.
    b += ase.Atom(FITTED_CRACK_TIP, (measured_tip_x, measured_tip_y,
                                     b.cell[2, 2]/2))

    b.info['bond_length'] = a.get_distance(bond1, bond2)
    b.info['energy'] = a.get_potential_energy()
    b.info['cell_origin'] = [0, 0, 0]
    ase.io.write('%s_%4.4i.xyz' % (basename, i), b, format='extxyz')