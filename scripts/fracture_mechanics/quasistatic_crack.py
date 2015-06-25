#! /usr/bin/env python

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
from matscipy.checkpoint import Checkpoint, NoCheckpoint
from matscipy.elasticity import fit_elastic_constants
from matscipy.logger import Logger, screen

###

sys.path += ['.', '..']
import params

###

def param(s, d):
    global logger
    try:
        val = params.__dict__[s]
        logger.pr('(user value)      {0} = {1}'.format(s, val))
    except KeyError:
        val = d
        logger.pr('(default value)   {0} = {1}'.format(s, val))
    return val

###

logger = screen
out = Logger('bond_length.out')

# Checkpointing
CP = Checkpoint(logger=screen)
fit_elastic_constants = CP(fit_elastic_constants)

###

cryst = params.cryst.copy()

# Double check elastic constants. We're just assuming this is really a periodic
# system. (True if it comes out of the cluster routines.)

compute_elastic_constants = param('compute_elastic_constants', False)

if compute_elastic_constants:
    pbc = cryst.pbc.copy()
    cryst.set_pbc(True)
    cryst.set_calculator(params.calc)
    C, C_err = fit_elastic_constants(cryst, verbose=False,
                                     optimizer=ase.optimize.FIRE)
    cryst.set_pbc(pbc)

    logger.pr('Measured elastic constants (in GPa):')
    logger.pr(np.round(C*10/GPa)/10)

    crk = crack.CubicCrystalCrack(params.crack_surface, params.crack_front,
                                  Crot=C/GPa)
else:
    if hasattr(params, 'C'):
        crk = crack.CubicCrystalCrack(params.crack_surface, params.crack_front,
                                      C=params.C)
    else:    
        crk = crack.CubicCrystalCrack(params.crack_surface, params.crack_front,
                                      params.C11, params.C12, params.C44)


logger.pr('Elastic constants used for boundary condition (in GPa):')
logger.pr(np.round(crk.C*10)/10)

# Get Griffith's k1.
k1g = crk.k1g(params.surface_energy)
logger.pr('Griffith k1 = {0}'.format(k1g))

# Compute crack tip position.
tip_x0 = param('tip_x', cryst.cell.diagonal()[0]/2)
tip_y0 = param('tip_y', cryst.cell.diagonal()[1]/2)

a = cryst.copy()
old_k1 = params.k1[0]
ux, uy = crk.displacements(cryst.positions[:,0], cryst.positions[:,1],
                           tip_x0, tip_y0, old_k1*k1g)
a.positions[:,0] += ux
a.positions[:,1] += uy

oldr = a[0].position.copy()
a.center(vacuum=params.vacuum, axis=0)
a.center(vacuum=params.vacuum, axis=1)
tip_x0 += a[0].position[0] - oldr[0]
tip_y0 += a[0].position[1] - oldr[1]
cryst.set_cell(a.cell)
cryst.translate(a[0].position - oldr)

g = a.get_array('groups')
mask = g!=0

# Choose which bond to break.
bondlength = param('bondlength', 1.85)
bond1, bond2 = param('bond', crack.find_tip_coordination(a, bondlength=bondlength))

logger.pr('Opening bond {0}--{1}, initial bond length {2}' \
          .format(bond1, bond2, a.get_distance(bond1, bond2, mic=True)))

# centre vertically on the opening bond
a.translate([0., a.cell[1,1]/2.0 - 
                (a.positions[bond1, 1] + 
                 a.positions[bond2, 1])/2.0, 0])

tip_x0 = (a.positions[bond1, 0] + a.positions[bond2, 0])/2
tip_y0 = (a.positions[bond1, 1] + a.positions[bond2, 1])/2

b = a.copy()
b += ase.Atom('H', (tip_x0, tip_y0, b.cell[2, 2]/2))
ase.io.write('notch.xyz', b, format='extxyz')

# Assign calculator.
a.set_calculator(params.calc)

# Determine simulation control
k1_list = params.k1
nsteps = len(params.k1)
tip_dx_list = param('tip_dx', np.zeros(nsteps))
tip_dy_list = param('tip_dy', np.zeros(nsteps))

# Run crack calculation.
tip_x = tip_x0
tip_y = tip_y0
for i, ( k1, tip_dx, tip_dy ) in enumerate(zip(k1_list, tip_dx_list,
                                               tip_dy_list)):
    logger.pr('=== k1 = {0}*k1g, tip_dx = {1}, tip_dy = {2} ===' \
              .format(k1, tip_dx, tip_dy))
    try:
        a = CP.load(a)
    except NoCheckpoint:
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
                a.positions[g==0] = b.positions[g==0]
                a.set_constraint(ase.constraints.FixAtoms(mask=g==0))
                logger.pr('Optimizing positions...')
                opt = ase.optimize.FIRE(a, logfile=None)
                opt.run(fmax=params.fmax)
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
            x, y = crk.scale_displacements(a.positions[:,0],
                                           a.positions[:,1],
                                           cryst.positions[:,0],
                                           cryst.positions[:,1],
                                           old_k1, k1)
            a.positions[:,0] = x
            a.positions[:,1] = y

            # Optimize atoms in center
            a.set_constraint(ase.constraints.FixAtoms(mask=g==0))
            logger.pr('Optimizing positions...')
            opt = ase.optimize.FIRE(a)
            opt.run(fmax=params.fmax)
            logger.pr('...done. Converged within {0} steps.' \
                      .format(opt.get_number_of_steps()))

            old_k1 = k1

        #
        # Finalize and output step.
        #

        CP.save(a)

    # Output optimized configuration ot file. Include the mask array in
    # output, so we know which atoms were used in fitting.
    a.set_array('atoms_used_for_fitting_crack_tip', mask)

    # Output a file that contains the target crack tip (used for
    # the displacmenets of the boundary atoms) and the fitted crack tip
    # positions. The target crack tip is marked by a Hydrogen atom.
    b = a.copy()
    b += ase.Atom('H', (tip_x, tip_y, b.cell[2, 2]/2))

    # Measure the true (fitted) crack tip position.
    try:
        measured_tip_x, measured_tip_y = \
            crk.crack_tip_position(a.positions[:,0], a.positions[:,1],
                                   cryst.positions[:,0], cryst.positions[:,1],
                                   tip_x, tip_y, k1*k1g, mask=mask)
    except:
        measured_tip_x = 0.0
        measured_tip_y = 0.0

    # The fitted crack tip is marked by a Helium atom.
    b += ase.Atom('He', (measured_tip_x, measured_tip_y, b.cell[2, 2]/2))
    ase.io.write('step_%2.2i.xyz' % i, b, format='extxyz')

    out.st(['k1', 'bond_length', 'tip_x', 'tip_y'],
           [k1, a.get_distance(bond1, bond2), measured_tip_x,
            measured_tip_y])
