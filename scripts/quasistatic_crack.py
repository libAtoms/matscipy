#! /usr/bin/env python

# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) Lars Pastewka, Karlsruhe Institute of Technology
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

import os
import sys

import numpy as np

import ase
import ase.io
import ase.constraints
import ase.optimize
from ase.parallel import parprint

import matscipy.fracture_mechanics.crack as crack

###

sys.path += [ "." ]
import params

###

crack = crack.CubicCrystalCrack(params.C11, params.C12, params.C44,
                                params.crack_surface, params.crack_front)

# Get Griffith's k1.
k1g = crack.k1g(params.surface_energy)
parprint('Griffith k1 = %f' % k1g)

# Crack tip position.
r0 = params.r0.copy()
print r0
cryst = params.cryst.copy()
a = cryst.copy()
old_k1 = params.k1[0]
a.positions += crack.displacements(cryst.positions, r0, old_k1*k1g)

oldr = a[0].position.copy()
a.center(vacuum=params.vacuum, axis=(0, 2))
r0 += a[0].position - oldr
cryst.set_cell(a.cell)
cryst.translate(a[0].position - oldr)

cell = a.cell

tip_x0, tip_y0, tip_z0 = r0
info = [(0.0, tip_x0, tip_z0, tip_x0, tip_z0, 0.0)]

b = a.copy()
b += ase.Atom('H', ( tip_x0, tip_y0, tip_z0 ))

parprint('Cell size = %f %f %f' % tuple(a.cell.diagonal()))

if os.path.exists('step_00.cfg'):
    a = ase.io.read('step_00.cfg')
    a.set_pbc([False, True, False])

    assert np.all(a.get_cell() - cell < 1e-6)

g = a.get_array('groups')

ase.io.write('crack_initial.cfg', a)

# Assign calculator.
a.set_calculator(params.calc)

# Determine simulation control
nsteps = None
if hasattr(params, 'k1'):
    nsteps = len(params.k1)
elif hasattr(params, 'tip_dx'):
    nsteps = len(params.tip_dx)
elif hasattr(params, 'tip_dz'):
    nsteps = len(params.tip_dz)

if hasattr(params, 'k1'):
    k1_list = params.k1
else:
    k1_list = [None]*nsteps
if hasattr(params, 'tip_dx'):
    tip_dx_list = params.tip_dx
else:
    tip_dx_list = [None]*nsteps
if hasattr(params, 'tip_dz'):
    tip_dz_list = params.tip_dz
else:
    tip_dz_list = [None]*nsteps

# Run crack calculation.
tip_x = tip_x0
tip_z = tip_z0
for i, ( k1, tip_dx, tip_dz ) in enumerate(zip(k1_list, tip_dx_list,
                                               tip_dz_list)):
    parprint('=== k1 = {0}*k1g, tip_dx = {1}, tip_dz = {2} ===' \
             .format(k1, tip_dx, tip_dz))
    if os.path.exists('step_%2.2i.cfg' % i):
        parprint('step_%2.2i.cfg found, skipping' % i)
        a = ase.io.read('step_%2.2i.cfg' % i)
        a.set_calculator(params.calc)
    else:
        ase.io.write('init_%2.2i.cfg' % i, a)

        mask = g!=0

        if tip_dz is None:
            if tip_dx is None:
                # Optimize x and z position of crack tip
                old_x = tip_x+1.0
                old_z = tip_z+1.0
                while abs(tip_x-old_x) > 1e-6 and abs(tip_z-old_z) > 1e-6:
                    b = cryst.copy()
                    r0 = np.array([tip_x, 0.0, tip_z])
                    b.positions += crack.displacements(cryst.positions, r0,
                                                       k*k1g)

                    a.set_constraint(None)
                    a.positions[g==0] = b.positions[g==0]
                    a.set_constraint(ase.constraints.FixAtoms(mask=g==0))
                    parprint('Optimizing positions...')
                    ase.optimize.FIRE(a, logfile=None).run(fmax=params.fmax)
                    parprint('...done. Converged within {0} steps.' \
                             .format(opt.get_number_of_steps()))
            
                    old_x = tip_x
                    old_z = tip_z
                    tip_x, tip_z = crack.crack_tip_position(a.positions,
                                                            cryst.positions,
                                                            r0, k*k1g,
                                                            mask=mask)
            else:
                # Optimize z position of crack tip
                x = tip_x0+tip_dx
                old_z = tip_z+1.0
                while abs(tip_z-old_z) > 1e-6:
                    b = cryst.copy()
                    r0 = np.array([tip_x, 0.0, tip_z])
                    b.positions += crack.displacements(cryst.positions, r0,
                                                       k1*k1g)

                    a.set_constraint(None)
                    a.positions[g==0] = b.positions[g==0]
                    a.set_constraint(ase.constraints.FixAtoms(mask=g==0))
                    parprint('Optimizing positions...')
                    ase.optimize.FIRE(a, logfile=None).run(fmax=params.fmax)
                    parprint('...done. Converged within {0} steps.' \
                             .format(opt.get_number_of_steps()))
        
                    #x, y, z = a.positions.T
                    #abs_dr = np.sqrt((x-tip_x)**2+(z-tip_z)**2)
                    old_z = tip_z
                    #mask = np.logical_and(g!=0, abs_dr>15.0)
                    tip_z = crack.crack_tip_position_z(a.positions,
                                                       cryst.positions, r0,
                                                       k1*k1g, mask=mask)

            parprint('Optimized crack tip position to %f %f' % (tip_x, tip_z))
        else:
            tip_x = tip_x0+tip_dx
            tip_z = tip_z0+tip_dz

            r0 = np.array([tip_x, 0.0, tip_z])

            parprint('Setting crack tip position to %f %f' % (tip_x, tip_z))

            # Scale strain field and optimize crack
            a.set_constraint(None)
            a.set_positions(crack.scale_displacements(a.positions,
                                                      cryst.positions,
                                                      old_k1, k1))
            a.set_constraint(ase.constraints.FixAtoms(mask=g==0))
            parprint('Optimizing positions...')
            opt = ase.optimize.FIRE(a, logfile=None)
            opt.run(fmax=params.fmax)
            parprint('...done. Converged within {0} steps.' \
                     .format(opt.get_number_of_steps()))

            old_k1 = k1

        # Output the mask array, so we know which atoms were used in fitting.
        a.set_array('atoms_used_for_fitting_crack_tip', mask)
        ase.io.write('step_%2.2i.cfg' % i, a)

        # The target crack tip is marked by a Hydrogen atom.
        b = a.copy()
        b += ase.Atom('H', ( tip_x, tip_y0, tip_z ))

        x0crack, z0crack = crack.crack_tip_position(a.positions,
                                                    cryst.positions,
                                                    r0, k1*k1g, mask=mask)

        parprint('Measured crack tip at %f %f' % (x0crack, z0crack))

        # The fitted crack tip is marked by a Helium atom.
        b += ase.Atom('He', ( x0crack, tip_y0, z0crack ))
        ase.io.write('step_with_crack_tip_%2.2i.cfg' % i, b)

        info += [ ( k1, tip_x, tip_z, x0crack, z0crack,
                    a.get_potential_energy() ) ]

# Output some aggregate data.
np.savetxt('crack.out', info)

