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

import atomistica

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
cryst = params.cryst.copy()
a = cryst.copy()
a.positions += crack.displacements(cryst.positions, r0, params.k1*k1g)

oldr = a[0].position.copy()
a.center(vacuum=params.vacuum)
r0 += a[0].position - oldr
cryst.set_cell(a.cell)
cryst.translate(a[0].position - oldr)

cell = a.cell

x0, y0, z0 = r0
info = [(0.0, x0, z0, 0.0)]
x0beg = x0
x0end = x0+params.dx0

b = a.copy()
b += ase.Atom('H', ( x0, y0, z0 ))
ase.io.write('step_with_crack_tip_00.cfg', b)

parprint('Cell size = %f %f %f' % tuple(a.cell.diagonal()))

if os.path.exists('step_00.cfg'):
    a = ase.io.read('step_00.cfg')
    a.set_pbc([False, True, False])

    assert np.all(a.get_cell() - cell < 1e-6)

g = a.get_array('groups')

ase.io.write('crack_initial.cfg', a)

# Assign calculator.
a.set_calculator(params.calc)

# Run crack calculation.
parprint('k = %f*k1g' % params.k1)
for i, x0 in enumerate(np.linspace(x0beg, x0end, params.nsteps)):
    parprint('Moved crack tip position to %f %f' % (x0, z0))
    if os.path.exists('step_%2.2i.cfg' % i):
        parprint('step_%2.2i.cfg found, skipping' % i)
        a = ase.io.read('step_%2.2i.cfg' % i)
        a.set_calculator(params.calc)
    else:
        ase.io.write('init_%2.2i.cfg' % i, a)
        ase.optimize.FIRE(a, logfile=None).run(fmax=params.fmax)

        old_z0 = z0+1.0
        while abs(z0-old_z0) > 1e-6:
            b = cryst.copy()
            r0 = np.array([x0, 0.0, z0])
            b.positions += crack.displacements(cryst.positions, r0,
                                               params.k1*k1g)

            a.set_constraint(None)
            a.positions[g==0] = b.positions[g==0]
            a.set_constraint(ase.constraints.FixAtoms(mask=g==0))
            ase.optimize.FIRE(a, logfile=None).run(fmax=params.fmax)
        
            x, y, z = a.positions.T
            abs_dr = np.sqrt((x-x0)**2+(z-z0)**2)
            old_z0 = z0
            #mask = np.logical_and(g!=0, abs_dr>15.0)
            mask = g!=0
            z0 = crack.crack_tip_position_z(a.positions, cryst.positions, r0,
                                            params.k1*k1g,
                                            mask=mask)

        parprint('Optimized crack tip position to %f %f' % (x0, z0))

        # Output the mask array, so we know which atoms were used in fitting.
        a.set_array('atoms_used_for_fitting_crack_tip', mask)
        ase.io.write('step_%2.2i.cfg' % i, a)

        # The target crack tip is marked by a Hydrogen atom.
        b = a.copy()
        b += ase.Atom('H', ( x0, y0, z0 ))

        x0crack, z0crack = crack.crack_tip_position(a.positions,
                                                    cryst.positions,
                                                    r0,
                                                    params.k1*k1g,
                                                    mask=mask)

        parprint('Detected crack tip at %f %f' % (x0crack, z0crack))

        # The fitted crack tip is marked by a Helium atom.
        b += ase.Atom('He', ( x0crack, y0, z0crack ))
        ase.io.write('step_with_crack_tip_%2.2i.cfg' % (i+1), b)

        info += [ ( params.k1, x0, z0, a.get_potential_energy() ) ]

# Output some aggregate data.
np.savetxt('crack.out', info)

