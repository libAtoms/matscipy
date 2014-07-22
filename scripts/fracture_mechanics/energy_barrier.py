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

cryst = params.cryst.copy()
crack = crack.CubicCrystalCrack(params.C11, params.C12, params.C44,
                                params.crack_surface, params.crack_front)

# Get Griffith's k1.
k1g = crack.k1g(params.surface_energy)
parprint('Griffith k1 = %f' % k1g)

# Crack tip position.
if hasattr(params, 'tip_x'):
    tip_x = params.tip_x
else:
    tip_x = cryst.cell.diagonal()[0]/2
if hasattr(params, 'tip_z'):
    tip_z = params.tip_z
else:
    tip_z = cryst.cell.diagonal()[2]/2

a = cryst.copy()
a.positions += crack.displacements(cryst.positions,
                                   np.array([tip_x, 0.0, tip_z]),
                                   params.k1*k1g)

oldr = a[0].position.copy()
a.center(vacuum=params.vacuum, axis=(0, 2))
tip_x += a[0].position[0] - oldr[0]
tip_z += a[0].position[2] - oldr[2]
cryst.set_cell(a.cell)
cryst.translate(a[0].position - oldr)

cell = a.cell

parprint('Cell size = %f %f %f' % tuple(a.cell.diagonal()))

if os.path.exists('step_00.cfg'):
    a = ase.io.read('step_00.cfg')
    a.set_pbc([False, True, False])

    assert np.all(a.get_cell() - cell < 1e-6)

g = a.get_array('groups')

ase.io.write('crack_initial.cfg', a)

# Simulation control
bond1, bond2 = params.bond

# Assign calculator.
a.set_calculator(params.calc)

info = []

# Run crack calculation.
for i, bond_length in enumerate(params.bond_lengths):
    parprint('=== bond_length = {0} ==='.format(bond_length))
    if os.path.exists('step_%2.2i.cfg' % i):
        parprint('step_%2.2i.cfg found, skipping' % i)
        a = ase.io.read('step_%2.2i.cfg' % i)
        a.set_calculator(params.calc)
    else:
        a.set_constraint(None)
        a.set_distance(bond1, bond2, bond_length)
        bond_length_constraint = ase.constraints.FixBondLength(bond1, bond2)

        ase.io.write('init_%2.2i.cfg' % i, a)

        mask = g!=0

        # Optimize x and z position of crack tip
        if hasattr(params, 'optimize_tip_position') and \
               params.optimize_tip_position:
            old_x = tip_x+1.0
            old_z = tip_z+1.0
            while abs(tip_x-old_x) > 1e-6 and abs(tip_z-old_z) > 1e-6:
                b = cryst.copy()
                r0 = np.array([tip_x, 0.0, tip_z])
                b.positions += crack.displacements(cryst.positions, r0,
                                                   params.k1*k1g)

                a.set_constraint(None)
                a.positions[g==0] = b.positions[g==0]
                a.set_constraint([ase.constraints.FixAtoms(mask=g==0),
                                  bond_length_constraint])
                parprint('Optimizing positions...')
                opt = ase.optimize.FIRE(a, logfile=None)
                opt.run(fmax=params.fmax)
                parprint('...done. Converged within {0} steps.' \
                         .format(opt.get_number_of_steps()))
            
                old_x = tip_x
                old_z = tip_z
                r0 = np.array([tip_x, 0.0, tip_z])
                tip_x, tip_z = crack.crack_tip_position(a.positions,
                                                        cryst.positions,
                                                        r0, params.k1*k1g,
                                                        mask=mask)
        else:
            a.set_constraint([ase.constraints.FixAtoms(mask=g==0),
                              bond_length_constraint])
            parprint('Optimizing positions...')
            opt = ase.optimize.FIRE(a, logfile=None)
            opt.run(fmax=params.fmax)
            parprint('...done. Converged within {0} steps.' \
                     .format(opt.get_number_of_steps()))

        # Output the mask array, so we know which atoms were used in fitting.
        a.set_array('atoms_used_for_fitting_crack_tip', mask)
        ase.io.write('step_%2.2i.cfg' % i, a)

        # The target crack tip is marked by a Hydrogen atom.
        b = a.copy()
        b += ase.Atom('H', (tip_x, b.cell.diagonal()[1]/2, tip_z))

        r0 = np.array([tip_x, 0.0, tip_z])
        x0crack, z0crack = crack.crack_tip_position(a.positions,
                                                    cryst.positions,
                                                    r0, params.k1*k1g,
                                                    mask=mask)

        parprint('Measured crack tip at %f %f' % (x0crack, z0crack))

        # The fitted crack tip is marked by a Helium atom.
        b += ase.Atom('He', (x0crack, b.cell.diagonal()[1]/2, z0crack))
        ase.io.write('step_with_crack_tip_%2.2i.cfg' % i, b)

        bond_dir = a[bond1].position - a[bond2].position
        bond_dir /= np.linalg.norm(bond_dir)
        force = np.dot(bond_length_constraint.get_constraint_force(), bond_dir)

        info += [ ( bond_length, force, a.get_potential_energy(), tip_x, tip_z,
                    x0crack, z0crack ) ]

print info

# Output some aggregate data.
np.savetxt('crack.out', info)
