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

import os
import sys

import numpy as np

import ase
import ase.constraints
import ase.io
import ase.optimize
from ase.data import atomic_numbers
from ase.parallel import parprint

import matscipy.fracture_mechanics.crack as crack

###

sys.path += [ "." ]
import params

###

# Atom types used for outputting the crack tip position.
ACTUAL_CRACK_TIP = 'Au'
FITTED_CRACK_TIP = 'Ag'

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

# Apply initial strain field.
a = cryst.copy()
a.positions += crack.displacements(cryst.positions,
                                   np.array([tip_x, 0.0, tip_z]),
                                   params.k1*k1g)

# Center notched configuration in simulation cell and ensure enough vacuum.
oldr = a[0].position.copy()
a.center(vacuum=params.vacuum, axis=0)
a.center(vacuum=params.vacuum, axis=2)
tip_x += a[0].position[0] - oldr[0]
tip_z += a[0].position[2] - oldr[2]
cryst.set_cell(a.cell)
cryst.translate(a[0].position - oldr)

# Groups mark the fixed region and the region use for fitting the crack tip.
g = a.get_array('groups')

# Which bond to break.
bond1, bond2 = params.bond

# Assign calculator.
a.set_calculator(params.calc)

info = []

# Run crack calculation.
for i, bond_length in enumerate(params.bond_lengths):
    parprint('=== bond_length = {0} ==='.format(bond_length))
    xyz_file = 'step_%4d.xyz' % int(bond_length*1000)
    if os.path.exists(xyz_file):
        parprint('%s found, skipping' % xyz_file)
        a = ase.io.read(xyz_file)
        del a[np.logical_or(a.numbers == atomic_numbers[ACTUAL_CRACK_TIP],
                            a.numbers == atomic_numbers[FITTED_CRACK_TIP])]
        a.set_calculator(params.calc)
    else:
        a.set_constraint(None)
        a.set_distance(bond1, bond2, bond_length)
        bond_length_constraint = ase.constraints.FixBondLength(bond1, bond2)

        # Atoms to be used for fitting the crack tip position.
        mask = g==1

        # Optimize x and z position of crack tip.
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
                parprint('New crack tip at {0} {1}'.format(tip_x, tip_z))
        else:
            a.set_constraint([ase.constraints.FixAtoms(mask=g==0),
                              bond_length_constraint])
            parprint('Optimizing positions...')
            opt = ase.optimize.FIRE(a, logfile=None)
            opt.run(fmax=params.fmax)
            parprint('...done. Converged within {0} steps.' \
                     .format(opt.get_number_of_steps()))

        # Store forces.
        a.set_constraint(None)
        a.set_array('forces', a.get_forces())

        # The target crack tip is marked by a gold atom.
        b = a.copy()
        b += ase.Atom(ACTUAL_CRACK_TIP, (tip_x, b.cell.diagonal()[1]/2, tip_z))
        b.info['actual_crack_tip'] = (tip_x, b.cell.diagonal()[1]/2, tip_z)

        r0 = np.array([tip_x, 0.0, tip_z])
        fit_x, fit_z = crack.crack_tip_position(a.positions,
                                                cryst.positions,
                                                r0, params.k1*k1g,
                                                mask=mask)

        parprint('Measured crack tip at %f %f' % (fit_x, fit_z))

        # The fitted crack tip is marked by a silver atom.
        b += ase.Atom(FITTED_CRACK_TIP, (fit_x, b.cell.diagonal()[1]/2, fit_z))
        b.info['fitted_crack_tip'] =  (fit_x, b.cell.diagonal()[1]/2, fit_z)

        bond_dir = a[bond1].position - a[bond2].position
        bond_dir /= np.linalg.norm(bond_dir)
        force = np.dot(bond_length_constraint.get_constraint_force(), bond_dir)

        info += [ ( bond_length, force, a.get_potential_energy() ) ]
        b.info['bond_length'] = bond_length
        b.info['force'] = force
        b.info['energy'] = a.get_potential_energy()
        b.info['cell_origin'] = [0, 0, 0]
        ase.io.write(xyz_file, b, format='extxyz')

# Output info data to seperate file.
np.savetxt('crack.out', info)
