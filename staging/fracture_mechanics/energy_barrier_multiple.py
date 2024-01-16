#
# Copyright 2014, 2021 Lars Pastewka (U. Freiburg)
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
crk = crack.CubicCrystalCrack(params.C11, params.C12, params.C44,
                              params.crack_surface, params.crack_front)

# Get Griffith's k1.
k1g = crk.k1g(params.surface_energy)
parprint('Griffith k1 = %f' % k1g)

# Crack tip position.
if hasattr(params, 'tip_x'):
    tip_x = params.tip_x
else:
    tip_x = cryst.cell.diagonal()[0]/2
if hasattr(params, 'tip_y'):
    tip_y = params.tip_y
else:
    tip_y = cryst.cell.diagonal()[1]/2

# Apply initial strain field.
ase.io.write('cluster.xyz', cryst, format='extxyz')
a = cryst.copy()
ux, uy = crk.displacements(cryst.positions[:,0], cryst.positions[:,1],
                           tip_x, tip_y, params.k1*k1g)
a.positions[:,0] += ux
a.positions[:,1] += uy

# Center notched configuration in simulation cell and ensure enough vacuum.
oldr = a[0].position.copy()
a.center(vacuum=params.vacuum, axis=0)
a.center(vacuum=params.vacuum, axis=1)
tip_x += a[0].position[0] - oldr[0]
tip_y += a[0].position[1] - oldr[1]
cryst.set_cell(a.cell)
cryst.translate(a[0].position - oldr)
ase.io.write('notch.xyz', a, format='extxyz')

parprint('Initial tip position {0} {1}'.format(tip_x, tip_y))

# Groups mark the fixed region and the region use for fitting the crack tip.
g = a.get_array('groups')

# Assign calculator.
a.set_calculator(params.calc)

for j in range(params.n_bonds):

    bond1, bond2 = crack.find_tip_coordination(a)
    print 'Breaking tip bond {0}--{1}'.format(bond1, bond2)

    # Run crack calculation.
    for i, bond_length in enumerate(params.bond_lengths):
        parprint('=== bond_length = {0} ==='.format(bond_length))
        xyz_file = 'bond_%d_%4d.xyz' % (j+1, int(bond_length*1000))
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
                old_y = tip_y+1.0
                while abs(tip_x-old_x) > 1e-6 and abs(tip_y-old_y) > 1e-6:
                    b = cryst.copy()
                    ux, uy = crk.displacements(cryst.positions[:,0],
                                                 cryst.positions[:,1],
                                                 tip_x, tip_y, params.k1*k1g)
                    b.positions[:,0] += ux
                    b.positions[:,1] += uy

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
                    old_y = tip_y
                    tip_x, tip_y = \
                        crk.crack_tip_position(a.positions[:,0],
                                                 a.positions[:,1],
                                                 cryst.positions[:,0],
                                                 cryst.positions[:,1],
                                                 tip_x, tip_y, params.k1*k1g,
                                                 mask=mask)
                    parprint('New crack tip at {0} {1}'.format(tip_x, tip_y))
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
            b += ase.Atom(ACTUAL_CRACK_TIP, (tip_x, tip_y, b.cell.diagonal()[2]/2))
            b.info['actual_crack_tip'] = (tip_x, tip_y, b.cell.diagonal()[2]/2)

            fit_x, fit_y = crk.crack_tip_position(a.positions[:,0],
                                                  a.positions[:,1],
                                                  cryst.positions[:,0],
                                                  cryst.positions[:,1],
                                                  tip_x, tip_y, params.k1*k1g,
                                                  mask=mask)

            parprint('Measured crack tip at %f %f' % (fit_x, fit_y))

            # The fitted crack tip is marked by a silver atom.
            b += ase.Atom(FITTED_CRACK_TIP, (fit_x, fit_y, b.cell.diagonal()[2]/2))
            b.info['fitted_crack_tip'] = (fit_x, fit_y, b.cell.diagonal()[2]/2)

            bond_dir = a[bond1].position - a[bond2].position
            bond_dir /= np.linalg.norm(bond_dir)
            force = np.dot(bond_length_constraint.get_constraint_force(), bond_dir)

            b.info['bond1'] = bond1
            b.info['bond2'] = bond2
            b.info['bond_length'] = bond_length
            b.info['force'] = force
            b.info['energy'] = a.get_potential_energy()
            b.info['cell_origin'] = [0, 0, 0]
            ase.io.write(xyz_file, b, format='extxyz')
