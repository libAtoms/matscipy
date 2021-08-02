#
# Copyright 2014, 2021 Lars Pastewka (U. Freiburg)
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

import glob
import sys

import numpy as np
from scipy.integrate import cumtrapz

import ase.io
import ase.units as units
import ase.optimize
from ase.data import atomic_numbers
from ase.parallel import parprint

from matscipy.atomic_strain import atomic_strain
from matscipy.neighbours import neighbour_list
from matscipy.elasticity import invariants, full_3x3_to_Voigt_6_strain, \
    cubic_to_Voigt_6x6, Voigt_6_to_full_3x3_stress, Voigt_6_to_full_3x3_strain,\
    rotate_cubic_elastic_constants
from matscipy.fracture_mechanics.energy_release import J_integral

###

sys.path += [ "." ]
import params

###

J_m2 = units.kJ/1000 / 1e20

###

# Atom types used for outputting the crack tip position.
ACTUAL_CRACK_TIP = 'Au'
FITTED_CRACK_TIP = 'Ag'

###

# Get cohesive energy
a = params.cryst.info['unitcell'].copy()
a.set_calculator(params.calc)
e0 = a.get_potential_energy()/len(a)
print 'cohesive energy = {} eV/atom'.format(e0)

# Get surface energy and stress
a *= (1,3,1)
a.set_pbc([True, False, True])
ase.optimize.FIRE(a, logfile=None).run(fmax=params.fmax)
coord = np.bincount(neighbour_list("i", a, 2.85))
esurf0 = (a.get_potential_energy() - e0*len(a))
sx, sy, sz = a.cell.diagonal()
print 'surface energy = {} J/m^2'.format(esurf0/(2*sx*sz)/J_m2)
esurf0 = a.get_potential_energies()
esurf1 = esurf0[1]
esurf0 = esurf0[0]
print 'cohesive energy (surface atoms) = {}, {} eV/atom'.format(esurf0, esurf1)
vsurf0 = a.get_stresses()
vsurf1 = Voigt_6_to_full_3x3_stress(vsurf0[1])
vsurf0 = Voigt_6_to_full_3x3_stress(vsurf0[0])

# Reference configuration for strain calculation
a = ase.io.read(sys.argv[1])
ref = ase.io.read(sys.argv[2])

###

# Get crack tip position
tip_x, tip_y, tip_z = a.info['fitted_crack_tip']

# Set calculator
a.set_calculator(params.calc)

# Relax positions
del a[np.logical_or(a.numbers == atomic_numbers[ACTUAL_CRACK_TIP],
                    a.numbers == atomic_numbers[FITTED_CRACK_TIP])]
g = a.get_array('groups')
a.set_constraint(ase.constraints.FixAtoms(mask=g==0))

a.set_calculator(params.calc)

parprint('Optimizing positions...')
opt = ase.optimize.FIRE(a, logfile=None)
opt.run(fmax=params.fmax)
parprint('...done. Converged within {0} steps.' \
             .format(opt.get_number_of_steps()))

# Get atomic strain
i, j = neighbour_list("ij", a, cutoff=2.85)
deformation_gradient, residual = atomic_strain(a, ref, neighbours=(i, j))

# Get atomic stresses
virial = a.get_stresses() # Note: get_stresses returns the virial in Atomistica!
strain = full_3x3_to_Voigt_6_strain(deformation_gradient)
vol_strain, dev_strain, J3_strain = invariants(strain)

a.set_array('strain', strain)
a.set_array('vol_strain', vol_strain)
a.set_array('dev_strain', dev_strain)
a.set_array('virial', virial)

virial = Voigt_6_to_full_3x3_stress(virial)

# Coordination count
coord = np.bincount(i)
coord_coord = np.bincount(i, weights=coord[j])
mask = coord!=4
mask = np.ones_like(mask)

eref = e0*np.ones(len(a))
eref[coord_coord==15] = esurf1
eref[coord!=4] = esurf0

#virial[coord_coord==15] -= vsurf1
#virial[coord!=4] -= vsurf0

a.set_array('coordination', coord)
a.set_array('coord_coord', coord_coord)
a.set_array('eref', eref)
ase.io.write('eval_J_integral.xyz', a, format='extxyz')

# Evaluate J-integral
epot = a.get_potential_energies()
for r1, r2 in zip(params.eval_r1, params.eval_r2):
    G = J_integral(a, deformation_gradient, virial, epot, eref, tip_x, tip_y,
                   r1, r2, mask)
    print '[From {0} A to {1} A]: J = {2} J/m^2'.format(r1, r2, G/J_m2)
