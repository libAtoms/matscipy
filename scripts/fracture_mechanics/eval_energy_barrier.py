#
# Copyright 2014-2016, 2021 Lars Pastewka (U. Freiburg)
#           2014-2017 James Kermode (Warwick U.)
#           2017 Punit Patel (Warwick U.)
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
from ase.data import atomic_numbers

from matscipy.atomic_strain import atomic_strain
from matscipy.elasticity import invariants, full_3x3_to_Voigt_6_strain, \
    cubic_to_Voigt_6x6, Voigt_6_to_full_3x3_stress, Voigt_6_to_full_3x3_strain,\
    rotate_cubic_elastic_constants
from matscipy.fracture_mechanics.energy_release import J_integral
from matscipy.io import savetbl

#from atomistica.analysis import voropp

###

sys.path += ['.', '..']
import params

###

J_m2 = units.kJ/1000 / 1e20

###

# Atom types used for outputting the crack tip position.
ACTUAL_CRACK_TIP = 'Au'
FITTED_CRACK_TIP = 'Ag'

###

compute_J_integral = len(sys.argv) == 3 and sys.argv[2] == '-J'

# Elastic constants
#C6 = cubic_to_Voigt_6x6(params.C11, params.C12, params.C44) * units.GPa

crack_surface = params.crack_surface
crack_front = params.crack_front

third_dir = np.cross(crack_surface, crack_front)
third_dir = np.array(third_dir) / np.sqrt(np.dot(third_dir,
                                                 third_dir))
crack_surface = np.array(crack_surface) / \
    np.sqrt(np.dot(crack_surface, crack_surface))
crack_front = np.array(crack_front) / \
    np.sqrt(np.dot(crack_front, crack_front))

A = np.array([third_dir, crack_surface, crack_front])
if np.linalg.det(A) < 0:
    third_dir = -third_dir
A = np.array([third_dir, crack_surface, crack_front])

#C6 = rotate_cubic_elastic_constants(params.C11, params.C12, params.C44, A) * units.GPa

###

#ref = ase.io.read('cluster.xyz')
ref = params.cryst.copy()
ref_sx, ref_sy, ref_sz = ref.cell.diagonal()

ref.set_pbc(True)
if compute_J_integral:
    ref.set_calculator(params.calc)
    epotref_per_at = ref.get_potential_energies()

###

try:
    prefix = sys.argv[1]
except:
    prefix = 'energy_barrier'

fns = sorted(glob.glob('%s_????.xyz' % prefix))
if len(fns) == 0:
    raise RuntimeError("Could not find files with prefix '{}'.".format(prefix))

tip_x = []
tip_y = []
epot_cluster = []
bond_length = []
bond_force = []
work = []
J_int = []

last_a = None
for fn in fns:
    a = ase.io.read(fn)
    print(fn, a.arrays.keys())

    _tip_x, _tip_y, _tip_z = a.info['fitted_crack_tip']
    tip_x += [ _tip_x ]
    tip_y += [ _tip_y ]

    # Bond length.
    bond1 = a.info['bond1']
    bond2 = a.info['bond2']
    dr = a[bond1].position - a[bond2].position
    bond_length += [ np.linalg.norm(dr) ]
    assert abs(bond_length[-1]-a.info['bond_length']) < 1e-6

    # Groups
    g = a.get_array('groups')

    # Get stored potential energy.
    epot_cluster += [ a.get_potential_energy() ]

    # Stored Forces on bond.
    #a.set_calculator(params.calc)
    forces = a.get_forces()
    df = forces[bond1, :] - forces[bond2, :]
    bond_force += [ 0.5 * np.dot(df, dr)/np.sqrt(np.dot(dr, dr)) ]
    #bond_force = a.info['force']
    #print bond_force[-1], a.info['force']
    assert abs(bond_force[-1]-a.info['force']) < 1e-2

    # Work due to moving boundary.
    if last_a is None:
        work += [ 0.0 ]
    else:
        #last_forces = last_a.get_array('forces')
        last_forces = last_a.get_forces() #(apply_constraint=True)
        # This is the trapezoidal rule.
        #print('MAX force', np.abs(forces[g==0,:]).max())
        #print('MAX last force', np.abs(last_forces[g==0,:]).max())
        #print('MAX d force', np.abs(forces[g==0,:] + last_forces[g==0,:]).max())
        work += [ np.sum(0.5 * (forces[g==0,:]+last_forces[g==0,:]) *
                         (a.positions[g==0,:]-last_a.positions[g==0,:])
                          ) ]

    # J-integral
    b = a.copy()
    mask = np.logical_or(b.numbers == atomic_numbers[ACTUAL_CRACK_TIP],
                         b.numbers == atomic_numbers[FITTED_CRACK_TIP])
    del b[mask]
    G = 0.0
    if compute_J_integral:
        b.set_calculator(params.calc)
        epot_per_at = b.get_potential_energies()

        assert abs(epot_cluster[-1]-b.get_potential_energy()) < 1e-6
        assert np.all(np.abs(forces[np.logical_not(mask)]-b.get_forces()) < 1e-6)

        #virial = params.calc.wpot_per_at
        deformation_gradient, residual = atomic_strain(b, ref, cutoff=2.85)
        virial = b.get_stresses()
        strain = full_3x3_to_Voigt_6_strain(deformation_gradient)
        #virial2 = strain.dot(C6)*vol0

        #vols = voropp(b)
        #virial3 = strain.dot(C6)*vols.reshape(-1,1)

        #print virial[175]
        #print virial2[175]
        #print virial3[175]

        virial = Voigt_6_to_full_3x3_stress(virial)
        #vol, dev, J3 = invariants(strain)
        #b.set_array('vol', vol)
        #b.set_array('dev', dev)
        x, y, z = b.positions.T
        r = np.sqrt((x-_tip_x)**2 + (y-_tip_y)**2)
        #b.set_array('J_eval', np.logical_and(r > params.eval_r1,
        #                                     r < params.eval_r2))

        #epot = b.get_potential_energies()
        G = J_integral(b, deformation_gradient, virial, epot_per_at, epotref_per_at,
                       _tip_x, _tip_y, (ref_sx+ref_sy)/8, 3*(ref_sx+ref_sy)/8)
    J_int += [ G ]

    #b.set_array('J_integral', np.logical_and(r > params.eval_r1,
    #                                         r < params.eval_r2))
    #ase.io.write('eval_'+fn, b, format='extxyz')

    last_a = a

epot_cluster = np.array(epot_cluster)-epot_cluster[0]
work = np.cumsum(work)

tip_x = np.array(tip_x)
tip_y = np.array(tip_y)
bond_length = np.array(bond_length)
print 'tip_x =', tip_x

# Integrate true potential energy.
epot = -cumtrapz(bond_force, bond_length, initial=0.0)

print 'epot =', epot

print 'epot_cluster + work - epot', epot_cluster + work - epot

savetbl('{}_eval.out'.format(prefix),
        bond_length=bond_length,
        bond_force=bond_force,
        epot=epot,
        epot_cluster=epot_cluster,
        work=work,
        tip_x=tip_x,
        tip_y=tip_y,
        J_int=J_int)

# Fit and subtract first energy minimum
i = 1
while epot[i] < epot[i-1]:
    i += 1
print 'i =', i
c, b, a = np.polyfit(bond_length[i-2:i+2]-bond_length[i-1], epot[i-2:i+2], 2)
print 'a, b, c =', a, b, c
min_bond_length = -b/(2*c)
print 'min_bond_length =', min_bond_length+bond_length[i-1], ', delta(min_bond_length) =', min_bond_length
min_epot = a+b*min_bond_length+c*min_bond_length**2
print 'min_epot =', min_epot, ', epot[i-1] =', epot[i-1]
c, b, a = np.polyfit(bond_length[i-2:i+2]-bond_length[i-1], tip_x[i-2:i+2], 2)
min_tip_x = a+b*min_bond_length+c*min_bond_length**2
c, b, a = np.polyfit(bond_length[i-2:i+2]-bond_length[i-1], tip_y[i-2:i+2], 2)
min_tip_y = a+b*min_bond_length+c*min_bond_length**2

min_bond_length += bond_length[i-1]

epot -= min_epot
tip_x = tip_x-min_tip_x
tip_y = tip_y-min_tip_y

savetbl('{}_eval.out'.format(prefix),
        bond_length=bond_length,
        bond_force=bond_force,
        epot=epot,
        epot_cluster=epot_cluster,
        work=work,
        tip_x=tip_x,
        tip_y=tip_y,
        J_int=J_int)

# Fit and subtract second energy minimum
i = len(epot)-1
while epot[i] > epot[i-1]:
    i -= 1
print 'i =', i
c, b, a = np.polyfit(bond_length[i-2:i+3]-bond_length[i], epot[i-2:i+3], 2)
min_bond_length2 = -b/(2*c)
print 'min_bond_length2 =', min_bond_length2+bond_length[i], ', delta(min_bond_length2) =', min_bond_length2
min_epot2 = a+b*min_bond_length2+c*min_bond_length2**2
print 'min_epot2 =', min_epot2, ', epot[i] =', epot[i]

min_bond_length2 += bond_length[i]

corrected_epot = epot - min_epot2*(bond_length-min_bond_length)/(min_bond_length2-min_bond_length)

savetbl('{}_eval.out'.format(prefix),
        bond_length=bond_length,
        bond_force=bond_force,
        epot=epot,
        corrected_epot=corrected_epot,
        epot_cluster=epot_cluster,
        work=work,
        tip_x=tip_x,
        tip_y=tip_y,
        J_int=J_int)
