#
# Copyright 2014, 2020 James Kermode (Warwick U.)
#           2016 Punit Patel (Warwick U.)
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

"""
Script to generate a crack slab, and apply initial strain ramp

James Kermode <james.kermode@kcl.ac.uk>
August 2014
"""

import numpy as np

from ase.lattice.cubic import Diamond
from ase.constraints import FixAtoms, StrainFilter
from ase.optimize import FIRE
import ase.io
import ase.units as units

from matscipy.elasticity import (measure_triclinic_elastic_constants,
                                 rotate_elastic_constants,
                                 youngs_modulus, poisson_ratio)

from matscipy.fracture_mechanics.crack import (print_crack_system,
                                               G_to_strain,
                                               thin_strip_displacement_y,
                                               find_tip_stress_field)
import sys
sys.path.insert(0, '.')
import params

a = params.cryst.copy()

# ***** Find eqm. lattice constant ******

# find the equilibrium lattice constant by minimising atoms wrt virial
# tensor given by SW pot (possibly replace this with parabola fit in
# another script and hardcoded a0 here)
a.set_calculator(params.calc)

if hasattr(params, 'relax_bulk') and params.relax_bulk:
    print('Minimising bulk unit cell...')
    opt = FIRE(StrainFilter(a, mask=[1, 1, 1, 0, 0, 0]))
    opt.run(fmax=params.bulk_fmax)

a0 = a.cell[0, 0]
print('Lattice constant %.3f A\n' % a0)

a.set_calculator(params.calc)

# ******* Find elastic constants *******

# Get 6x6 matrix of elastic constants C_ij
C = measure_triclinic_elastic_constants(a, optimizer=FIRE,
                                        fmax=params.bulk_fmax)
print('Elastic constants (GPa):')
print((C / units.GPa).round(0))
print('')

E = youngs_modulus(C, params.cleavage_plane)
print('Young\'s modulus %.1f GPa' % (E / units.GPa))
nu = poisson_ratio(C, params.cleavage_plane, params.crack_direction)
print('Poisson ratio % .3f\n' % nu)

# **** Setup crack slab unit cell ******

directions = [params.crack_direction,
              params.cleavage_plane,
              params.crack_front]
print_crack_system(directions)

# now, we build system aligned with requested crystallographic orientation
unit_slab = Diamond(directions=directions,
                    size=(1, 1, 1),
                    symbol='Si',
                    pbc=True,
                    latticeconstant=a0)


if (hasattr(params, 'check_rotated_elastic_constants') and
    # Check that elastic constants of the rotated system
    # lead to same Young's modulus and Poisson ratio

    params.check_rotated_elastic_constants):
    unit_slab.set_calculator(params.calc)
    C_r1 = measure_triclinic_elastic_constants(unit_slab,
                                               optimizer=FIRE,
                                               fmax=params.bulk_fmax)

    R = np.array([ np.array(x)/np.linalg.norm(x) for x in directions ])
    C_r2 = rotate_elastic_constants(C, R)

    for C_r in [C_r1, C_r2]:
        S_r = np.linalg.inv(C_r)
        E_r = 1./S_r[1,1]
        print('Young\'s modulus from C_r: %.1f GPa' % (E_r / units.GPa))
        assert (abs(E_r - E)/units.GPa < 1e-3)

        nu_r = -S_r[1,2]/S_r[1,1]
        print('Possion ratio from C_r: %.3f' % (nu_r))
        assert (abs(nu_r - nu) < 1e-3)

print('Unit slab with %d atoms per unit cell:' % len(unit_slab))
print(unit_slab.cell)
print('')

# center vertically half way along the vertical bond between atoms 0 and 1
unit_slab.positions[:, 1] += (unit_slab.positions[1, 1] -
                              unit_slab.positions[0, 1]) / 2.0

# map positions back into unit cell
unit_slab.set_scaled_positions(unit_slab.get_scaled_positions())

# Make a surface unit cell by repllcating and adding some vaccum along y
surface = unit_slab * [1, params.surf_ny, 1]
surface.center(params.vacuum, axis=1)


# ********** Surface energy ************

# Calculate surface energy per unit area
surface.set_calculator(params.calc)

if hasattr(params, 'relax_bulk') and params.relax_bulk:
    print('Minimising surface unit cell...')
    opt = FIRE(surface)
    opt.run(fmax=params.bulk_fmax)

E_surf = surface.get_potential_energy()
E_per_atom_bulk = a.get_potential_energy() / len(a)
area = surface.get_volume() / surface.cell[1, 1]
gamma = ((E_surf - E_per_atom_bulk * len(surface)) /
         (2.0 * area))

print('Surface energy of %s surface %.4f J/m^2\n' %
      (params.cleavage_plane, gamma / (units.J / units.m ** 2)))


# ***** Setup crack slab supercell *****

# Now we will build the full crack slab system,
# approximately matching requested width and height
nx = int(params.width / unit_slab.cell[0, 0])
ny = int(params.height / unit_slab.cell[1, 1])

# make sure ny is even so slab is centered on a bond
if ny % 2 == 1:
    ny += 1

# make a supercell of unit_slab
crack_slab = unit_slab * (nx, ny, 1)

# open up the cell along x and y by introducing some vaccum
crack_slab.center(params.vacuum, axis=0)
crack_slab.center(params.vacuum, axis=1)

# centre the slab on the origin
crack_slab.positions[:, 0] -= crack_slab.positions[:, 0].mean()
crack_slab.positions[:, 1] -= crack_slab.positions[:, 1].mean()

orig_width = (crack_slab.positions[:, 0].max() -
              crack_slab.positions[:, 0].min())
orig_height = (crack_slab.positions[:, 1].max() -
               crack_slab.positions[:, 1].min())

print(('Made slab with %d atoms, original width and height: %.1f x %.1f A^2' %
       (len(crack_slab), orig_width, orig_height)))

top = crack_slab.positions[:, 1].max()
bottom = crack_slab.positions[:, 1].min()
left = crack_slab.positions[:, 0].min()
right = crack_slab.positions[:, 0].max()

# fix atoms in the top and bottom rows
fixed_mask = ((abs(crack_slab.positions[:, 1] - top) < 1.0) |
              (abs(crack_slab.positions[:, 1] - bottom) < 1.0))
const = FixAtoms(mask=fixed_mask)
crack_slab.set_constraint(const)
print('Fixed %d atoms\n' % fixed_mask.sum())

# Save all calculated materials properties inside the Atoms object
crack_slab.info['nneightol'] = 1.3 # nearest neighbour tolerance
crack_slab.info['LatticeConstant'] = a0
crack_slab.info['C11'] = C[0, 0]
crack_slab.info['C12'] = C[0, 1]
crack_slab.info['C44'] = C[3, 3]
crack_slab.info['YoungsModulus'] = E
crack_slab.info['PoissonRatio_yx'] = nu
crack_slab.info['SurfaceEnergy'] = gamma
crack_slab.info['OrigWidth'] = orig_width
crack_slab.info['OrigHeight'] = orig_height
crack_slab.info['CrackDirection'] = params.crack_direction
crack_slab.info['CleavagePlane'] = params.cleavage_plane
crack_slab.info['CrackFront'] = params.crack_front
crack_slab.info['cell_origin'] = -np.diag(crack_slab.cell)/2.0

crack_slab.set_array('fixed_mask', fixed_mask)
ase.io.write('slab.xyz', crack_slab, format='extxyz')

# ****** Apply initial strain ramp *****

strain = G_to_strain(params.initial_G, E, nu, orig_height)

crack_slab.positions[:, 1] += thin_strip_displacement_y(
                                 crack_slab.positions[:, 0],
                                 crack_slab.positions[:, 1],
                                 strain,
                                 left + params.crack_seed_length,
                                 left + params.crack_seed_length +
                                        params.strain_ramp_length)

print('Applied initial load: strain=%.4f, G=%.2f J/m^2' %
      (strain, params.initial_G / (units.J / units.m**2)))


# ***** Relaxation of crack slab  *****

# optionally, relax the slab, keeping top and bottom rows fixed
if hasattr(params, 'relax_slab') and params.relax_slab:
    print('Relaxing slab...')
    crack_slab.set_calculator(params.calc)
    opt = FIRE(crack_slab)
    opt.run(fmax=params.relax_fmax)

# Find initial position of crack tip
crack_pos = find_tip_stress_field(crack_slab, calc=params.calc)
print('Found crack tip at position %s' % crack_pos)

crack_slab.info['strain'] = strain
crack_slab.info['G'] = params.initial_G
crack_slab.info['CrackPos'] = crack_pos

# ******** Save output file **********

# Save results in extended XYZ format, including extra properties and info
print('Writing crack slab to file "crack.xyz"')
ase.io.write('crack.xyz', crack_slab, format='extxyz')
