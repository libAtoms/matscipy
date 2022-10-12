#
# Copyright 2014 James Kermode (Warwick U.)
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
import glob
import sys

import numpy as np

import ase.io
import ase.units

import matplotlib.pyplot as plt

###

sys.path += [ "." ]
import params

###

J_per_m2 = ase.units.J/ase.units.m**2

separation = []
bond_length = []
energy = []

for fn in sorted(glob.glob('separation_*.xyz')):
    a = ase.io.read(fn)
    separation += [ a.info['separation'] ]
    bond_length += [ a.info['bond_length'] ]
    energy += [ a.get_potential_energy() ]

separation = np.array(separation)
bond_length = np.array(bond_length)
energy = np.array(energy)

cryst = ase.io.read('cryst.xyz')
surface = ase.io.read('surface.xyz')
e_per_atom_bulk = cryst.get_potential_energy()/len(cryst)
area = surface.get_volume() / surface.cell[1, 1]
gamma = ((energy - e_per_atom_bulk * len(surface)) / (2.0 * area))
gamma_relaxed = ((surface.get_potential_energy() -
                  e_per_atom_bulk * len(surface)) / (2.0 * area))

print 'gamma unrelaxed', gamma[-1]/J_per_m2, 'J/m^2'
print 'gamma relaxed  ', gamma_relaxed/J_per_m2, 'J/m^2'

plt.clf()
plt.subplot(211)
dE_db = ((energy[1:] - energy[:-1])/(bond_length[1:] - bond_length[:-1]))/area
dE_db /= ase.units.GPa
plt.plot((bond_length[1:] + bond_length[:-1])/2., dE_db, 'b-')
plt.ylabel(r'Cohesive stress / GPa')
plt.xlim(bond_length[0], bond_length[-1])

plt.subplot(212)
plt.plot(bond_length, gamma/J_per_m2, 'b-')
plt.axhline(gamma_relaxed/J_per_m2, linestyle='--')
plt.xlabel(r'Bond length / $\mathrm{\AA}$')
plt.ylabel(r'Energy / J/m^2')
plt.xlim(bond_length[0], bond_length[-1])
plt.ylim(0, 1.0)
plt.draw()

np.savetxt('gamma_bond_length.out', np.c_[bond_length, gamma])
