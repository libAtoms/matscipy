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

import numpy as np
import matplotlib.pyplot as plt

import ase.units
import ase.io

from scipy.signal import argrelextrema
from scipy.interpolate import splrep, splev

J_per_m2 = ase.units.J/ase.units.m**2

bond_lengths1, bond1_forces1, epot1, epot_cluster1, work1, tip_x1, tip_z1 = \
  np.loadtxt('bond_1_eval.out', unpack=True)
bond_lengths2, bond2_forces2, epot2, epot_cluster2, work2, tip_x2, tip_z2 = \
  np.loadtxt('bond_2_eval.out', unpack=True)

minima1 = argrelextrema(epot1, np.less)
minima2 = argrelextrema(epot2, np.less)

maxima1 = argrelextrema(epot1, np.greater)
maxima2 = argrelextrema(epot2, np.greater)

Ea_1 = epot1[maxima1][0] - epot1[minima1][0]
dE_1 = epot1[minima1][1] - epot1[minima1][0]
print 'Ea_1 = %.3f eV' % Ea_1
print 'dE_1 = %.3f eV' % dE_1
print

Ea_2 = epot2[maxima2][0] - epot2[minima2][0]
dE_2 = epot2[minima2][1] - epot2[minima2][0]
print 'Ea_2 = %.3f eV' % Ea_2
print 'dE_2 = %.3f eV' % dE_2
print

E0_1 = epot1[minima1][0]
E0_2 = epot2[minima2][0]
E0_12 = epot1[minima1][-1] - epot2[minima1][0]

plt.figure(1)
plt.clf()

plt.plot(tip_x1, epot1 - E0_1, 'b.-', label='Bond 1')
plt.plot(tip_x2, epot2 - E0_1 + E0_12, 'c.-', label='Bond 2')

plt.plot(tip_x2 - tip_x2[minima2][0] + tip_x1[minima1][0],
         epot2 - E0_2, 'c.--', label='Bond 2, shifted')

plt.plot(tip_x1[minima1], epot1[minima1] - E0_1, 'ro', mec='r',
         label='Bond 1 minima')
plt.plot(tip_x2[minima2], epot2[minima1] - E0_1 + E0_12, 'mo', mec='m',
         label='Bond 2 minima')
plt.plot(tip_x2[minima2]-tip_x2[minima2][0] + tip_x1[minima1][0],
            epot2[minima1] - E0_1, 'mo', mec='m', mfc='w',
            label='Bond 2 minima, shifted')

plt.plot(tip_x1[maxima1], epot1[maxima1] - E0_1, 'rd', mec='r', label='Bond 1 TS')
plt.plot(tip_x2[maxima2], epot2[maxima1] - E0_1 + E0_12, 'md', mec='m', label='Bond 2 TS')
plt.plot(tip_x2[maxima2] - tip_x2[minima2][0] + tip_x1[minima1][0],
            epot2[maxima1] - E0_1, 'md', mec='m', mfc='w', label='Bond 2 TS, shifted')


plt.xlabel(r'Crack position / $\mathrm{\AA}$')
plt.ylabel(r'Potential energy / eV')
plt.legend(loc='upper right')
#plt.ylim(-0.05, 0.30)
plt.draw()


plt.figure(2)
#plt.clf()

bond_length, gamma = np.loadtxt('../cohesive-stress/gamma_bond_length.out', unpack=True)

s = bond_length - bond_length[0]
s1 = bond_lengths1 - bond_length[0]

surface = ase.io.read('../cohesive-stress/surface.xyz')
area = surface.cell[0,0]*surface.cell[2,2]
gamma_sp = splrep(bond_length, gamma)
E_surf = splev(bond_lengths1, gamma_sp)*area

plt.plot(s1, (epot1 - E0_1), '-', label='total energy')
plt.plot(s, E_surf, '-', label=r'surface energy')
plt.plot(s1, (epot1 - E0_1 - E_surf), '--', label='elastic energy')
#plt.xlim(2.35, 4.5)

plt.axhline(0, color='k')

plt.xlabel(r'Bond extension $s$ / $\mathrm{\AA}$')
plt.ylabel(r'Energy / eV/cell')
plt.legend(loc='upper right')
plt.draw()
