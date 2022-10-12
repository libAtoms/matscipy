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
import os

import numpy as np

from ase.atoms import Atoms
from ase.io import read, write
from ase.calculators.neighborlist import NeighborList
from ase.units import GPa, J, m

nx = 8
slab_height = 35.0
final_height = 27.0
vacuum = 10.0
tetra = True
terminate = False
skin = 4.0
k1 = 1.0
bond = (190, 202)
bond_lengths = np.linspace(1.6, 2.5, 11)
fmax = 0.1

crack_surface = [1, 0, 1]
crack_front = [0,1,0]

aq = Atoms(symbols='Si3O6',
           pbc=True,
           cell=np.array(
    [[ 2.51418  , -4.3546875,  0.       ],
     [ 2.51418  ,  4.3546875,  0.       ],
     [ 0.       ,  0.       ,  5.51193  ]]),
           positions=np.array(
  [[ 1.21002455, -2.095824  ,  3.67462   ],
   [ 1.21002455,  2.095824  ,  1.83731   ],
   [-2.4200491 ,  0.        , -0.        ],
   [ 1.66715276, -0.73977431,  4.42391176],
   [-0.19291303,  1.8136838 ,  8.09853176],
   [-1.47423973, -1.07390948,  6.26122176],
   [ 1.66715276,  0.73977431, -4.42391176],
   [-1.47423973,  1.07390948, -0.74929176],
   [-0.19291303, -1.8136838 , -2.58660176]]))

if not os.path.exists('slab.xyz'):
    from quippy.structures import rotation_matrix
    from quippy.surface import orthorhombic_slab
    from quippy.atoms import Atoms as QuippyAtoms
    aq = QuippyAtoms(aq)
    slab = orthorhombic_slab(aq,
                             rot=rotation_matrix(aq, crack_surface, crack_front),
                             periodicity=[0.0, slab_height, 0.0],
                             vacuum=[0.0, vacuum, 0.0], verbose=True)
    write('slab.xyz', slab)

atoms = read('slab.xyz')

eqm_bond_lengths = {(8, 14): 1.60,
                    (1, 8):  1.10}

# Quartz elastic constants computed with DFT (PBE-GGA)
C = np.zeros((6,6))
C[0,0] = C[1,1] = 87.1*GPa
C[0,1] = C[1,0] = -7.82*GPa
C[0,2] = C[2,0] = C[0,3] = C[3,0] = 6.30*GPa
C[0,3] = C[3,0] = -17.0*GPa
C[1,3] = C[3,1] = -C[0,3]
C[4,5] = C[0,3]
C[5,4] = C[0,3]
C[2,2] = 87.1*GPa
C[3,3] = 49.1*GPa
C[4,4] = 49.1*GPa
C[5,5] = 47.5*GPa
    
# Surface energy computed with DFT (PBE-GGA)
surface_energy = 0.161*(J/m**2)*10

a = atoms.copy()
b = a * [nx, 1, 1]
b.center()
b.positions[:,0] += 0.5
b.set_scaled_positions(b.get_scaled_positions())

mask = ((b.positions[:,0] > 0.) &
        (b.positions[:,0] < 2*a.cell[0,0]) &
        (b.positions[:,1] < (a.cell[1,1]/2.0 + final_height/2.0)) &
        (b.positions[:,1] > (a.cell[1,1]/2.0 - final_height/2.0)))

nl = NeighborList([1.0]*len(b),
                 self_interaction=False,
                 bothways=True)
nl.update(b)

term = Atoms()

if tetra:
    for i in range(len(b)):
        if not mask[i]:
            continue
        indices, offsets = nl.get_neighbors(i)
        if b.numbers[i] == 8 and mask[indices].sum() == 0:
            mask[i] = False
        if b.numbers[i] == 14:
            # complete tetrahedra
            for (j, o) in zip(indices, offsets):        
                if b.numbers[j] == 8:
                    mask[j] = True

if terminate:
    for i in range(len(b)):
        if not mask[i]:
            continue    
        indices, offsets = nl.get_neighbors(i)    
        for (j, o) in zip(indices, offsets):
            if mask[j]:
                continue

            # i is IN, j is OUT, we need to terminate cut bond ij
            z1 = b.numbers[i]
            z2 = b.numbers[j]
            print 'terminating %d-%d bond (%d, %d)' % (z1, z2, i, j)
            if z1 != 8:
                raise ValueError('all IN term atoms must be O')        

            r_ij = (b.positions[j] + np.dot(o, b.cell) - b.positions[i])

            t = Atoms('H')
            d = r_ij/(eqm_bond_lengths[min(z1,z2), max(z1,z2)]*
                      eqm_bond_lengths[min(z1, 1), max(z1, 1)])
            t.translate(b.positions[i] + d)
            term += t

cryst = b[mask] + term

# calculator
from quippy import Potential
calc = Potential('IP TS', param_filename='ts_params.xml')

cryst.set_calculator(calc)
cryst.get_potential_energy() # obtain reference dipole moments

cryst.set_scaled_positions(cryst.get_scaled_positions())
cryst.positions[:,0] += cryst.cell[0,0]/2. - cryst.positions[:,0].mean()
cryst.positions[:,1] += cryst.cell[1,1]/2. - cryst.positions[:,1].mean()

cryst.set_scaled_positions(cryst.get_scaled_positions())
cryst.center(vacuum, axis=0)
cryst.center(vacuum, axis=1)

# fix atoms near outer boundaries
r = cryst.get_positions()
minx = r[:, 0].min() + skin
maxx = r[:, 0].max() - skin
miny = r[:, 1].min() + skin
maxy = r[:, 1].max() - skin
g = np.where(
    np.logical_or(
        np.logical_or(
            np.logical_or(
                r[:, 0] < minx, r[:, 0] > maxx),
            r[:, 1] < miny),
        r[:, 1] > maxy),
    np.zeros(len(cryst), dtype=int),
    np.ones(len(cryst), dtype=int))
cryst.set_array('groups', g)

# zero dipole moments on outer boundaries
#cryst.set_array('fixdip', np.logical_not(g))
#cryst.set_array('dipoles', calc.atoms.arrays['dipoles'])
#cryst.arrays['dipoles'][g==0, :] = 0.0

write('cryst.xyz', cryst, format='extxyz')
