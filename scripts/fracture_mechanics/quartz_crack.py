import os
import sys

import numpy as np

from ase.atoms import Atoms
from ase.io import read, write
from ase.calculators.neighborlist import NeighborList
from ase.units import GPa, J, m

from ase.optimize import FIRE

from matscipy.elasticity import measure_triclinic_elastic_constants
import matscipy.fracture_mechanics.crack as crack

sys.path.insert(0, '.')
import params

a = params.atoms.copy()

b = a * [6, 1, 1]
b.center()
b.positions[:,0] += 0.5
b.set_scaled_positions(b.get_scaled_positions())

mask = ((b.positions[:,0] > 0.) &
        (b.positions[:,0] < 2*a.cell[0,0]) &
        (b.positions[:,1] < (a.cell[1,1]/2.0 + params.final_height/2.0)) &
        (b.positions[:,1] > (a.cell[1,1]/2.0 - params.final_height/2.0)))

nl = NeighborList([1.0]*len(b),
                 self_interaction=False,
                 bothways=True)
nl.update(b)

term = Atoms()

if params.tetra:
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

if params.terminate:                
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
            d = r_ij/(params.eqm_bond_lengths[min(z1,z2), max(z1,z2)]*
                      params.eqm_bond_lengths[min(z1, 1), max(z1, 1)])
            t.translate(b.positions[i] + d)
            term += t

cryst = b[mask] + term

n = NeighborList([1.0]*len(cryst),
                 self_interaction=False,
                 bothways=True)
n.update(cryst)

cryst.set_scaled_positions(cryst.get_scaled_positions())
cryst.positions[:,0] += cryst.cell[0,0]/2. - cryst.positions[:,0].mean()
cryst.positions[:,1] += cryst.cell[1,1]/2. - cryst.positions[:,1].mean()
cryst.set_scaled_positions(cryst.get_scaled_positions())
cryst.center(params.vacuum, axis=0)
cryst.center(params.vacuum, axis=1)

print 'Elastic constants / GPa'
print (params.C/GPa).round(2)

crack = crack.CubicCrystalCrack(C11=None, C12=None, C44=None,
                                crack_surface=params.crack_surface,
                                crack_front=params.crack_front,
                                C=params.C)

k1g = crack.k1g(params.surface_energy)
print 'k1g', k1g

# Crack tip position.
tip_x = cryst.cell.diagonal()[0]/2
tip_y = cryst.cell.diagonal()[1]/2

k1 = 1.0

# Apply initial strain field.
c = cryst.copy()
ux, uy = crack.displacements(cryst.positions[:,0],
                             cryst.positions[:,1],
                             tip_x, tip_y, k1*k1g)
c.positions[:,0] += ux
c.positions[:,1] += uy

# Center notched configuration in simulation cell and ensure enough vacuum.
oldr = a[0].position.copy()
c.center(vacuum=params.vacuum, axis=0)
c.center(vacuum=params.vacuum, axis=1)
tip_x += c[0].position[0] - oldr[0]
tip_y += c[0].position[1] - oldr[1]

cryst.set_cell(c.cell)
cryst.translate(c[0].position - oldr)

write('cryst.xyz', cryst)
write('crack.xyz', c)

water = Atoms('H2O')
