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

print 'Fixed %d atoms' % (g==0).sum()

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

write('cryst.xyz', cryst, format='extxyz')
write('crack.xyz', c, format='extxyz')

