import os

import numpy as np

from ase.atoms import Atoms
from ase.io import read, write
from ase.calculators.neighborlist import NeighborList

import matscipy.fracture_mechanics.crack as crack

slab_height = 25.0
final_height = 18.0
vacuum = 10.0

crack_surface = [1, 0, 1]
crack_direction = [0,1,0]

if not os.path.exists('slab.xyz'):
    from quippy.structures import rotation_matrix
    from quippy.surface import orthorhombic_slab, alpha_quartz, quartz_params
    
    #aq = alpha_quartz(**quartz_params['CASTEP_GGA'])
    # alpha-quartz unit cell with DFT (GGA, USPP) lattice constants:
    #  a=5.02836, c=5.51193, u=0.48128, x=0.41649, y=0.24661, z=0.13594
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
       [-0.19291303, -1.8136838 , -2.58660176]])),

    slab = orthorhombic_slab(aq,
                             rot=rotation_matrix(aq, crack_surface, crack_direction),
                             periodicity=[0.0, slab_height, 0.0],
                             vacuum=[0.0, vacuum, 0.0], verbose=True)
    write('slab.xyz', slab)

a = read('slab.xyz')

b = a * [6, 1, 1]
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

eqm_bond_lengths = {(8, 14): 1.60,
                    (1, 8):  1.10}
    
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

n = NeighborList([1.0]*len(cryst),
                 self_interaction=False,
                 bothways=True)
n.update(cryst)

cryst.set_scaled_positions(cryst.get_scaled_positions())
cryst.positions[:,0] += cryst.cell[0,0]/2. - cryst.positions[:,0].mean()
cryst.positions[:,1] += cryst.cell[1,1]/2. - cryst.positions[:,1].mean()
cryst.set_scaled_positions(cryst.get_scaled_positions())
cryst.center(vacuum, axis=0)
cryst.center(vacuum, axis=1)

from ase.units import GPa, J, m

C11 = 89.1*GPa
C12 = -7.82*GPa
C44 = 49.1*GPa

crack = crack.CubicCrystalCrack(C11, C12, C44,
                                crack_surface,
                                crack_direction)

surface_energy = 0.161*(J/m**2)*10

k1g = crack.k1g(surface_energy)
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
c.center(vacuum=vacuum, axis=0)
c.center(vacuum=vacuum, axis=1)
tip_x += c[0].position[0] - oldr[0]
tip_y += c[0].position[1] - oldr[1]

cryst.set_cell(c.cell)
cryst.translate(c[0].position - oldr)

write('cryst.xyz', cryst)
write('crack.xyz', c)
