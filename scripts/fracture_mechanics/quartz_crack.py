import os

import numpy as np

from ase.io import read, write
from ase import Atoms
from ase.calculators.neighborlist import NeighborList

import matscipy.fracture_mechanics.crack as crack

from quippy.structures import rotation_matrix
from quippy.surface import orthorhombic_slab, alpha_quartz, quartz_params

def hydrolyse(atoms):
    n = NeighborList([1.0]*len(atoms),
                     self_interaction=False,
                     bothways=True)
    n.update(atoms)
    
    m1 = Atoms('H', [[0, 0, 0]])
    m2 = Atoms('OH', [[0, 0, 0], [0.8, 0.6, 0.0]])

    for i in range(len(atoms)):
        indices, offsets = n.get_neighbors(i)
        nn = len(indices)

        if atoms.numbers[i] == 14 and nn == 3:
            print 'Saturating atom', i, 'with -OH'
            p = atoms.positions[i]
            oh = m2.copy()
            oh.positions[1,1] *= np.sign(p[1])
            oh.positions[:,1] += 1.5*np.sign(p[1])
            oh.translate(p)            
            atoms += oh

        if atoms.numbers[i] == 8 and nn == 1:
            print 'Saturating atom', i, 'with -H'
            p = atoms.positions[i]
            h = m1.copy()
            h.positions[:,1] += 1.1*np.sign(p[1])
            h.translate(p)
            atoms += h


initial_height = 25.0
final_height = 20.0
vacuum = 10.0

crack_surface = [1, 0, 1]
crack_direction = [0,1,0]

if not os.path.exists('slab.xyz'):
    aq = alpha_quartz(**quartz_params['CASTEP_GGA'])
    slab = orthorhombic_slab(aq,
                             rot=rotation_matrix(aq, crack_surface, crack_direction),
                             periodicity=[0.0, height, 0.0],
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
    if b.numbers[i] == 8 and len(indices) == 1:
        mask[i] = True
        
    for (j, o) in zip(indices, offsets):
        if b.numbers[i] == 14 and b.numbers[j] == 8 and not mask[j]:
            mask[j] = True
        if mask[j]:
            continue
        # i is IN, j is OUT, we need to terminate cut bond ij
        r_ij = (b.positions[j] + np.dot(o, b.cell) - b.positions[i])
        u_ij = r_ij/np.sqrt((r_ij**2).sum())

        z1 = b.numbers[i]
        z2 = b.numbers[j]
        if z1 == 8:
            t = Atoms('H',  [[0, 0, 0]])
        elif z1 == 14:
            t = Atoms('OH', [[0, 0, 0], u_ij*eqm_bond_lengths[(1, 8)]])

        z3 = t.numbers[0]
        d = r_ij/(eqm_bond_lengths[min(z1,z2), max(z1,z2)]*
                  eqm_bond_lengths[min(z1,z3), max(z1,z3)])
        
        t.translate(b.positions[i] + d)
        term += t
        
cryst = b[mask] + term
width = cryst.positions[:,0].max() - cryst.positions[:,0].min()
height = cryst.positions[:,1].max() - cryst.positions[:,1].min()
cell = np.zeros((3,3))
cell[0,0] = width + vacuum
cell[1,1] = height + vacuum
cell[2,2] = cryst.cell[2,2]
cryst.set_cell(cell)
#cryst.set_scaled_positions(cryst.get_scaled_positions())

from ase.units import GPa, J, m

C11 = 89.1*GPa
C12 = -7.82*GPa
C44 = 49.1*GPa

crack = crack.CubicCrystalCrack(C11, C12, C44,
                                crack_surface,
                                crack_direction)

surface_energy = 0.161/(J/m**2)*10

k1g = crack.k1g(surface_energy)
print 'k1g', k1g

# Crack tip position.
tip_x = cryst.cell.diagonal()[0]/2
tip_y = cryst.cell.diagonal()[1]/2

k1 = 1.0

# Apply initial strain field.
a = cryst.copy()
ux, uy = crack.displacements(cryst.positions[:,0],
                             cryst.positions[:,1],
                             tip_x, tip_y, k1*k1g)
a.positions[:,0] += ux
a.positions[:,1] += uy

# Center notched configuration in simulation cell and ensure enough vacuum.
oldr = a[0].position.copy()
a.center(vacuum=vacuum, axis=0)
a.center(vacuum=vacuum, axis=1)
tip_x += a[0].position[0] - oldr[0]
tip_y += a[0].position[1] - oldr[1]

#cryst.set_cell(a.cell)
#cryst.translate(a[0].position - oldr)


