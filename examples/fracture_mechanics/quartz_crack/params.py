import os

import numpy as np

from ase.atoms import Atoms
from ase.io import read, write
from ase.units import GPa, J, m

slab_height = 25.0
final_height = 18.0
vacuum = 10.0

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

