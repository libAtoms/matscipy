# Fracture mechanics

## `matscipy-quasistatic-crack`

This command carries out a quasistatic K-field-controlled crack calculation on a small cluster. The outer boundaries of
the cluster are fixed according to the near field solution of linear elastic fracture mechanics, considering elastic
anisotropy, as described in [Sih, Paris, Irwin, Int. J. Fract. Mech. 1, 189 (1965)](https://doi.org/10.1007/BF00186854).
The crack advances by stepwise updates of $K_\textrm{I}$ and and crack tip position.

The command does not take any arguments but can be configured using a `params.py` file in the current directory.
An example `params.py` file follows:
```Python
import numpy as np

import ase.io

from matscipy.fracture_mechanics.clusters import diamond, set_groups

from atomistica import TersoffScr, Tersoff_PRB_39_5566_Si_C__Scr

###
# Interaction potential
calc = TersoffScr(**Tersoff_PRB_39_5566_Si_C__Scr)

# Fundamental material properties
el = 'C'
a0 = 3.57

# Elastic constants: either specify explicitly or compute automatically
compute_elastic_constants = True
# C11 = 1220.   # GPa
# C12 = -3.    # GPa
# C44 = 535.    # GPa

# Surface energy
surface_energy = 2.7326 * 10  # GPa*A = 0.1 J/m^2

# Crack system
crack_surface = [1, 1, 1]  # Normal of crack face
crack_front = [1, -1, 0]  # Direction of crack front
# bond = (10, 11)
bondlength = 1.7  # Bond length for crack tip detection
bulk_nn = 4  # Number of nearest neighbors in the bulk

vacuum = 6.0  # Vacuum surrounding the cracked cluster

# Simulation control
nsteps = 31
# Increase stress intensity factor
k1 = np.linspace(0.8, 1.2, nsteps)
# Don't move crack tip
tip_dx = np.zeros_like(k1)
tip_dz = np.zeros_like(k1)

fmax = 0.05  # Tolerance for quasistatic optimizer

# The actual crack system
n = [1, 1, 1]
skin_x, skin_y = 1, 1
cryst = diamond(el, a0, n, crack_surface, crack_front)
set_groups(cryst, n, skin_x, skin_y)  # Outer fixed atoms
ase.io.write('cryst.xyz', cryst, format='extxyz')  # Dump initial crack system (without notch)
```

## `matscipy-sinclair-continuation`

## `matscipy-sinclair-crack`