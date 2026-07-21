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

## `matscipy-sinclair-crack`

This command carries out a relaxation of an atomistic fracture system with flexible boundary conditions, following the approach of [Sinclair (1975)](https://www.tandfonline.com/doi/abs/10.1080/14786437508226544).
It reads a `params.py` which is similar to that used for `matscipy-quasi-static-fracture` described above.

An example parameter file for silicon using the screened Kumagai potential from Atomistica is given below:

```Python
import numpy as np
from matscipy.fracture_mechanics.clusters import diamond, set_groups, set_regions
import ase.io
import atomistica

# Interaction potential
calc = atomistica.KumagaiScr()

# Fundamental material properties
el = 'Si'
a0 = 5.429
surface_energy = 1.08 * 10  # GPa*A = 0.1 J/m^2

elastic_symmetry = 'cubic'

# Crack system
crack_surface = [1, 1, 1]
crack_front = [1, -1, 0]

vacuum = 6.0

# Simulation control
ds = 0.01
nsteps = 10000
continuation = True

k0 = 0.9
dk = 1e-4

fmax = 1e-2
max_steps = 50

r_I = 15.0
cutoff = 6.0
r_III = 40.0

n = [2 * int((r_III + cutoff)/ a0), 2 * int((r_III + cutoff)/ a0) - 1, 1]
print('n=', n)

# Setup crack system and regions I, II, III and IV
cryst = diamond(el, a0, n, crack_surface, crack_front)
cluster = set_regions(cryst, r_I, cutoff, r_III)  # carve circular cluster

ase.io.write('cryst.cfg', cryst)
ase.io.write('cluster.cfg', cluster)
```

## `matscipy-sinclair-continuation`

This command build on the functionality of `matscipy-sinclair-crack` by carrying out numerical continuation
to map out a solution path containing both stable and unstable equilibria, following the approach described
in [Buze and Kermode (2021)](http://dx.doi.org/10.1103/PhysRevE.103.033002).

The `params.py` parameter file given above for `matscipy-sinclair-crack` is also suitable as an example
input file for this script.