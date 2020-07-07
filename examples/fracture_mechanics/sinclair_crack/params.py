import numpy as np
from matscipy.fracture_mechanics.clusters import diamond, set_groups
import ase.io
import atomistica

# Interaction potential
calc = atomistica.KumagaiScr()

# Fundamental material properties
el              = 'Si'
a0              = 5.429
C11             = 166.   # GPa
C12             = 65.    # GPa
C44             = 77.    # GPa
surface_energy  = 1.08  * 10    # GPa*A = 0.1 J/m^2

# Crack system
n               = [ 10, 9, 1 ]
crack_surface   = [ 1, 1, 1 ]
crack_front     = [ 1, -1, 0 ]
skin_x, skin_y = 1.5, 1.5

vacuum          = 6.0

# Simulation control
nsteps          = 31
# Increase stress intensity factor
k1              = [1.0] #np.linspace(0.8, 1.5, nsteps)

fmax            = 1e-3

# Setup crack system and regions I, II and III
cryst = diamond(el, a0, n, crack_surface, crack_front)

set_groups(cryst, n, skin_x, skin_y)
regionIII = cryst.arrays['groups'] == 0
regionI_II = cryst.arrays['groups'] == 1

set_groups(cryst, n, 2*skin_x, 2*skin_y)
regionII_III = cryst.arrays['groups'] == 0
regionI = cryst.arrays['groups'] == 1

regionII = regionI_II & regionII_III

print('sum(region) I, II, III = ', sum(regionI), sum(regionII), sum(regionIII))

cryst.new_array('region', np.zeros(len(cryst), dtype=int))
cryst.arrays['region'][regionI] = 1
cryst.arrays['region'][regionII] = 2
cryst.arrays['region'][regionIII] = 3
del cryst.arrays['groups']

ase.io.write('cryst.cfg', cryst)

