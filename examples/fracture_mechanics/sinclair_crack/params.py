#
# Copyright 2020 James Kermode (Warwick U.)
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

circular_regions = True

# circular regions I-II-III-IV
if circular_regions:
    r_I = 15.0
    cutoff = 6.0
    r_III = 40.0

    n = [2 * int((r_III + cutoff)/ a0), 2 * int((r_III + cutoff)/ a0) - 1, 1]
    print('n=', n)

    # Setup crack system and regions I, II, III and IV
    cryst = diamond(el, a0, n, crack_surface, crack_front)
    cluster = set_regions(cryst, r_I, cutoff, r_III)  # carve circular cluster

else:
    # boxy regions, with fixed dimension n
    n = [20, 19, 1]
    R_III = 4
    R_II = 4

    cryst = diamond(el, a0, n, crack_surface, crack_front)

    set_groups(cryst, n, R_III, R_III)
    regionIII = cryst.arrays['groups'] == 0
    regionI_II = cryst.arrays['groups'] == 1

    set_groups(cryst, n, R_II + R_III, R_II + R_III)
    regionII_III = cryst.arrays['groups'] == 0
    regionI = cryst.arrays['groups'] == 1

    regionII = regionI_II & regionII_III

    print(sum(regionI), sum(regionII), sum(regionIII))

    cryst.new_array('region', np.zeros(len(cryst), dtype=int))
    cryst.arrays['region'][regionI] = 1
    cryst.arrays['region'][regionII] = 2
    cryst.arrays['region'][regionIII] = 3

    del cryst.arrays['groups']
    cluster = cryst  # cluster and crystal only differ by PBC

ase.io.write('cryst.cfg', cryst)
ase.io.write('cluster.cfg', cluster)
