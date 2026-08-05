#
# Copyright 2016 James Kermode (Warwick U.)
#           2014 Lars Pastewka (U. Freiburg)
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

from math import log

import numpy as np

import ase.io

from matscipy.fracture_mechanics.clusters import diamond, set_groups, set_regions

import atomistica

###

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
crack_surface   = [ 1,-1, 0 ]
crack_front     = [ 1, 1, 0 ]
n               = [ 4, 6, 1 ] # Required only for rectangular clusters
crack_tip       = [ 41, 56 ] # Set to None for default: centre of cell.
skin_x, skin_y = 1, 1 # Required only for rectangular clusters

vacuum          = 6.0

# Cyclindrical cluster dimensions
cylcell = True
r_I = 50 
cutoff = 15

# Simulation control
nsteps          = 31
# Increase stress intensity factor
k1              = np.linspace(0.8, 1.5, nsteps)
# Don't move crack tip
tip_dx          = np.zeros_like(k1)
tip_dz          = np.zeros_like(k1)

fmax            = 0.05

# Setup crystal
if cylcell:
    # set r_II & r_IV equal to r_II, as these regions are unnecessary for quasistatic crack simulations
    r_II = r_I + cutoff ; r_III = r_II ; r_IV = r_II
    # include padding to stop atoms leaving box during simulation
    padding = 3 
    # compute size of simulation cell based on input radii
    [ax,ay,az] = diamond(el, a0, [1,1,1], crack_surface, crack_front).cell.lengths()
    n = [2*(int(np.ceil(r_II / ax))+padding), 2*(int(np.ceil(r_II / ay))+padding), 1]
cryst = diamond(el, a0, n, crack_surface, crack_front)
ase.io.write('cryst.cfg', cryst)

# Setup cell shape and regions
if cylcell:
    # setup cylindrical cluster
    cluster = set_regions(cryst, r_I, cutoff, r_III, r_IV)
    regions = cluster.arrays['region']
    groups = np.zeros_like(regions)
    groups[regions == 1] = 1
    cluster.set_array('groups', groups)
    del cluster.arrays['region']
else:
    # setup rectangular cluster
    cluster = cryst.copy()
    set_groups(cluster, n, skin_x, skin_y)

ase.io.write('cluster.cfg', cluster)

# Compute crack tip position
if cylcell or crack_tip is None:
    r0 = np.array(cryst.cell.diagonal())/2 # center of cell
else:
    r0 = np.sum(cryst.get_positions()[crack_tip,:], axis=0)/len(crack_tip)
tip_x, tip_y, tip_z = r0
