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

from matscipy.fracture_mechanics.clusters import diamond, set_groups

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
n               = [ 4, 6, 1 ]
crack_surface   = [ 1,-1, 0 ]
crack_front     = [ 1, 1, 0 ]
crack_tip       = [ 41, 56 ]
skin_x, skin_y = 1, 1

vacuum          = 6.0

# Simulation control
nsteps          = 31
# Increase stress intensity factor
k1              = np.linspace(0.8, 1.5, nsteps)
# Don't move crack tip
tip_dx          = np.zeros_like(k1)
tip_dz          = np.zeros_like(k1)

fmax            = 0.05

# Setup crack system
cryst = diamond(el, a0, n, crack_surface, crack_front)
set_groups(cryst, n, skin_x, skin_y)

ase.io.write('cryst.cfg', cryst)

# Compute crack tip position
r0 = np.sum(cryst.get_positions()[crack_tip,:], axis=0)/len(crack_tip)
tip_x0, tip_y0, tip_z0 = r0
