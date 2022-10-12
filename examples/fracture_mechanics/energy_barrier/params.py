#
# Copyright 2015 James Kermode (Warwick U.)
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

from matscipy.fracture_mechanics.clusters import diamond_111_110

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
n               = [ 6, 4, 1 ]
crack_surface   = [ 1, 1, 1 ]
crack_front     = [ 1, -1, 0 ]

vacuum          = 6.0

# Simulation control
k1              = 0.900
bond_lengths    = np.linspace(2.5, 4.5, 21)

fmax            = 0.001

cutoff          = 5.0

# Setup crack system
cryst = diamond_111_110(el, a0, n, crack_surface, crack_front)
ase.io.write('cryst.cfg', cryst)

optimize_tip_position = True

basename = 'k_%04d' % int(k1*1000.0)
